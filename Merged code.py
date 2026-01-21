import cv2
import numpy as np
import time
from collections import deque

# ---------------- CONFIG ----------------
PI_BALL_HSV_LOW  = (85, 60, 40)
PI_BALL_HSV_HIGH = (140, 255, 255)

WHITE_V_MIN = 180
WHITE_S_MAX = 80
BLACK_V_MAX = 80

MIN_PLAYER_AREA = 80
MAX_PLAYER_AREA = 5000

PIXELS_PER_CM = 6

FRAME_W = 800   # reduced resolution for performance
FRAME_H = 600
FRAME_SKIP = 2  # process every 2nd frame

# ---------------- STATE ----------------
possession_counts = {"white": 0, "black": 0}
prev_ball_pos = None
prev_time = None
last_print_time = 0
PRINT_INTERVAL = 1.0
speed_queue = deque(maxlen=5)
frame_counter = 0

INVERSE_RODS = False  # set True if Pi camera mapping is flipped

# ---------------- BALL DETECTION ----------------
ball_confidence = 0
def detect_ball_pi(hsv, last_ball_pos=None):
    global ball_confidence
    mask = cv2.inRange(hsv, np.array(PI_BALL_HSV_LOW), np.array(PI_BALL_HSV_HIGH))
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 2500: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4*np.pi*area/(perimeter**2)
        if circularity < 0.6: continue
        (x,y), r = cv2.minEnclosingCircle(cnt)
        if r < 5 or r > 20: continue
        score = circularity*area
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))

    # Require 2 consecutive detections for confidence
    if best_center:
        if last_ball_pos and np.linalg.norm(np.array(best_center)-np.array(last_ball_pos))<30:
            ball_confidence += 1
        else:
            ball_confidence = 1
    else:
        ball_confidence = 0

    if ball_confidence >= 2:
        return best_center
    return None

# ---------------- PLAYER DETECTION ----------------
def detect_players(hsv, gray):
    players = []
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    white_mask = ((v>=WHITE_V_MIN) & (s<=WHITE_S_MAX)).astype(np.uint8)*255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    _, black_mask = cv2.threshold(gray, BLACK_V_MAX, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    for mask,label in [(white_mask,"white"),(black_mask,"black")]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_PLAYER_AREA<area<MAX_PLAYER_AREA:
                M = cv2.moments(c)
                if M["m00"]==0: continue
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                x,y,w,h = cv2.boundingRect(c)
                players.append({"center":(cx,cy),"bbox":(x,y,w,h),"color":label})
    return players

# ---------------- NEAREST PLAYER ----------------
def nearest_player_to_ball(players, ball_center):
    if ball_center is None or not players: return None,None
    bx,by = ball_center
    best = None
    best_dist = float('inf')
    for p in players:
        x,y,w,h = p["bbox"]
        fx = x + w//2
        fy = y + h
        d = ((fx-bx)**2 + (fy-by)**2)**0.5
        if d < best_dist:
            best_dist = d
            best = p
    return best,best_dist

# ---------------- SPEED ESTIMATION ----------------
def estimate_speed(ball_center):
    global prev_ball_pos, prev_time, speed_queue
    if ball_center is None:
        speed_queue.append(0)
        return 0
    now = time.time()
    speed = 0
    if prev_ball_pos is not None:
        dx = ball_center[0]-prev_ball_pos[0]
        dy = ball_center[1]-prev_ball_pos[1]
        dist_px = (dx**2+dy**2)**0.5
        dt = now-prev_time
        if dt>0:
            speed_cm_s = (dist_px/PIXELS_PER_CM)/dt
            speed = speed_cm_s if speed_cm_s>0.5 else 0
    prev_ball_pos = ball_center
    prev_time = now
    speed_queue.append(speed)
    return sum(speed_queue)/len(speed_queue)

# ---------------- PREPROCESS ----------------
def preprocess(frame,last_ball_pos=None):
    # Resize for performance
    frame = cv2.resize(frame,(FRAME_W,FRAME_H),interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    return hsv,gray,frame

# ---------------- DYNAMIC RODS ----------------
def detect_rods_dynamic(frame,angle_tol=15):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
    rod_ys = []
    if lines is not None:
        horizontal_lines=[]
        for l in lines:
            x1,y1,x2,y2 = l[0]
            angle = np.degrees(np.arctan2(y2-y1,x2-x1))
            if abs(angle)<=angle_tol:
                horizontal_lines.append(l[0])
        horizontal_lines.sort(key=lambda l: ((l[2]-l[0])**2+(l[3]-l[1])**2)**0.5, reverse=True)
        for line in horizontal_lines[:8]:
            y_center = (line[1]+line[3])//2
            rod_ys.append(y_center)
    rod_ys.sort()
    return rod_ys

def assign_players_to_rods(players, rod_ys):
    if not rod_ys: return players
    rod_ys_sorted = sorted(rod_ys)
    for p in players:
        px,py = p["center"]
        closest_rod = min(rod_ys_sorted,key=lambda y: abs(y-py))
        p["rod_y"] = closest_rod
    return players

def get_player_color_by_rod(rod_index):
    white_rods = [3,5,7,8]
    if INVERSE_RODS: white_rods = [i for i in range(1,9) if i not in white_rods]
    return "white" if rod_index in white_rods else "black"

# ---------------- PROCESS FRAME ----------------
def process_frame(frame,last_ball_pos=None):
    global possession_counts,last_print_time,frame_counter
    frame_counter +=1
    if frame_counter%FRAME_SKIP!=0: return None

    hsv,gray,frame = preprocess(frame,last_ball_pos)
    ball_center = detect_ball_pi(hsv,last_ball_pos)
    players = detect_players(hsv,gray)

    rod_ys = detect_rods_dynamic(frame)
    if rod_ys: players = assign_players_to_rods(players,rod_ys)

    for idx,p in enumerate(players):
        rod_idx = min(idx+1,8)
        p["color"] = get_player_color_by_rod(rod_idx)

    nearest,dist = nearest_player_to_ball(players,ball_center)
    possession_code = 0
    if nearest:
        possession_code = 1 if nearest["color"]=="white" else 2
        possession_counts[nearest["color"]] +=1

    speed = estimate_speed(ball_center)

    now = time.time()
    if now - last_print_time >= PRINT_INTERVAL:
        total = possession_counts["white"]+possession_counts["black"]
        white_pct = (possession_counts["white"]/total*100) if total else 0
        black_pct = (possession_counts["black"]/total*100) if total else 0
        print(f"Speed: {speed:.2f} cm/s | Possession: {possession_code} | White: {white_pct:.1f}% | Black: {black_pct:.1f}%")
        last_print_time = now

    # ---- Mask visualization ----
    ball_mask = np.zeros_like(gray)
    if ball_center:
        color_val = 255 if possession_code==1 else 180 if possession_code==2 else 100
        cv2.circle(ball_mask,ball_center,7,color_val,-1)
    rod_mask = np.zeros_like(gray)
    for y in rod_ys: cv2.line(rod_mask,(0,y),(frame.shape[1],y),255,3)
    field_mask = cv2.inRange(hsv,np.array([30,30,30]),np.array([90,255,255]))
    field_mask = cv2.morphologyEx(field_mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    combined_mask = cv2.bitwise_or(field_mask,ball_mask)
    combined_mask = cv2.bitwise_or(combined_mask,rod_mask)
    cv2.imshow("Field Mask",combined_mask)
    cv2.waitKey(1)

# ---------------- RUN IMAGE ----------------
def run_image(path):
    img = cv2.imread(path)
    if img is None: print("Error opening image",path); return
    last_ball_pos = None
    while True:
        process_frame(img,last_ball_pos)
        k = cv2.waitKey(0) & 0xFF
        if k==27: break
    cv2.destroyAllWindows()

# ---------------- RUN PI CAMERA ----------------
def run_pi_camera():
    try: from picamera2 import Picamera2
    except ImportError: print("picamera2 not available"); return
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size":(FRAME_W,FRAME_H),"format":"BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    last_ball_pos = None
    print("Pi Camera started. Press CTRL+C to stop.")
    try:
        while True:
            frame = picam2.capture_array()
            process_frame(frame,last_ball_pos)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__=="__main__":
    print("Choose mode:\n1=Image\n2=Pi Camera")
    choice = input("Mode: ")
    if choice=="1": run_image(r"C:\Users\Deepika\OneDrive\Documents\Deepika\balltracker2.jpg")
    elif choice=="2": run_pi_camera()
    else: print("Invalid choice")
