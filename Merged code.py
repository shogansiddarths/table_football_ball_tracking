import cv2
import numpy as np
import time
from collections import deque
from picamera2 import Picamera2


# ===================== CONSTANTS =====================

WHITE_V_MIN = 180
WHITE_S_MAX = 80
BLACK_V_MAX = 80

MIN_PLAYER_AREA = 200
MAX_PLAYER_AREA = 5000

PIXELS_PER_CM = 6

FRAME_W = 1280
FRAME_H = 720

POSSESSION_HOLD_TIME = 0.3
PRINT_INTERVAL = 1.0

# ===================== GLOBAL STATE =====================

possession_counts = {"white": 0, "black": 0}

prev_ball_pos = None
prev_time = None

current_possession = None
possession_start_time = None

speed_buffer = deque(maxlen=5)

last_print_time = 0

# ===================== BALL DETECTION =====================

def detect_ball_pi(hsv):
    lower = np.array([85, 60, 40])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_score = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 80 or area > 2500:
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circularity = 4*np.pi*area/(peri*peri)
        if circularity < 0.6:
            continue

        (x,y), r = cv2.minEnclosingCircle(c)
        if r < 5 or r > 20:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))

    return best_center

# ===================== PLAYER DETECTION =====================

def detect_players(hsv, gray):
    players = []

    v = hsv[:,:,2]
    s = hsv[:,:,1]

    white_mask = ((v >= WHITE_V_MIN) & (s <= WHITE_S_MAX)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    _, black_mask = cv2.threshold(gray, BLACK_V_MAX, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    for mask, label in [(white_mask,"white"), (black_mask,"black")]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (MIN_PLAYER_AREA < area < MAX_PLAYER_AREA):
                continue

            x,y,w,h = cv2.boundingRect(c)

            # ---- SHAPE FILTERS ----
            if h < 20:
                continue

            aspect_ratio = h / float(w + 1)
            if aspect_ratio < 1.3:
                continue

            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = 4*np.pi*area/(peri*peri)
                if circularity > 0.7:
                    continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            players.append({
                "center": (cx,cy),
                "bbox": (x,y,w,h),
                "color": label
            })

    return players

# ===================== NEAREST PLAYER =====================

def nearest_player_to_ball(players, ball_center):
    if ball_center is None:
        return None

    bx,by = ball_center
    best = None
    best_dist = 1e9

    for p in players:
        x,y,w,h = p["bbox"]
        fx = x + w//2

        fy_top = y
        fy_bottom = y + h

        d = min(
            ((fx-bx)**2 + (fy_top-by)**2)**0.5,
            ((fx-bx)**2 + (fy_bottom-by)**2)**0.5
        )

        if d < best_dist:
            best_dist = d
            best = p

    return best

# ===================== SPEED =====================

def estimate_speed(ball_center):
    global prev_ball_pos, prev_time

    if ball_center is None:
        return 0

    now = time.time()
    speed = 0

    if prev_ball_pos is not None:
        dx = ball_center[0] - prev_ball_pos[0]
        dy = ball_center[1] - prev_ball_pos[1]
        dist_px = (dx*dx + dy*dy)**0.5
        dt = now - prev_time
        if dt > 0:
            speed = (dist_px/PIXELS_PER_CM)/dt

    prev_ball_pos = ball_center
    prev_time = now
    return speed

# ===================== FRAME PROCESS =====================

def process_frame(frame):
    global current_possession, possession_start_time, last_print_time

    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ball = detect_ball_pi(hsv)
    players = detect_players(hsv, gray)
    nearest = nearest_player_to_ball(players, ball)

    now = time.time()

    # ---- POSSESSION (TIME BASED) ----
    if nearest:
        if current_possession != nearest["color"]:
            current_possession = nearest["color"]
            possession_start_time = now
        elif now - possession_start_time >= POSSESSION_HOLD_TIME:
            possession_counts[current_possession] += 1
    else:
        current_possession = None

    # ---- SPEED (SMOOTHED) ----
    raw_speed = estimate_speed(ball)
    speed_buffer.append(raw_speed)
    speed = sum(speed_buffer)/len(speed_buffer)

    # ---- PRINT ONCE PER SECOND ----
    if now - last_print_time >= PRINT_INTERVAL:
        last_print_time = now

        total = possession_counts["white"] + possession_counts["black"]
        wp = possession_counts["white"]/total*100 if total else 0
        bp = possession_counts["black"]/total*100 if total else 0

        code = 0
        if current_possession:
            code = 1 if current_possession=="white" else 2

        print("\n===============================")
        print(" Foosball Tracking Statistics")
        print("===============================")
        print(f"Ball position:      {ball}")
        print(f"Ball Speed:         {speed:.2f} cm/s")
        print(f"Possession Code:    {code}")
        print(f"Possession White:   {wp:.1f}%")
        print(f"Possession Black:   {bp:.1f}%")
        print("===============================")

    # ---- DISPLAY ----
    disp = frame.copy()
    if ball:
        cv2.circle(disp, ball, 6, (0,0,255), -1)

    for p in players:
        x,y,w,h = p["bbox"]
        col = (255,255,255) if p["color"]=="white" else (0,0,0)
        cv2.rectangle(disp,(x,y),(x+w,y+h),col,2)

    cv2.imshow("Tracking", disp)
    cv2.waitKey(1)

# ===================== RUN MODES =====================

def run_camera(device=0):
    cap = cv2.VideoCapture(device)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)

def run_image(path):
    img = cv2.imread(path)
    process_frame(img)
    cv2.waitKey(0)

def run_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
def run_picamera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280,720)})
    picam2.configure(config)
    picam2.start()

    while True:
        frame = picam2.capture_array()
        process_frame(frame)  # <-- use your existing processing function
