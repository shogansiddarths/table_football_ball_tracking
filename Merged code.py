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

FRAME_W = 1280
FRAME_H = 720

# ---------------- STATE ----------------
possession_counts = {"white": 0, "black": 0}
prev_ball_pos = None
prev_time = None

last_print_time = 0
PRINT_INTERVAL = 1.0
speed_queue = deque(maxlen=5)

# ----------------- BALL DETECTION ----------------
def detect_ball_pi(hsv):
    mask = cv2.inRange(hsv, np.array(PI_BALL_HSV_LOW), np.array(PI_BALL_HSV_HIGH))
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 2500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5 or r > 20:
            continue

        score = circularity * area
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))

    return best_center

# ---------------- PLAYER DETECTION ----------------
def detect_players(hsv, gray):
    players = []
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    white_mask = ((v >= WHITE_V_MIN) & (s <= WHITE_S_MAX)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    _, black_mask = cv2.threshold(gray, BLACK_V_MAX, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    for mask, label in [(white_mask, "white"), (black_mask, "black")]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_PLAYER_AREA < area < MAX_PLAYER_AREA:
                M = cv2.moments(c)
                if M["m00"] == 0: 
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(c)

                players.append({
                    "center": (cx, cy),
                    "bbox": (x, y, w, h),
                    "color": label
                })
    return players

# ---------------- NEAREST PLAYER ----------------
def nearest_player_to_ball(players, ball_center):
    if ball_center is None or not players:
        return None

    bx, by = ball_center
    best = None
    best_dist = 1e9

    for p in players:
        x, y, w, h = p["bbox"]
        fx = x + w // 2
        fy = y + h
        d = ((fx - bx)**2 + (fy - by)**2)**0.5
        if d < best_dist:
            best_dist = d
            best = p
    return best

# ---------------- SPEED ESTIMATION ----------------
def estimate_speed(ball_center):
    global prev_ball_pos, prev_time, speed_queue

    if ball_center is None:
        speed_queue.append(0)
        return 0

    now = time.time()
    speed = 0

    if prev_ball_pos is not None:
        dx = ball_center[0] - prev_ball_pos[0]
        dy = ball_center[1] - prev_ball_pos[1]
        dist_px = (dx*dx + dy*dy)**0.5
        dt = now - prev_time
        if dt > 0:
            speed_cm_s = (dist_px / PIXELS_PER_CM) / dt
            speed = speed_cm_s if speed_cm_s > 0.5 else 0

    prev_ball_pos = ball_center
    prev_time = now

    # Smooth over last 5 frames
    speed_queue.append(speed)
    smooth_speed = sum(speed_queue)/len(speed_queue)
    return smooth_speed

# ---------------- PREPROCESS ----------------
def preprocess(frame):
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return hsv, gray

# ---------------- ROD DETECTION ----------------
def detect_rods(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    rod_ys = []
    if lines is not None:
        # keep only horizontal lines
        horizontal_lines = [line[0] for line in lines if abs(line[0][1]-line[0][3])<5]
        # sort by line length descending
        horizontal_lines.sort(key=lambda l: abs(l[2]-l[0]), reverse=True)
        # pick top 8 lines only
        for line in horizontal_lines[:8]:
            y_center = (line[1]+line[3])//2
            rod_ys.append(y_center)
    rod_ys.sort()
    return rod_ys

def assign_players_to_rods(players, rod_ys):
    for p in players:
        py = p["center"][1]
        closest_rod = min(rod_ys, key=lambda y: abs(y-py))
        p["rod_y"] = closest_rod
    return players

# ---------------- PROCESS FRAME ----------------
def process_frame(frame):
    global possession_counts, last_print_time

    hsv, gray = preprocess(frame)
    ball_center = detect_ball_pi(hsv)
    ball_mask = np.zeros_like(gray)  # define it first
    if ball_center is not None:
        cv2.circle(ball_mask, ball_center, 7, 255, -1)  # ball = white

    # ---- Rods ----
    rod_ys = detect_rods(frame)
    rod_mask = np.zeros_like(gray)
    for y in rod_ys:
        cv2.line(rod_mask, (0,y), (frame.shape[1],y), 255, 3)

    # ---- Field ----
    # Simple field detection using thresholding (adjust if needed)
    field_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))  # green-ish field
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # ---- Combine masks ----
    combined_mask = cv2.bitwise_or(field_mask, ball_mask)
    combined_mask = cv2.bitwise_or(combined_mask, rod_mask)

    # ---- Player Detection (optional for possession) ----
    players = detect_players(hsv, gray)
    if rod_ys:
        players = assign_players_to_rods(players, rod_ys)
    nearest = nearest_player_to_ball(players, ball_center)
    if nearest:
        possession_counts[nearest["color"]] += 1

    speed = estimate_speed(ball_center)

    # Print once per second
    now = time.time()
    if now - last_print_time >= 1.0:
        total = possession_counts["white"] + possession_counts["black"]
        white_pct = (possession_counts["white"]/total*100) if total else 0
        black_pct = (possession_counts["black"]/total*100) if total else 0
        possession_color = nearest["color"] if nearest else "None"

        print(f"Ball with: {possession_color}, White: {white_pct:.1f}%, Black: {black_pct:.1f}%")
        last_print_time = now
    cv2.imshow("Field Mask", combined_mask)
    cv2.waitKey(1)

    # Display (optional)
    disp = frame.copy()
    if ball_center:
        cv2.circle(disp, ball_center, 5, (0,0,255), -1)
    for p in players:
        color = (255,255,255) if p["color"]=="white" else (0,0,0)
        cv2.circle(disp, p["center"], 5, color, -1)
    cv2.imshow("Tracking", disp)
    cv2.waitKey(1)
# ---------------- RUN PI CAMERA ----------------
def run_pi_camera():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not available")
        return

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size":(FRAME_W,FRAME_H),"format":"BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("Pi Camera started. Press CTRL+C to stop.")

    try:
        while True:
            frame = picam2.capture_array()
            process_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def run_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Error opening image:",path)
        return
    process_frame(img)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Choose mode:")
    print("1 = Video file")
    print("2 = Image file")
    print("3 = USB Camera")
    print("4 = Raspberry Pi Camera")

    choice = input("Mode: ")

    if choice == "1":
        video_path = r"C:\Users\Deepika\OneDrive\Documents\Deepika\tracker3 (1).mp4"
        run_video(video_path)

    elif choice == "2":
        image_path = r"C:\Users\Deepika\OneDrive\Documents\Deepika\balltracker2.jpg"
        run_image(image_path)

    elif choice == "3":
        run_camera(0)

    elif choice == "4":
        run_pi_camera()

    else:
        print("Invalid choice.")
        
