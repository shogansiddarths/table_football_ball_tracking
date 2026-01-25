import cv2
import numpy as np
import time
from collections import deque

# ===============================
# FIELD ROD DEFINITIONS 
# ===============================
RODS = [
    {"color": "white", "x1": 44,  "x2": 465, "y": (49 + 65) // 2},
    {"color": "white", "x1": 44,  "x2": 467, "y": (148 + 166) // 2},
    {"color": "black", "x1": 43,  "x2": 470, "y": (218 + 258) // 2},
    {"color": "white", "x1": 43,  "x2": 465, "y": (325 + 358) // 2},
    {"color": "black", "x1": 44,  "x2": 471, "y": (427 + 464) // 2},
    {"color": "white", "x1": 46,  "x2": 476, "y": (518 + 527) // 2},
    {"color": "black", "x1": 48,  "x2": 473, "y": (619 + 623) // 2},
]

POSSESSION_Y_THRESHOLD = 40
PIXELS_PER_CM = 6
FRAME_W = 1000
FRAME_H = 700

possession_counts = {"white": 0, "black": 0}
prev_ball_pos = None
prev_time = None

# ===============================
# WARPED FIELD GLOBALS
# ===============================
M = None
field_ready = False
field_width = 0
field_height = 0

# ===============================
# BALL DETECTION 
# ===============================
def detect_ball_pi(hsv):
    lower_ball = np.array([85, 60, 40])
    upper_ball = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 2500:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.6:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5 or r > 20:
            continue

        score = circ * area
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))

    return best_center

# ===============================
# SPEED ESTIMATION 
# ===============================
def estimate_speed(ball_center):
    global prev_ball_pos, prev_time

    if ball_center is None:
        return 0

    now = time.time()
    speed = 0

    if prev_ball_pos is not None:
        dx = ball_center[0] - prev_ball_pos[0]
        dy = ball_center[1] - prev_ball_pos[1]
        dist_px = (dx*dx + dy*dy) ** 0.5
        dt = now - prev_time
        if dt > 0:
            speed = (dist_px / PIXELS_PER_CM) / dt

    prev_ball_pos = ball_center
    prev_time = now
    return speed

# ===============================
# POSSESSION LOGIC 
# ===============================
def determine_possession(ball_center):
    if ball_center is None:
        return None

    bx, by = ball_center
    for rod in RODS:
        if rod["x1"] <= bx <= rod["x2"]:
            if abs(by - rod["y"]) <= POSSESSION_Y_THRESHOLD:
                return rod["color"]
    return None

# ===============================
# PLAYFIELD DETECTION 
# ===============================
def detect_playfield(frame):
    global M, field_ready, field_width, field_height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (120, 80, 80), (165, 255, 255))
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = []

    for c in contours:
        if cv2.contourArea(c) > 150:
            m = cv2.moments(c)
            if m["m00"] != 0:
                corners.append([
                    int(m["m10"]/m["m00"]),
                    int(m["m01"]/m["m00"])
                ])

    if len(corners) != 4:
        return

    pts = np.array(corners, dtype="float32")
    pts = pts[np.argsort(pts[:,1])]
    top = pts[:2][np.argsort(pts[:2,0])]
    bot = pts[2:][np.argsort(pts[2:,0])]
    ordered = np.array([top[0], top[1], bot[1], bot[0]], dtype="float32")

    field_width = int(np.linalg.norm(ordered[0] - ordered[1]))
    field_height = int(np.linalg.norm(ordered[0] - ordered[3]))

    dst = np.array([
        [0,0],
        [field_width,0],
        [field_width,field_height],
        [0,field_height]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    field_ready = True

# ===============================
# PROCESS FRAME
# ===============================
def preprocess(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

def process_frame(frame):
    global possession_counts

    if not field_ready:
        detect_playfield(frame)

    hsv = preprocess(frame)
    ball_center = detect_ball_pi(hsv)

    possession = determine_possession(ball_center)
    if possession:
        possession_counts[possession] += 1

    speed = estimate_speed(ball_center)

    total = possession_counts["white"] + possession_counts["black"]
    white_pct = (possession_counts["white"] / total * 100) if total else 0
    black_pct = (possession_counts["black"] / total * 100) if total else 0

    possession_code = 1 if possession == "white" else 2 if possession == "black" else 0

    print("\n===============================")
    print(" Foosball Tracking Statistics")
    print("===============================")
    print(f"Ball position:      {ball_center}")
    print(f"Ball Speed:         {speed:.2f} cm/s")
    print(f"Possession Code:    {possession_code}")
    print(f"Possession White:   {white_pct:.1f}%")
    print(f"Possession Black:   {black_pct:.1f}%")
    print("===============================")

    disp = frame.copy()
    if ball_center:
        cv2.circle(disp, ball_center, 6, (0, 0, 255), -1)

    for rod in RODS:
        color = (255,255,255) if rod["color"]=="white" else (0,0,0)
        cv2.line(disp, (rod["x1"], rod["y"]), (rod["x2"], rod["y"]), color, 2)

    cv2.imshow("Camera View", disp)

    # -------- WARPED FIELD WITH SAME RODS --------
    if field_ready:
        warped = cv2.warpPerspective(frame, M, (field_width, field_height))
        for rod in RODS:
            pts = np.array([[[rod["x1"], rod["y"]], [rod["x2"], rod["y"]]]], dtype="float32")
            wpts = cv2.perspectiveTransform(pts, M)
            cv2.line(
                warped,
                tuple(wpts[0][0].astype(int)),
                tuple(wpts[0][1].astype(int)),
                (255,255,255) if rod["color"]=="white" else (0,0,0),
                2
            )
        cv2.imshow("Warped Field", warped)

    cv2.waitKey(1)

# ===============================
# PI CAMERA RUN
# ===============================
def run_pi_camera():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not available")
        return

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("Pi Camera started. CTRL+C to stop")

    try:
        while True:
            frame = picam2.capture_array()
            process_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    run_pi_camera()
