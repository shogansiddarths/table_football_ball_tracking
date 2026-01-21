import cv2
import numpy as np
import time
from collections import deque

# ================== CONFIG ==================
FRAME_W = 640
FRAME_H = 360
PIXELS_PER_CM = 6.0

# Ball HSV (blue / dark ball – adjust if needed)
BALL_HSV_LOW  = (85, 60, 40)
BALL_HSV_HIGH = (140, 255, 255)

MIN_BALL_AREA = 80
MAX_BALL_AREA = 2500

# Possession distance threshold (px)
POSSESSION_DIST = 35

# ================== STATE ==================
prev_ball_pos = None
prev_time = None

last_team = None
last_team_time = time.time()
possession_time = {"white": 0.0, "black": 0.0}

speed_queue = deque(maxlen=5)

# ================== PLAYFIELD ROI ==================
def find_playfield_roi(frame):
    """
    Simple, robust ROI finder based on green field.
    Eliminates white line flicker.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 40, 40]),
        np.array([90, 255, 255])
    )
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )

    cnts, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return 0, 0, frame.shape[1], frame.shape[0]

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

# ================== BALL DETECTION ==================
def detect_ball(hsv, last_pos):
    mask = cv2.inRange(hsv, np.array(BALL_HSV_LOW), np.array(BALL_HSV_HIGH))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.6:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))

        # Motion gating → removes white line flicker
        if last_pos is not None:
            if np.linalg.norm(np.array(center) - np.array(last_pos)) > 40:
                continue

        score = circularity * area
        if score > best_score:
            best_score = score
            best = center

    return best

# ================== SPEED ==================
def estimate_speed(ball_pos):
    global prev_ball_pos, prev_time

    if ball_pos is None:
        return 0.0

    now = time.time()

    if prev_ball_pos is None:
        prev_ball_pos = ball_pos
        prev_time = now
        return 0.0

    dx = ball_pos[0] - prev_ball_pos[0]
    dy = ball_pos[1] - prev_ball_pos[1]
    dist_px = (dx * dx + dy * dy) ** 0.5
    dt = now - prev_time

    prev_ball_pos = ball_pos
    prev_time = now

    if dt <= 0 or dist_px < 2:
        return 0.0

    speed = (dist_px / PIXELS_PER_CM) / dt
    speed_queue.append(speed)
    return sum(speed_queue) / len(speed_queue)

# ================== POSSESSION ==================
def update_possession(ball_pos):
    global last_team, last_team_time

    if ball_pos is None:
        return

    bx, by = ball_pos
    mid_x = FRAME_W // 2

    # Simple side-based possession (robust & fast)
    team = "white" if bx < mid_x else "black"
    now = time.time()

    if last_team is None:
        last_team = team
        last_team_time = now
        return

    if team == last_team:
        possession_time[team] += now - last_team_time

    last_team = team
    last_team_time = now

# ================== PI CAMERA ==================
def run_pi_camera():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed")
        return

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "BGR888"},
        controls={"FrameRate": 120}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("Running… Press CTRL+C to stop")

    # Detect playfield ONCE
    frame = picam2.capture_array()
    fx, fy, fw, fh = find_playfield_roi(frame)

    try:
        while True:
            frame = picam2.capture_array()
            frame = frame[fy:fy + fh, fx:fx + fw]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            ball = detect_ball(hsv, prev_ball_pos)
            speed = estimate_speed(ball)
            update_possession(ball)

            total = possession_time["white"] + possession_time["black"]
            w_pct = 100 * possession_time["white"] / total if total else 0
            b_pct = 100 * possession_time["black"] / total if total else 0

            if ball:
                cv2.circle(frame, ball, 6, (0, 255, 255), -1)

            cv2.putText(
                frame,
                f"Speed: {speed:.1f} cm/s | White {w_pct:.1f}% Black {b_pct:.1f}%",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("Foosball Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# ================== MAIN ==================
if __name__ == "__main__":
    run_pi_camera()
