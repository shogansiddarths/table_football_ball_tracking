import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ----------------------------------------
# Ball & Player Tracking Settings
# ----------------------------------------
PIXELS_PER_CM = 6
FRAME_W, FRAME_H = 480, 270
MIN_PLAYER_AREA, MAX_PLAYER_AREA = 80, 5000
WHITE_V_MIN, WHITE_S_MAX = 180, 80
BLACK_V_MAX = 80

possession_counts = {"white": 0, "black": 0}
prev_ball_pos = None
prev_time = None

# ----------------------------------------
# Player Detection Functions
# ----------------------------------------
def preprocess(frame):
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return hsv, gray

def detect_players(hsv, gray):
    players = []
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    white_mask = ((v >= WHITE_V_MIN) & (s <= WHITE_S_MAX)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    _, black_mask = cv2.threshold(gray, BLACK_V_MAX, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

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
                players.append({"center": (cx, cy), "color": label})
    return players

def nearest_player_to_ball(players, ball_center):
    if ball_center is None or not players:
        return None, None
    bx, by = ball_center
    best = None
    best_dist = 1e9
    for p in players:
        px, py = p["center"]
        d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = p
    return best, best_dist

def estimate_speed(ball_center):
    global prev_ball_pos, prev_time
    if ball_center is None:
        prev_ball_pos = None
        prev_time = None
        return 0
    now = time.time()
    speed = 0
    if prev_ball_pos is not None:
        dx = ball_center[0] - prev_ball_pos[0]
        dy = ball_center[1] - prev_ball_pos[1]
        dist_px = (dx*dx + dy*dy)**0.5
        dt = now - prev_time
        if dt > 0:
            speed = (dist_px / PIXELS_PER_CM) / dt
    prev_ball_pos = ball_center
    prev_time = now
    return speed

# ----------------------------------------
# Ball Detection (BLUE BALL Version)
# ----------------------------------------
def detect_ball_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_circle = None
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
            best_cnt = cnt
            best_circle = (int(x), int(y), int(r))
    if best_circle is not None:
        x, y, r = best_circle
        return (x, y), mask
    return None, mask

# ----------------------------------------
# Processing Frame
# ----------------------------------------
def process_frame(frame):
    global possession_counts
    hsv, gray = preprocess(frame)
    players = detect_players(hsv, gray)

    ball_center, mask = detect_ball_blue(frame)
    nearest, dist = nearest_player_to_ball(players, ball_center)
    if nearest:
        possession_counts[nearest["color"]] += 1
    speed = estimate_speed(ball_center)

    # Output stats
    total = possession_counts["white"] + possession_counts["black"]
    white_pct = (possession_counts["white"] / total * 100) if total > 0 else 0
    black_pct = (possession_counts["black"] / total * 100) if total > 0 else 0
    possession_code = 0 if nearest is None else (1 if nearest["color"]=="white" else 2)

    print(f"Ball: {ball_center}, Speed: {speed:.2f} cm/s, Possession: {possession_code}, White: {white_pct:.1f}%, Black: {black_pct:.1f}%")

    # Display
    display = frame.copy()
    if ball_center:
        cv2.circle(display, ball_center, 10, (0,255,0), 2)
    cv2.imshow("Frame", display)
    cv2.imshow("Ball Mask", mask)
    cv2.waitKey(1)

# ----------------------------------------
# Main Loop (Pi Camera)
# ----------------------------------------
def run_pi_camera():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("Pi Camera started — tracking blue ball")

    try:
        while True:
            frame = picam2.capture_array()
            process_frame(frame)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# Run
if __name__=="__main__":
    run_pi_camera()
