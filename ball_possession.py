import cv2
import numpy as np
import time
from collections import deque
from picamera2 import Picamera2

# HSV color range for detecting the foosball (orange ball)

BALL_HSV_LOW  = (5, 100, 100)
BALL_HSV_HIGH = (25, 255, 255)

# Thresholds for detecting white and black players

WHITE_V_MIN = 180
WHITE_S_MAX = 80
BLACK_V_MAX = 80

MIN_PLAYER_AREA = 80
MAX_PLAYER_AREA = 5000

# Conversion factor for speed estimation

PIXELS_PER_CM = 6

# Frame size for processing (lower = faster)

FRAME_W = 480
FRAME_H = 270

# Stats
possession_counts = {"white": 0, "black": 0}

# Previous ball position and time (used for speed calculation)

prev_ball_pos = None
prev_time = None


def preprocess(frame):
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return hsv, gray

def detect_ball(hsv):
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None

    x, y, w, h = cv2.boundingRect(c)
    return (int(x + w / 2), int(y + h / 2))


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
                players.append({"center": (cx, cy), "color": label})

    return players

//Find the player closest to the ball.
Returns the player and distance to the ball.//

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

    //Estimate ball speed in cm/s using frame-to-frame displacement.//

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
            speed = (dist_px / PIXELS_PER_CM) / dt             # Convert pixels to centimeters and divide by time


    prev_ball_pos = ball_center
    prev_time = now
    return speed


def process_frame(frame):
    global possession_counts

    hsv, gray = preprocess(frame)
    ball_center = detect_ball(hsv)
    players = detect_players(hsv, gray)
    nearest, dist = nearest_player_to_ball(players, ball_center)

    if nearest:
        possession_counts[nearest["color"]] += 1

    speed = estimate_speed(ball_center)

    # ---------------- TERMINAL OUTPUT ----------------
        # Calculate possession percentages

    total = possession_counts["white"] + possession_counts["black"]
    if total == 0:
        white_pct = 0
        black_pct = 0
    else:
        white_pct = (possession_counts["white"] / total) * 100
        black_pct = (possession_counts["black"] / total) * 100

    print(total)
    print(possession_counts["white"])
    print(possession_counts["black"])
    
    print("\n===============================")
    print(" Foosball Tracking Statistics")
    print("===============================")
    print(f"Ball position:      {ball_center}")
    print(f"Ball Speed:         {speed:.2f} cm/s")

    if nearest is None:
        possession_code = 0
    else:
        possession_code = 1 if nearest["color"] == "white" else 2
    print(f"Possession Code:    {possession_code}")
    print(f"Possession White:   {white_pct:.1f}%")
    print(f"Possession Black:   {black_pct:.1f}%")
    print("===============================")


def run_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open video:", path)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)

    cap.release()


def run_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Error opening image:",path)
        return
    process_frame(img)


def run_camera(device=0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("Camera not available.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        process_frame(frame)

def run_pi_camera():
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Pi Camera started. Press CTRL+C to stop.")

    try:
        while True:
            frame = picam2.capture_array()
            process_frame(frame)
    except KeyboardInterrupt:
        print("Stopping Pi Camera...")
    finally:
        picam2.stop()

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
        image_path = r"C:\Users\Deepika\OneDrive\Documents\Deepika\balltracker1.jpg"
        run_image(image_path)

    elif choice == "3":
        run_camera(0)

    elif choice == "4":
        run_pi_camera()

    else:
        print("Invalid choice.")
