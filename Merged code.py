
import cv2
import numpy as np
import time
from collections import deque

# Thresholds for detecting white and black players

WHITE_V_MIN = 180
WHITE_S_MAX = 75
BLACK_V_MAX = 60

MIN_PLAYER_AREA = 80
MAX_PLAYER_AREA = 5000

# Conversion factor for speed estimation

PIXELS_PER_CM = 6

# Frame size for processing (lower = faster)

FRAME_W = 1280
FRAME_H = 720

# Stats
possession_counts = {"white": 0, "black": 0}
using_pi_camera = False  # default: not using Pi camera

# Previous ball position and time (used for speed calculation)

prev_ball_pos = None
prev_time = None


def detect_ball_pi(hsv):
    """
    Robust ball detection (works on Pi camera, images, video)
    Uses color + circularity scoring
    """

    #  BLUE ball HSV (from your working Pi code)
    lower_ball = np.array([85, 60, 40])
    upper_ball = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

#Find the player closest to the ball. Returns the player and distance to the ball

def nearest_player_to_ball(players, ball_center):
    if ball_center is None or not players:
        return None, None

    bx, by = ball_center
    best = None
    best_dist = 1e9

    for p in players:
        x, y, w, h = p["bbox"]
        # foot point = bottom-center of player
        fx = x + w // 2
        fy = y + h

        d = ((fx - bx)**2 + (fy - by)**2)**0.5
        if d < best_dist:
            best_dist = d
            best = p

    return best, best_dist

    #Estimate ball speed in cm/s using frame-to-frame displacement.

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
            speed = (dist_px / PIXELS_PER_CM) / dt             # Convert pixels to centimeters and divide by time


    prev_ball_pos = ball_center
    prev_time = now
    return speed

def preprocess(frame):
    """
    Prepares the frame for processing:
    - Applies Gaussian blur
    - Converts to HSV (for ball + player detection)
    - Converts to grayscale (for black player detection)
    """
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return hsv, gray

def process_frame(frame):

    global possession_counts, using_pi_camera

    hsv, gray = preprocess(frame)

    # Use the Pi-camera’s working ball detection
    ball_center = detect_ball_pi(hsv)  
# This already uses blue HSV + circularity

    players = detect_players(hsv, gray)
    nearest, dist = nearest_player_to_ball(players, ball_center)

    # Update possession
    if nearest:
        possession_counts[nearest["color"]] += 1

    # Speed calculation
    speed = estimate_speed(ball_center)

    # ---------------- TERMINAL OUTPUT ----------------
    total = possession_counts["white"] + possession_counts["black"]
    white_pct = (possession_counts["white"] / total * 100) if total else 0
    black_pct = (possession_counts["black"] / total * 100) if total else 0

    print("\n===============================")
    print(" Foosball Tracking Statistics")
    print("===============================")
    print(f"Ball position:      {ball_center}")
    print(f"Ball Speed:         {speed:.2f} cm/s")

    possession_code = 0
    if nearest:
        possession_code = 1 if nearest["color"] == "white" else 2

    print(f"Possession Code:    {possession_code}")
    print(f"Possession White:   {white_pct:.1f}%")
    print(f"Possession Black:   {black_pct:.1f}%")
    print("===============================")

    # OPTIONAL: show Pi-camera’s mask
    disp = frame.copy()
    if ball_center is not None:
        cv2.circle(disp, ball_center, 5, (0,0,255), -1)
    for p in players:
        color = (255,255,255) if p["color"]=="white" else (0,0,0)
        cv2.circle(disp, p["center"], 5, color, -1)
    cv2.imshow("Tracking", disp)
    cv2.waitKey(1)


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

def run_pi_camera():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 is not available. This only works on Raspberry Pi.")
        return

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
    )
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


if __name__ == "__main__":
    print("Choose mode:")
    print("1 = Video file")
    print("2 = Image file")
    print("3 = Raspberry Pi Camera")

    choice = input("Mode: ")

    if choice == "1":
        video_path = r"C:\Users\Deepika\OneDrive\Documents\Deepika\tracker3 (1).mp4"
        run_video(video_path)

    elif choice == "2":
        image_path = r"C:\Users\Deepika\OneDrive\Documents\Deepika\balltracker2.jpg"
        run_image(image_path)

    elif choice == "3":
        run_pi_camera()

    else:
        print("Invalid choice.")
        
