import cv2
import numpy as np
from picamera2 import Picamera2
import time

# ==========================================
# STEP 0: Initialize Pi Camera @120 FPS
# ==========================================
FPS_TARGET = 120

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "BGR888"},
    controls={"FrameRate": FPS_TARGET}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

print("Camera started — Ball + Player Possession Tracking")

# ==========================================
# GLOBAL VARIABLES
# ==========================================
M = None
field_ready = False
field_width = 0
field_height = 0

# FPS measurement
frame_count = 0
start_time = time.time()
fps = 0.0

# ==========================================
# FUNCTION: Detect Playfield using Violet Tape
# ==========================================
def detect_playfield(frame):
    global M, field_ready, field_width, field_height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([120, 80, 80])
    upper_purple = np.array([165, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            Mnts = cv2.moments(cnt)
            if Mnts["m00"] != 0:
                cx = int(Mnts["m10"] / Mnts["m00"])
                cy = int(Mnts["m01"] / Mnts["m00"])
                corners.append([cx, cy])

    if len(corners) != 4:
        return frame, False

    corners = np.array(corners, dtype="float32")
    corners = corners[np.argsort(corners[:, 1])]
    top = corners[:2][np.argsort(corners[:2, 0])]
    bottom = corners[2:][np.argsort(corners[2:, 0])]
    ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    field_width = int(max(
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[2] - ordered[3])
    ))
    field_height = int(max(
        np.linalg.norm(ordered[0] - ordered[3]),
        np.linalg.norm(ordered[1] - ordered[2])
    ))

    dst = np.array([
        [0, 0],
        [field_width - 1, 0],
        [field_width - 1, field_height - 1],
        [0, field_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    field_ready = True

    cv2.polylines(frame, [ordered.astype(int)], True, (255, 0, 0), 3)
    return frame, True

# ==========================================
# FUNCTION: Get player centers from mask
# ==========================================
def get_player_centers(mask, min_area=400):
    centers = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        Mnts = cv2.moments(cnt)
        if Mnts["m00"] == 0:
            continue

        cx = int(Mnts["m10"] / Mnts["m00"])
        cy = int(Mnts["m01"] / Mnts["m00"])
        centers.append((cx, cy))

    return centers

# ==========================================
# FUNCTION: Closest distance
# ==========================================
def closest_distance(ball, players):
    if not players:
        return float("inf")
    return min(np.hypot(ball[0] - px, ball[1] - py) for px, py in players)

# ==========================================
# REAL-TIME LOOP
# ==========================================
while True:
    frame = picam2.capture_array()
    output = frame.copy()

    # ---------------- FPS COUNTER ----------------
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    # ------------------------------------------
    # STEP 1: Detect Playfield
    # ------------------------------------------
    if not field_ready:
        output, _ = detect_playfield(output)
        cv2.putText(output, "Detecting Playfield...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ------------------------------------------
    # STEP 2: HSV Conversion
    # ------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------------------------------
    # STEP 3: Blue Ball Detection
    # ------------------------------------------
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.medianBlur(mask_blue, 7)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 2500:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.6:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5 or r > 20:
            continue

        score = circularity * area
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(r))

    # ------------------------------------------
    # STEP 4: Player Detection (Black & White)
    # ------------------------------------------
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    black_players = get_player_centers(black_mask)
    white_players = get_player_centers(white_mask)

    # ------------------------------------------
    # STEP 5: Draw Ball + Possession Logic
    # ------------------------------------------
    if best_circle is not None:
        x, y, r = best_circle
        ball = (x, y)

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

        d_black = closest_distance(ball, black_players)
        d_white = closest_distance(ball, white_players)

        if min(d_black, d_white) > 40:
            possession = "FREE BALL"
        elif d_black < d_white:
            possession = "BLACK"
        else:
            possession = "WHITE"

        cv2.putText(output, f"Possession: {possession}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

    # ------------------------------------------
    # DISPLAY
    # ------------------------------------------
    cv2.imshow("Ball Tracking", output)
    cv2.imshow("Blue Mask", mask_blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cv2.destroyAllWindows()
picam2.stop()
