import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ble_send import BLESender
 
#*********************************
# CAMERA SETUP
#*********************************
FPS_TARGET = 120
 
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1000, 700), "format": "BGR888"},
    controls={"FrameRate": FPS_TARGET}
)
picam2.configure(config)
picam2.start()
time.sleep(1)
 
print("Camera started")

# Initialize the BLE Sender
ble = BLESender()
#*********************************
# GLOBAL VARIABLES
#*********************************
M = None
Minv = None
field_ready = False
field_width = 0
field_height = 0
 
field_corners_camera = None
corners_printed = False
 
frame_count = 0
start_time = time.time()
fps = 0.0
 
goal_width_ratio = 0.4
goal_margin_y_top = 0.07
goal_margin_y_bottom = 0.93
goal_thickness = 4  
 
# SCORES & BALL HISTORY
top_player_goals = 0
bottom_player_goals = 0
goal_event = 0
prev_ball_y = None
 
# False Goal Avoidance
goal_cooldown = 3.0
last_goal_time = 0
 
prev_ball_pos=None
#*********************************
# DRAWING THE LINE IN WARPED SPACE
#*********************************
def draw_warped_line(img, pt1, pt2, Minv, color, thickness=2):
    pts = np.array([[pt1, pt2]], dtype="float32")
    pts = cv2.perspectiveTransform(pts, Minv)
    p1 = tuple(pts[0][0].astype(int))
    p2 = tuple(pts[0][1].astype(int))
    cv2.line(img, p1, p2, color, thickness)
 
# Checking if the Ball is near the Corners
def is_near_corner(x, y, corners, threshold=40):
    for cx, cy in corners:
        if abs(x - cx) < threshold and abs(y - cy) < threshold:
            return True
    return False
 
 
#******************************
# PLAYFIELD DETECTION
#******************************
def detect_playfield(frame):
    global M, Minv, field_ready, field_width, field_height, field_corners_camera
 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    lower_purple = np.array([120, 80, 80])
    upper_purple = np.array([165, 255, 255])
 
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    mask = cv2.medianBlur(mask, 5)
 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[Playfield] Number of contours detected: {len(contours)}")
 
    corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                corners.append([cx, cy])
 
    if len(corners) != 4:
        return False
 
    corners = np.array(corners, dtype="float32")
 
    corners = corners[np.argsort(corners[:, 1])]
    top = corners[:2][np.argsort(corners[:2, 0])]
    bottom = corners[2:][np.argsort(corners[2:, 0])]
    ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
 
    field_corners_camera = ordered.copy()
 
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
        [field_width, 0],
        [field_width, field_height],
        [0, field_height]
    ], dtype="float32")
 
    M = cv2.getPerspectiveTransform(ordered, dst)
    Minv = np.linalg.inv(M)
    field_ready = True
 
    return True
 
#******************************
# MAIN LOOP
#******************************
while True:
    frame = picam2.capture_array()
    output = frame.copy()
 
    # FPS
    frame_count += 1
    if time.time() - start_time >= 1.0:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
 
    # Detect playfield
    if not field_ready:
        detect_playfield(frame)
        cv2.putText(output, "Detecting Playfield",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(output, "Playfield Locked",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # PRINT CORNERS ONCE
    if field_ready and not corners_printed:
        print("\n=== PLAYFIELD CORNERS ===")
 
        print("\nCamera-space corners (pixels):")
        print(f"Top-Left     : {field_corners_camera[0]}")
        print(f"Top-Right    : {field_corners_camera[1]}")
        print(f"Bottom-Right : {field_corners_camera[2]}")
        print(f"Bottom-Left  : {field_corners_camera[3]}")
 
        print("\nWarped-space corners (top-down):")
        print("Top-Left     : (0, 0)")
        print(f"Top-Right    : ({field_width}, 0)")
        print(f"Bottom-Right : ({field_width}, {field_height})")
        print(f"Bottom-Left  : (0, {field_height})")
 
        print("\nCamera → Warped mapping:")
        warped_pts = [(0, 0),
                      (field_width, 0),
                      (field_width, field_height),
                      (0, field_height)]
        for name, cam, warp in zip(["TL", "TR", "BR", "BL"],
                                   field_corners_camera, warped_pts):
            print(f"{name}: {cam}  →  {warp}")
 
        corners_printed = True
    #******************************
    # BALL DETECTION
    #******************************
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.medianBlur(mask_blue, 7)
 
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0
    best_circle = None
    ball_center_warped = None
 
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
 
    #******************************
    # WARP AND GOAL LOGIC
    #******************************
    if best_circle and field_ready:
        x, y, r = best_circle
        # FILTER 1: To ignore detections near corners
        if is_near_corner(x, y, field_corners_camera):
            best_circle = None
 
    if best_circle and field_ready:
        x, y, r = best_circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
 
        pt = np.array([[[x, y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pt, M)
        wx, wy = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
        ball_center_warped = (wx, wy)
        #if frame_count % 10 == 0:
        print(f"Ball: camera=({x},{y})  warped=({wx},{wy})| FPS:{fps:.1f}")
        #FILTER 2: Ignore ball outside field
        if wx < 0 or wx > field_width or wy < 0 or wy > field_height:
            ball_center_warped = None
 
        #FILTER 3: Ignore static objects
        if ball_center_warped is not None:
            if prev_ball_pos is not None:
                dist = np.linalg.norm(np.array(prev_ball_pos) - np.array([wx, wy]))
                if dist < 3:  # not moving
                    ball_center_warped = None
            prev_ball_pos = (wx, wy)
        # Drawing the field border
        draw_warped_line(output, (0, 0), (field_width, 0), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (field_width, 0), (field_width, field_height), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (field_width, field_height), (0, field_height), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (0, field_height), (0, 0), Minv, (0, 255, 0), 3)
 
    if field_ready:
        warped = cv2.warpPerspective(frame, M, (field_width, field_height))
 
        center_x = field_width // 2
        goal_half_width = int((field_width * goal_width_ratio) / 2)
 
        top_goal_y = int(goal_margin_y_top * field_height)
        bottom_goal_y = int(goal_margin_y_bottom * field_height)
 
        cv2.line(warped,
                 (center_x - goal_half_width, top_goal_y),
                 (center_x + goal_half_width, top_goal_y),
                 (0, 0, 255), goal_thickness)
 
        cv2.line(warped,
                 (center_x - goal_half_width, bottom_goal_y),
                 (center_x + goal_half_width, bottom_goal_y),
                 (0, 0, 255), goal_thickness)
 
        if ball_center_warped is not None:
            wx, wy = ball_center_warped
            current_time = time.time()
 
            if prev_ball_y is not None and (current_time - last_goal_time > goal_cooldown):
                if (prev_ball_y < top_goal_y and wy >= top_goal_y and
                    center_x - goal_half_width <= wx <= center_x + goal_half_width):
                    bottom_player_goals += 1
                    last_goal_time = current_time
                    print("GOAL for BLACK PLAYER")
                    goal_event=1
 
                elif (prev_ball_y > bottom_goal_y and wy <= bottom_goal_y and
                      center_x - goal_half_width <= wx <= center_x + goal_half_width):
                    top_player_goals += 1
                    last_goal_time = current_time
                    print("GOAL for WHITE PLAYER")
                    goal_event=2
 
            prev_ball_y = wy
            # Bluetooth Send Data
            ble.send_data(wx, wy, 0, bottom_player_goals, top_player_goals, goal_event, 0, frame_count)
            # Reseting goal event after sending
            goal_event = 0
 
        cv2.putText(warped, f"TOP: {top_player_goals}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
 
        cv2.putText(warped, f"BOTTOM: {bottom_player_goals}",
                    (20, field_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
 
        cv2.imshow("Warped Field", warped)
 
    cv2.putText(output, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
 
    cv2.imshow("Ball Tracking", output)
    cv2.imshow("Blue Mask", mask_blue)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#****************************** 
# CLEANUP
#******************************
cv2.destroyAllWindows()
picam2.stop()
