import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ble_sender import BLESender
 
#******************************
# CAMERA SETUP
#******************************
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
 
ble = BLESender()
 
#******************************
# GLOBAL VARIABLES
#******************************
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
prev_ball_pos_speed = None
prev_valid_ball = None
field_mask=None
 
# GOAL SETTINGS
goal_width_ratio = 0.16
goal_depth_ratio = 0.06
goal_thickness = 6
 
# SCORE & BALL HISTORY
top_player_goals = 0
bottom_player_goals = 0
goal_event = 0
prev_ball_pos = None
prev_ball_axis = None
goal_cooldown = 2.5
last_goal_time = 0
goal_axis = "y"  # auto-detected later
 
#*********************************
# FIELD & ROD DEFINITIONS
#*********************************
RODS = [
    {"color": "white", "x": 146},
    {"color": "white", "x": 233},
    {"color": "black", "x": 340},
    {"color": "white", "x": 440},
    {"color": "black", "x": 540},
    {"color": "white", "x": 640},
    {"color": "black", "x": 730},
    {"color": "black", "x": 830},
]
 
TABLE_Y_MIN = 223
TABLE_Y_MAX = 627
 
POSSESSION_X_THRESHOLD = 27  # pixels
PIXELS_PER_CM = 6
 
FRAME_W = 1000
FRAME_H = 700
 
possession_counts = {"white": 0, "black": 0}
 
prev_time = None
 
#*********************************
# DRAWING THE LINE IN WARPED SPACE
#*********************************
def draw_warped_line(img, pt1, pt2, Minv, color, thickness=2):
    pts = np.array([[pt1, pt2]], dtype="float32")
    pts = cv2.perspectiveTransform(pts, Minv)
    p1 = tuple(pts[0][0].astype(int))
    p2 = tuple(pts[0][1].astype(int))
    cv2.line(img, p1, p2, color, thickness)
 
 
# CHECK IF BALL IS NEAR CORNER
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
 
    #lower_purple = np.array([120, 80, 80])
    #upper_purple = np.array([165, 255, 255])
    lower_purple = np.array([100, 50, 50])  # Lower values for Hue, Saturation, and Value
    upper_purple = np.array([160, 255, 255])  # Higher values for Hue, Saturation, and Value
 
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
 
# ===============================
# SPEED ESTIMATION
# ===============================
 
def estimate_speed(ball_center):
    global prev_ball_pos_speed, prev_time
 
    if ball_center is None:
        prev_ball_pos_speed = None
        prev_time = None
        return 0
 
    now = time.time()
    speed = 0
 
    if prev_ball_pos_speed is not None:
        dx = ball_center[0] - prev_ball_pos_speed[0]
        dy = ball_center[1] - prev_ball_pos_speed[1]
        dist_px = (dx * dx + dy * dy) ** 0.5
        dt = now - prev_time
 
        if dt > 0:
            speed = (dist_px / PIXELS_PER_CM) / dt
 
    prev_ball_pos_speed = ball_center
    prev_time = now
    return speed
 
# ===============================
# POSSESSION LOGIC
# ===============================
 
def determine_possession(ball_center):
    if ball_center is None:
        return None
 
    bx, by = ball_center
 
    if not (TABLE_Y_MIN <= by <= TABLE_Y_MAX):
        return None
 
    closest_rod = None
    min_dx = float("inf")
 
    for rod in RODS:
        dx = abs(bx - rod["x"])
        if dx <= POSSESSION_X_THRESHOLD and dx < min_dx:
            min_dx = dx
            closest_rod = rod
 
    if closest_rod:
        return closest_rod["color"]
 
    return None
 
 
 
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
    # Printing the Corners
    if field_ready and not corners_printed:
        print("\n=== PLAYFIELD CORNERS ===")
 
        print("\nCamera-space corners (pixels):")
        print(f"Top-Left: {field_corners_camera[0]}")
        print(f"Top-Right: {field_corners_camera[1]}")
        print(f"Bottom-Right: {field_corners_camera[2]}")
        print(f"Bottom-Left: {field_corners_camera[3]}")
 
        print("\nWarped-space corners (top-down):")
        print("Top-Left: (0, 0)")
        print(f"Top-Right: ({field_width}, 0)")
        print(f"Bottom-Right: ({field_width}, {field_height})")
        print(f"Bottom-Left: (0, {field_height})")
 
        print("\nCamera -> Warped mapping:")
        warped_pts = [(0, 0),
                      (field_width, 0),
                      (field_width, field_height),
                      (0, field_height)]
        for name, cam, warp in zip(["TL", "TR", "BR", "BL"],
                                   field_corners_camera, warped_pts):
            print(f"{name}: {cam}->{warp}")
 
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
        if r < 8 or r > 20:
            continue
 
        score = circularity * area
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(r))
 
    ball_center_warped = None
 
    #******************************
    # WARP AND GOAL LOGIC
    #******************************
 
    if best_circle and field_ready:
        x, y, r = best_circle
        best_center = (int(x), int(y))
        possession = determine_possession(best_center)
        if possession:
            possession_counts[possession] += 1
        speed = estimate_speed(best_center)
        total = possession_counts["white"] + possession_counts["black"]
        white_pct = (possession_counts["white"] / total * 100) if total else 0
        black_pct = (possession_counts["black"] / total * 100) if total else 0
        possession_code = 1 if possession == "white" else 2 if possession == "black" else 0
        print("\n===============================")
        print(" Foosball Tracking Statistics")
        print("===============================")
        #print(f"Ball position:      {ball_center}")
        print(f"Ball speed:         {speed:.2f} cm/s")
        print(f"Possession code:    {possession_code}")
        print(f"White possession:   {white_pct:.1f}%")
        print(f"Black possession:   {black_pct:.1f}%")
        print("===============================")
        #FILTER 1: Ignore detections near corners
        if not is_near_corner(x, y, field_corners_camera):
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
 
            pt = np.array([[[x, y]]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(pt, M)
            wx, wy = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
            ball_center_warped = (wx, wy)
 
 
            # if frame_count % 10 == 0:
            print(f"Ball: camera=({x},{y})  warped=({wx},{wy})| FPS:{fps:.1f}")
 
            # FILTER 2: Ignore ball outside field
            if 0 <= wx <= field_width and 0 <= wy <= field_height:
                ball_center_warped = (wx, wy)
                # FILTER 3: Ignore static objects (corners don't move) 
                if prev_ball_pos is not None:
                    dist = np.linalg.norm(np.array(prev_ball_pos) - np.array([wx, wy]))
                    if dist < 3:# not moving
                        ball_center_warped = None
 
                prev_ball_pos = (wx, wy)
            # Drawing the field border
            draw_warped_line(output, (0, 0), (field_width, 0), Minv, (0, 255, 0), 3)
            draw_warped_line(output, (field_width, 0), (field_width, field_height), Minv, (0, 255, 0), 3)
            draw_warped_line(output, (field_width, field_height), (0, field_height), Minv, (0, 255, 0), 3)
            draw_warped_line(output, (0, field_height), (0, 0), Minv, (0, 255, 0), 3)
 
    #**************************************
    # WARP FIELD + AUTO GOAL AXIS DETECTION
    #**************************************
    if field_ready:
        warped = cv2.warpPerspective(frame, M, (field_width, field_height))
        # Player position marking
        for rod in RODS:
            color = (255, 255, 255) if rod["color"] == "white" else (0, 0, 0)
            cv2.line(frame, (rod["x"], TABLE_Y_MIN), (rod["x"], TABLE_Y_MAX), color, 2)
 
        #cv2.imshow("Camera View", frame)
        for rod in RODS:
            pts = np.array(
                [[[rod["x"], TABLE_Y_MIN], [rod["x"], TABLE_Y_MAX]]],
                dtype="float32"
            )
            wpts = cv2.perspectiveTransform(pts, M)
            cv2.line(
                warped,
                tuple(wpts[0][0].astype(int)),
                tuple(wpts[0][1].astype(int)),
                (255, 255, 255) if rod["color"] == "white" else (0, 0, 0),
                5  # Thicker line for better visibility
            )
        # Auto Detect the Goal Axis
        if field_width > field_height:
            goal_axis = "x"  # goals left-right
        else:
            goal_axis = "y"  # goals top-bottom
 
        center_x = field_width // 2
        center_y = field_height // 2
        goal_half_width = int(field_width * goal_width_ratio / 2)
        goal_depth = int(field_height * goal_depth_ratio)
 
        # Drawing the Goal Lines
        if goal_axis == "y":
            top_goal_y = goal_depth
            bottom_goal_y = field_height - goal_depth
 
            cv2.line(warped, (center_x - goal_half_width, top_goal_y),
                     (center_x + goal_half_width, top_goal_y), (0, 0, 255), goal_thickness)
 
            cv2.line(warped, (center_x - goal_half_width, bottom_goal_y),
                     (center_x + goal_half_width, bottom_goal_y), (0, 0, 255), goal_thickness)
 
        else:
            left_goal_x = goal_depth
            right_goal_x = field_width - goal_depth
 
            cv2.line(warped, (left_goal_x, center_y - goal_half_width),
                     (left_goal_x, center_y + goal_half_width), (0, 0, 255), goal_thickness)
 
            cv2.line(warped, (right_goal_x, center_y - goal_half_width),
                     (right_goal_x, center_y + goal_half_width), (0, 0, 255), goal_thickness)
 
        #******************************
        # GOAL DETECTION LOGIC
        #******************************
        if ball_center_warped is not None:
            wx, wy = ball_center_warped
            current_time = time.time()
 
            if prev_valid_ball is not None and (current_time - last_goal_time > goal_cooldown):
                px,py = prev_valid_ball
 
                if goal_axis == "y":
                    if py < top_goal_y <= wy:
                        bottom_player_goals += 1
                        goal_event = 1
                        last_goal_time = current_time
                        print("GOAL -> BOTTOM PLAYER")
 
                    elif py > bottom_goal_y >= wy:
                        top_player_goals += 1
                        goal_event = 2
                        last_goal_time = current_time
                        print("GOAL -> TOP PLAYER")
 
                else:
                    if px < left_goal_x <= wx: 
                        bottom_player_goals += 1
                        goal_event = 1
                        last_goal_time = current_time
                        print("GOAL -> BOTTOM PLAYER")
 
                    elif px > right_goal_x >= wx: 
                        top_player_goals += 1
                        goal_event = 2
                        last_goal_time = current_time
                        print("GOAL -> TOP PLAYER")
 
            prev_valid_ball = (wx, wy)
            prev_ball_axis = wy if goal_axis == "y" else wx
 
            ble.send_data(wx, wy, possession_code, bottom_player_goals, top_player_goals, goal_event, int(speed), frame_count)
 
        # SCORE DISPLAY
        cv2.putText(warped, f"TOP: {top_player_goals}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        cv2.putText(warped, f"BOTTOM: {bottom_player_goals}", (20, field_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        cv2.putText(warped, f"GOAL AXIS: {goal_axis.upper()}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
 
        cv2.imshow("Warped Field", warped)
 
    cv2.putText(output, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
 
    cv2.imshow("Ball Tracking", output)
    #cv2.imshow("Blue Mask", mask_blue)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
#******************************
# Cleanup
#****************************** 
cv2.destroyAllWindows()
picam2.stop()
 
