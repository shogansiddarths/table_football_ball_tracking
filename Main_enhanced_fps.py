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
    main={"size": (384, 216), "format": "BGR888"},
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
field_corners_warped = None
corners_printed = False
frame_count = 0
start_time = time.time()
fps = 0.0
prev_ball_pos_speed = None
prev_time = None 
prev_valid_ball = None
 
# GOAL SETTINGS
goal_width_ratio = 0.12
goal_depth_ratio = 0.09
goal_thickness = 2
 
# SCORE & BALL HISTORY
top_player_goals = 0
bottom_player_goals = 0
prev_ball_pos = None
prev_ball_axis = None
goal_event = 0
goal_cooldown = 1
last_goal_time = 0
goal_axis = "y"  # auto-detected later
 
# Corner Exclusion Radius in warped space
CORNER_EXCLUSION_RADIUS = 20  # pixels around each corner marker to avoid false detections
 
#*********************************
# FIELD & ROD DEFINITIONS
#*********************************
RODS = [
    {"color": "white", "x1": 87,"x2": 76},
    {"color": "white", "x1": 118,"x2": 107},
    {"color": "black", "x1": 147,"x2": 140},
    {"color": "white", "x1": 178,"x2": 170},
    {"color": "black", "x1": 208,"x2": 200},
    {"color": "white", "x1": 238,"x2": 228},
    {"color": "black", "x1": 262,"x2": 262},
    {"color": "black", "x1": 293,"x2": 293},
]
 
TABLE_Y_MIN = 63
TABLE_Y_MAX = 205
 
POSSESSION_X_THRESHOLD = 12 # pixels
PIXELS_PER_CM = None
W_cm = 130  # Real table width in cm
H_cm = 80   # Real table height in cm
FRAME_W = 384
FRAME_H = 216
 
possession_counts = {"white": 0, "black": 0}

#*********************************
# DRAWING THE LINE IN WARPED SPACE
#*********************************
def draw_warped_line(img, pt1, pt2, Minv, color, thickness=2):
    pts = np.array([[pt1, pt2]], dtype="float32")
    pts = cv2.perspectiveTransform(pts, Minv)
    p1 = tuple(pts[0][0].astype(int))
    p2 = tuple(pts[0][1].astype(int))
    cv2.line(img, p1, p2, color, thickness)
 
 
def is_near_corner_marker(wx, wy, corners_warped, radius):
    # To check if point is near any of the 4 corners in warped space
    for corner in corners_warped:
        cx, cy = corner
        dist = np.sqrt((wx - cx)**2 + (wy - cy)**2)
        if dist < radius:
            return True
    return False

#******************************
# PLAYFIELD DETECTION
#******************************
def detect_playfield(frame):
    global M, Minv, field_ready, field_width, field_height, field_corners_camera, field_corners_warped
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # HSV range to match the table's corner in the image
    #lower_purple = np.array([100, 50, 50])
    #upper_purple = np.array([160, 255, 255])
    lower_purple = np.array([120, 80, 80])
    upper_purple = np.array([165, 255, 255])
 
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    mask = cv2.medianBlur(mask, 5)
 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==4:
        print(f"In Playfield: Number of contours detected: {len(contours)}")
 
    corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 75:
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
 
    # Storing the corner positions in warped space
    field_corners_warped = [
        (0, 0),  # Top-left
        (field_width, 0),  # Top-right
        (field_width, field_height),  # Bottom-right
        (0, field_height)  # Bottom-left
    ]
 
    field_ready = True
    calculate_pixels_per_cm()
    return True

#**********************************
# SPEED ESTIMATION (in warped space)
#**********************************
def calculate_pixels_per_cm():
    global PIXELS_PER_CM

    if field_width == 0 or field_height == 0:
        return None

    px_per_cm_x = field_width / W_cm
    px_per_cm_y = field_height / H_cm

    PIXELS_PER_CM = (px_per_cm_x + px_per_cm_y) / 2
    return PIXELS_PER_CM
    
def estimate_speed(ball_center_warped):
    global prev_ball_pos_speed, prev_time
 
    if ball_center_warped is None or PIXELS_PER_CM is None:
        prev_ball_pos_speed = None
        prev_time = None
        return 0
 
    now = time.time()
    speed = 0
 
    if prev_ball_pos_speed is not None:
        dx = ball_center_warped[0] - prev_ball_pos_speed[0]
        dy = ball_center_warped[1] - prev_ball_pos_speed[1]
        dist_px = (dx * dx + dy * dy) ** 0.5
        dt = now - prev_time
 
        if dt > 0:
            speed = (dist_px / PIXELS_PER_CM) / dt
 
    prev_ball_pos_speed = ball_center_warped
    prev_time = now
    return speed
 
#***********************************
# POSSESSION LOGIC (in warped space)
#***********************************
 
def determine_possession(ball_center_warped):
    if ball_center_warped is None:
        return None
 
    wx, wy = ball_center_warped
 
    # Convert warped position back to camera space for rod comparison
    pt = np.array([[[wx, wy]]], dtype="float32")
    camera_pt = cv2.perspectiveTransform(pt, Minv)
    bx, by = int(camera_pt[0][0][0]), int(camera_pt[0][0][1])
 
    if not (TABLE_Y_MIN <= by <= TABLE_Y_MAX):
        return None
 
    closest_rod = None
    min_dx = float("inf")
 
    for rod in RODS:
        rod_x = (rod["x1"] + rod["x2"]) / 2
        dx = abs(bx - rod_x)
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
        print("\n********PLAYFIELD CORNERS********")
 
        print("\nCamera-space corners (pixels):")
        print(f"Top-Left: {field_corners_camera[0]}")
        print(f"Top-Right: {field_corners_camera[1]}")
        print(f"Bottom-Right: {field_corners_camera[2]}")
        print(f"Bottom-Left: {field_corners_camera[3]}")
 
        print("\nWarped-space corners (top-down):")
        for i, corner in enumerate(field_corners_warped):
            print(f"Corner {i}: {corner}")
 
        print(f"\nCorner exclusion radius: {CORNER_EXCLUSION_RADIUS} pixels")
 
        corners_printed = True
 
    ball_center_warped = None
 
    #**************************************
    # BALL DETECTION (ONLY IN WARPED SPACE)
    #**************************************
    if field_ready:
        # First, warp the entire frame to top-down view
        warped = cv2.warpPerspective(frame, M, (field_width, field_height))
 
        # Draw exclusion zones for debugging
        warped_debug = warped.copy()
        for corner in field_corners_warped:
            cv2.circle(warped_debug, corner, CORNER_EXCLUSION_RADIUS, (255, 0, 255), 1)
 
        # Now do ball detection on the warped image
        hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv_warped, lower_blue, upper_blue)
        mask_blue = cv2.medianBlur(mask_blue, 5)
 
        cv2.imshow("Mask", mask_blue) 
 
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_circle = None
        best_score = 0
 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 50:
                continue
 
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
 
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < 0.6:
                continue
 
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r < 0.6 or r > 12:
                continue
 
            score = circularity * area
            if score > best_score:
                best_score = score
                best_circle = (int(x), int(y), int(r))
 
        #******************************
        # PROCESS BALL DETECTION
        #******************************
        if best_circle:
            wx, wy, r = best_circle
 
            # FILTER 1: Ignore detections near corner markers
            if is_near_corner_marker(wx, wy, field_corners_warped, CORNER_EXCLUSION_RADIUS):
                # Silently ignore corner markers - no print spam
                pass
            else:
                # Ball is already in warped space coordinates
                ball_center_warped = (wx, wy)
 
                # Draw circle on warped view
                cv2.circle(warped_debug, (wx, wy), r, (0, 255, 0), 2)
 
                # FILTER 2: Ignore static objects
                if prev_ball_pos is not None:
                    dist = np.linalg.norm(np.array(prev_ball_pos) - np.array([wx, wy]))
                    if dist < 3:  # not moving
                        ball_center_warped = None
 
                if ball_center_warped is not None:
                    prev_ball_pos = (wx, wy)
 
                    # For visualization, draw on camera view too
                    # Convert warped coordinates back to camera space for display
                    pt = np.array([[[wx, wy]]], dtype="float32")
                    camera_pt = cv2.perspectiveTransform(pt, Minv)
                    cam_x, cam_y = int(camera_pt[0][0][0]), int(camera_pt[0][0][1])
                    cv2.circle(output, (cam_x, cam_y), r, (0, 255, 0), 2)
 
                    print(f"Ball: warped=({wx},{wy}) camera=({cam_x},{cam_y}) | FPS:{fps:.1f}")
 
        # Calculate possession and speed using warped coordinates
        if ball_center_warped is not None:
            possession = determine_possession(ball_center_warped)
            if possession:
                possession_counts[possession] += 1
 
            speed = estimate_speed(ball_center_warped)
 
            total = possession_counts["white"] + possession_counts["black"]
            white_pct = (possession_counts["white"] / total * 100) if total else 0
            black_pct = (possession_counts["black"] / total * 100) if total else 0
            possession_code = 1 if possession == "white" else 2 if possession == "black" else 0
 
            #if frame_count % 30 == 0:  # Print stats every 30 frames to reduce spam
            print("\n===============================")
            print(" Foosball Tracking Statistics")
            print("===============================")
            print(f"Ball speed:         {speed:.2f} cm/s")
            print(f"Possession code:    {possession_code}")
            print(f"White possession:   {white_pct:.1f}%")
            print(f"Black possession:   {black_pct:.1f}%")
            print("===============================")
 
        # Drawing the field border in camera view
        draw_warped_line(output, (0, 0), (field_width, 0), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (field_width, 0), (field_width, field_height), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (field_width, field_height), (0, field_height), Minv, (0, 255, 0), 3)
        draw_warped_line(output, (0, field_height), (0, 0), Minv, (0, 255, 0), 3)
 
        #******************************
        # DRAW RODS ON WARPED VIEW
        #******************************
        for rod in RODS:
            # Player position marking on camera view
            color = (255, 255, 255) if rod["color"] == "white" else (0, 0, 0)
            cv2.line(output, (rod["x1"], TABLE_Y_MIN), (rod["x2"], TABLE_Y_MAX), color, 2)
 
            # Player position marking on Warped view
            pts = np.array(
                [[[rod["x1"], TABLE_Y_MIN], [rod["x2"], TABLE_Y_MAX]]],
                dtype="float32"
            )
            wpts = cv2.perspectiveTransform(pts, M)
            cv2.line(
                warped_debug,
                tuple(wpts[0][0].astype(int)),
                tuple(wpts[0][1].astype(int)),
                (255, 255, 255) if rod["color"] == "white" else (0, 0, 0),
                2
            )
 
        #************************************
        # DETECTING GOAL AXIS & DRAWING GOALS
        #************************************
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
 
            cv2.line(warped_debug, (center_x - goal_half_width, top_goal_y),
                     (center_x + goal_half_width, top_goal_y), (0, 0, 255), goal_thickness)
 
            cv2.line(warped_debug, (center_x - goal_half_width, bottom_goal_y),
                     (center_x + goal_half_width, bottom_goal_y), (0, 0, 255), goal_thickness)
 
        else:
            left_goal_x = goal_depth
            right_goal_x = field_width - goal_depth
 
            cv2.line(warped_debug, (left_goal_x, center_y - goal_half_width),
                     (left_goal_x, center_y + goal_half_width), (0, 0, 255), goal_thickness)
 
            cv2.line(warped_debug, (right_goal_x, center_y - goal_half_width),
                     (right_goal_x, center_y + goal_half_width), (0, 0, 255), goal_thickness)
 
        #******************************
        # GOAL DETECTION LOGIC
        #******************************

        if ball_center_warped is not None:
            wx, wy = ball_center_warped
            current_time = time.time()
 
            if prev_valid_ball is not None and (current_time - last_goal_time > goal_cooldown):
                px, py = prev_valid_ball
 
                if goal_axis == "y":
                    '''# printing ball movement
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] Ball Y: prev={py}, curr={wy}, top_goal={top_goal_y}, bottom_goal={bottom_goal_y}")'''
 
                    if py < top_goal_y <= wy:
                        bottom_player_goals += 1
                        goal_event = 2
                        last_goal_time = current_time
                        print(f"\nGOAL -> BLACK PLAYER")
                        #print(f"Ball crossed from Y={py} to Y={wy} (threshold={top_goal_y})\n")
 
                    elif py > bottom_goal_y >= wy:
                        top_player_goals += 1
                        goal_event = 1
                        last_goal_time = current_time
                        print(f"\nGOAL -> WHITE PLAYER")
                        #print(f"Ball crossed from Y={py} to Y={wy} (threshold={bottom_goal_y})\n")
 
                else:
                    '''# printing ball movement
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] Ball X: prev={px}, curr={wx}, left_goal={left_goal_x}, right_goal={right_goal_x}")'''
 
                    if px < left_goal_x <= wx: 
                        bottom_player_goals += 1
                        goal_event = 2
                        last_goal_time = current_time
                        print(f"\nGOAL -> BLACK PLAYER")
                        #print(f"Ball crossed from X={px} to X={wx} (threshold={left_goal_x})\n")
 
                    elif px > right_goal_x >= wx: 
                        top_player_goals += 1
                        goal_event = 1
                        last_goal_time = current_time
                        print(f"\nGOAL -> WHITE PLAYER")
                        #print(f"Ball crossed from X={px} to X={wx} (threshold={right_goal_x})\n")
 
            prev_valid_ball = (wx, wy)
            prev_ball_axis = wy if goal_axis == "y" else wx
 
            ble.send_data(wx, wy, possession_code, bottom_player_goals, top_player_goals, goal_event, int(speed), frame_count)
 
        # SCORE DISPLAY
        cv2.putText(warped_debug, f"WHITE: {top_player_goals}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        cv2.putText(warped_debug, f"BLACK: {bottom_player_goals}", (20, field_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        cv2.putText(warped_debug, f"GOALAXIS: {goal_axis.upper()}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
 
        # Display goal line positions for debugging
        '''if goal_axis == "y":
            cv2.putText(warped_debug, f"Goal Lines: Y={top_goal_y}, Y={bottom_goal_y}",
                        (20, field_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(warped_debug, f"Goal Lines: X={left_goal_x}, X={right_goal_x}",
                        (20, field_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)'''
 
        cv2.imshow("Warped Field", warped_debug)
 
    cv2.putText(output, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
 
    cv2.imshow("Ball Tracking", output)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
#******************************
# Cleanup
#****************************** 
cv2.destroyAllWindows()
picam2.stop()
