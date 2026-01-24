import cv2
import numpy as np

#STEP 1: Preprocessing the image
img = cv2.imread(r"C:\Users\shoga\Desktop\table_football_ball_tracking\webcam_implementation\football_table.png")
if img is None:
    raise Exception("Image not found!")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#STEP 2: Detecting the Field Corners (Purple Tape)
lower_purple = np.array([120, 30, 30])   # Lower HSV bound
upper_purple = np.array([160, 255, 255])  # Upper HSV bound

mask = cv2.inRange(hsv, lower_purple, upper_purple)
mask = cv2.medianBlur(mask, 5)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

corners = []
for cnt in contours:
    if cv2.contourArea(cnt) > 100:  # Filtering
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            corners.append([cx, cy])

#STEP 3: Checking if exactly 4 corners are detected
if len(corners) != 4:
    print(f"Error: Expected 4 detected corners, found {len(corners)}. Try tuning HSV values.")
else:
    corners = np.array(corners)

    # Sorting corners [TL, TR, BR, BL]
    corners = corners[np.argsort(corners[:, 1])]  # By Y (top-bottom)
    top = corners[:2][np.argsort(corners[:2, 0])]  # First two (top) by X
    bottom = corners[2:][np.argsort(corners[2:, 0])]  # Bottom two (bottom) by X
    ordered_corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    print("Detected field corners (TL, TR, BR, BL):")
    print(ordered_corners)

    #STEP 4: Drawing the corners on the original image
    for x, y in ordered_corners:
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)

    pts = ordered_corners.reshape((-1, 1, 2)).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

    cv2.imwrite("field_detection_result.jpg", img)
    cv2.imshow("Detected Play Field", img)

    #STEP 5: Perspective Transformation
    width = int(max(
        np.linalg.norm(ordered_corners[0] - ordered_corners[1]),   # Top width
        np.linalg.norm(ordered_corners[2] - ordered_corners[3])    # Bottom width
    ))
    height = int(max(
        np.linalg.norm(ordered_corners[0] - ordered_corners[3]),   # Left height
        np.linalg.norm(ordered_corners[1] - ordered_corners[2])    # Right height
    ))

    dst_corners = np.array([
        [0, 0],                  # Top-Left
        [width - 1, 0],          # Top-Right
        [width - 1, height - 1], # Bottom-Right
        [0, height - 1]          # Bottom-Left
    ], dtype="float32")

    # Computing transforms
    M = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    warped = cv2.warpPerspective(img, M, (width, height))

    #STEP 6: Saving the play field
    cv2.imwrite("cropped_field.jpg", warped)
    cv2.imshow("Cropped Field", warped)
    
    print(f"Cropped image saved as 'cropped_field.jpg' with size: {width}x{height}")

    #STEP 7: Defining the coordinates directly
    new_corners = {
        "Top-Left (0,0)": (0, 0),
        "Top-Right": (width - 1, 0),
        "Bottom-Right": (width - 1, height - 1),
        "Bottom-Left": (0, height - 1)
    }

    print("\n New Field Coordinates (after cropping):")
    for name, coord in new_corners.items():
        print(f"{name} → {coord}")

    # Drawing corners on the warped output
    for name, (x, y) in new_corners.items():
        cv2.circle(warped, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(warped, name, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Cropped Field with Coordinates", warped)
    cv2.imwrite("Cropped Field with Coordinates.jpg", warped)

#STEP 8: Detecting the Orange Ball in the Original Image
lower_orange = np.array([5, 120, 120])
upper_orange = np.array([20, 255, 255])
mask_ball = cv2.inRange(hsv, lower_orange, upper_orange)
mask_ball = cv2.medianBlur(mask_ball, 7)

contours_ball, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ball_center_warped = None

for cnt in contours_ball:
    area = cv2.contourArea(cnt)
    if 100 < area < 5000:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:
            # Converting the ball center from original to warped image
            pt = np.array([[x, y]], dtype='float32')
            pt = np.array([pt])
            warped_pt = cv2.perspectiveTransform(pt, M)
            ball_center_warped = (int(warped_pt[0][0][0]), int(warped_pt[0][0][1]))

            # Drawing on warped image
            cv2.circle(warped, ball_center_warped, int(radius), (0, 165, 255), 2)
            cv2.circle(warped, ball_center_warped, 5, (0, 255, 255), -1)
            print(f"Ball in warped image: {ball_center_warped}")

    
    cv2.imwrite("football_detection_result.jpg", warped)
    cv2.imshow("Final Detection", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
