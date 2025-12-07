import cv2
import numpy as np

# Loading the image
img = cv2.imread("football_table.png")
if img is None:
    raise Exception("Image not found!")

# Converting to HSV for color detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV range for purple tape
lower_purple = np.array([120, 30, 30])   # Lower bound
upper_purple = np.array([160, 255, 255])  # Upper bound

# Creating mask for purple areas
mask = cv2.inRange(hsv, lower_purple, upper_purple)

# Blur and clean mask
mask = cv2.medianBlur(mask, 5)

# Finding contours of the purple regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

corners = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # Filtering out
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            corners.append([cx, cy])

# Needed exactly 4 corners
if len(corners) != 4:
    print(f"Error: Expected 4 detected corners, found {len(corners)}. Try tuning HSV values.")
else:
    corners = np.array(corners)

    # Sorting corners [Top-left, Top-right, Bottom-right, Bottom-left]
    corners = corners[np.argsort(corners[:, 1])]  # Sorting by Y (top to bottom)
    top = corners[:2][np.argsort(corners[:2, 0])]  # Sorting first two by X (left to right)
    bottom = corners[2:][np.argsort(corners[2:, 0])]
    ordered_corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    print("Detected field corners (TL, TR, BR, BL):")
    print(ordered_corners)

    # Drawing corners
    for x, y in ordered_corners:
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)

    # Drawing bounding box around field
    pts = ordered_corners.reshape((-1, 1, 2)).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

    # Result
    cv2.imwrite("field_detection_result.jpg", img)
    # Output
    cv2.imshow("Detected Play Field", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
