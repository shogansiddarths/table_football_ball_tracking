import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ble_sender import BLESender

# Initialising the Raspberry Pi Camera

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(1)

def get_frame():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Initialize BLE Sender
ble = BLESender()

# Real-time Ball Tracking

while True:
    img = get_frame()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detecting the Orange Ball
    
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([40, 255, 255])

    mask_ball = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_ball = cv2.medianBlur(mask_ball, 7)

    contours_ball, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_center = None

    for cnt in contours_ball:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # Adjust based on ball size
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            if radius > 5:
                ball_center = (int(x), int(y))

                # Drawing the ball
                cv2.circle(img, ball_center, int(radius), (0, 165, 255), 2)
                cv2.circle(img, ball_center, 5, (0, 255, 255), -1)

                # Printing the coordinates
                print(f"Ball detected at: {ball_center}")

                # Send x and y through BLE
                ble.send_data(x, y, 0)

    # Displaying
    
    cv2.imshow("Ball Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
picam2.stop()
