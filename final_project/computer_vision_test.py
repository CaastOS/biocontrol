import cv2
import numpy as np

# --- Camera Config ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Nothing function for trackbars
def nothing(x):
    pass

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Create a window with trackbars for LAB thresholds
cv2.namedWindow("Calibration")

cv2.createTrackbar("L min", "Calibration", 0, 255, nothing)
cv2.createTrackbar("A min", "Calibration", 0, 255, nothing)
cv2.createTrackbar("B min", "Calibration", 0, 255, nothing)
cv2.createTrackbar("L max", "Calibration", 255, 255, nothing)
cv2.createTrackbar("A max", "Calibration", 255, 255, nothing)
cv2.createTrackbar("B max", "Calibration", 255, 255, nothing)

print("Adjust trackbars until the mask highlights your target color. Press 's' to save values, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Get current trackbar positions
    l_min = cv2.getTrackbarPos("L min", "Calibration")
    a_min = cv2.getTrackbarPos("A min", "Calibration")
    b_min = cv2.getTrackbarPos("B min", "Calibration")
    l_max = cv2.getTrackbarPos("L max", "Calibration")
    a_max = cv2.getTrackbarPos("A max", "Calibration")
    b_max = cv2.getTrackbarPos("B max", "Calibration")

    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])

    mask = cv2.inRange(lab, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show frames
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("Lower LAB:", lower)
        print("Upper LAB:", upper)

cap.release()
cv2.destroyAllWindows()
