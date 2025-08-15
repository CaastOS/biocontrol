import cv2
import numpy as np

# This function does nothing, it's a required callback for the trackbar
def nothing(x):
    pass

# --- Setup ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# --- Create Windows ---
cv2.namedWindow("Live Feed")
cv2.namedWindow("Mask")
cv2.namedWindow("Controls")

# --- Create Trackbars for HSV Range ---
cv2.createTrackbar("H_MIN", "Controls", 0, 179, nothing)
cv2.createTrackbar("H_MAX", "Controls", 179, 179, nothing)
cv2.createTrackbar("S_MIN", "Controls", 0, 255, nothing)
cv2.createTrackbar("S_MAX", "Controls", 255, 255, nothing)
cv2.createTrackbar("V_MIN", "Controls", 0, 255, nothing)
cv2.createTrackbar("V_MAX", "Controls", 255, 255, nothing)

print("\n--- HSV Color Calibrator ---")
print("Adjust sliders until only your target object is white in the 'Mask' window.")
print("Press 'q' to quit.")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    h_min = cv2.getTrackbarPos("H_MIN", "Controls")
    h_max = cv2.getTrackbarPos("H_MAX", "Controls")
    s_min = cv2.getTrackbarPos("S_MIN", "Controls")
    s_max = cv2.getTrackbarPos("S_MAX", "Controls")
    v_min = cv2.getTrackbarPos("V_MIN", "Controls")
    v_max = cv2.getTrackbarPos("V_MAX", "Controls")

    # Create the lower and upper HSV bounds from the trackbar values
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Create a mask using the current bounds
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    # Create a result image that shows the original color where the mask is white
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the windows
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result (Filtered Color)", result)

    # Quit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Print the final values before quitting
        print("\n--- Final HSV Values ---")
        print(f"LOWER_BOUND = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"UPPER_BOUND = np.array([{h_max}, {s_max}, {v_max}])")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
