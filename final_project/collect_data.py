import cv2
import numpy as np
import math
import csv
import time
import random
from FableAPI.fable_init import api

# --- Configuration ---
NUM_TRIALS = 100
OUTPUT_FILE = 'mlp_data.csv'
FRAME_WIDTH = 640 # TODO check with new camera
FRAME_HEIGHT = 480 # TODO check with new camera

# --- Parameters ---
K_FORWARD = 0.5
K_TURN = 1.2
STOPPING_DISTANCE = 15 # How close (in pixels) to get before stopping

# --- HSV Color Ranges ---
GREEN_LOWER = np.array([0,184, 97])
GREEN_UPPER = np.array([179, 255, 153])
BLUE_LOWER = np.array([0, 167, 194])
BLUE_UPPER = np.array([160, 255, 255])
RED_LOWER = np.array([117, 158, 36])
RED_UPPER = np.array([179, 255, 255])


# Module
def find_modules():
    """Discover modules and return the first moduleID (or None)."""
    moduleids = api.discoverModules()
    if not moduleids:
        return None
    return moduleids

api.setup(blocking=True)
wheels = find_modules()[0]

# Helper

def find_color_center(frame, lower_hsv, upper_hsv, is_red=False):
    """Finds the center (x, y) of a specified color in a camera frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Only consider contours of a reasonable size to filter out noise
        if cv2.contourArea(c) < 10:
            return None
        
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)
    return None

def get_system_state(frame):
    """Gets the full state (robot & target) from a camera frame."""
    target_pos = find_color_center(frame, GREEN_LOWER, GREEN_UPPER)
    blue_pos = find_color_center(frame, BLUE_LOWER, BLUE_UPPER)
    red_pos = find_color_center(frame, RED_LOWER, RED_UPPER)

    if not all([target_pos, blue_pos, red_pos]):
        print("Red: {}, Blue: {}, Target: {}".format(red_pos, blue_pos, target_pos))
        return None # Can't see all objects

    # Robot's position is the midpoint between the blue and red markers
    Rx = (blue_pos[0] + red_pos[0]) // 2
    Ry = (blue_pos[1] + red_pos[1]) // 2

    # Robot's angle is the angle from the back (red) to the front (blue)
    R_angle = math.atan2(blue_pos[1] - red_pos[1], blue_pos[0] - red_pos[0])

    Tx, Ty = target_pos

    return [Rx, Ry, R_angle, Tx, Ty]


def calculate_expert_action(state): # THIS IS GPT GENERATED AND WRONG! NEEDS TO BE FIXED ASAP
    """Calculates the expert motor commands based on the current state."""
    Rx, Ry, R_angle, Tx, Ty = state
    
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
    
    # If we are close enough, stop
    if distance < STOPPING_DISTANCE:
        return [0, 0]

    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    
    # Handle angle wrap-around
    if error_angle > math.pi:
        error_angle -= 2 * math.pi
    elif error_angle < -math.pi:
        error_angle += 2 * math.pi

    # Calculate motor speeds
    forward_command = K_FORWARD * distance
    turn_command = K_TURN * error_angle

    # Combine commands for the motors
    left_motor_speed = forward_command - turn_command
    right_motor_speed = forward_command + turn_command

    return [left_motor_speed, right_motor_speed]


# MAIN SCRIPT
# --- Setup Video Capture and CSV File ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header for the CSV file
    header = ['Rx', 'Ry', 'R_angle', 'Tx', 'Ty', 'left_motor_speed', 'right_motor_speed']
    writer.writerow(header)

    print(f"Starting data collection for {NUM_TRIALS} trials...")

    for trial in range(NUM_TRIALS):
        print(f"\n--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
        input("Place the target at a new location and press Enter to start trial...")
        
        # --- Real-time Control Loop ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. PERCEIVE THE STATE from the camera frame
            state = get_system_state(frame)

            if state is None:
                # Can't see everything, so stop the robot
                api.setSpinSpeed(0, 0, wheels)
                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue # Skip to the next loop iteration

            # 2. CALCULATE THE EXPERT ACTION
            action = calculate_expert_action(state)
            
            # 3. LOG THE DATA to the CSV file
            log_row = state + action
            writer.writerow(log_row)

            # 4. ACT by sending the command to the robot
            api.setSpinSpeed(-action[1], action[0], wheels)
            
            # Check for completion of the trial
            distance_to_target = math.sqrt((state[3] - state[0])**2 + (state[4] - state[1])**2)
            if distance_to_target < STOPPING_DISTANCE:
                print("Target reached!")
                api.setSpinSpeed(0, 0, wheels)
                time.sleep(1)
                break
            
            # Press 'q' to quit the whole script early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Break the outer loop if 'q' was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("\nData collection complete.")
api.setSpinSpeed(0, 0, wheels)
api.terminate()
cap.release()
cv2.destroyAllWindows()
