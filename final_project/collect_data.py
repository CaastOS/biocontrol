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
FRAME_WIDTH = 640
BASE_SPEED = 60
FRAME_HEIGHT = 480

# --- Parameters ---
K_FORWARD = 0.1
K_TURN = 30
STOPPING_DISTANCE = 50 # How close (in pixels) to get before stopping

# --- LAB Color Ranges (from calibration) ---
GREEN_LOWER = np.array([0,   0,   0])
GREEN_UPPER = np.array([255, 106, 255])

RED_LOWER   = np.array([97,  151, 111])
RED_UPPER   = np.array([212, 213, 219])

BLUE_LOWER  = np.array([0,   137, 0])
BLUE_UPPER  = np.array([255, 255, 106])

# --- Module ---
def find_modules():
    """Discover modules and return the first moduleID (or None)."""
    moduleids = api.discoverModules()
    if not moduleids:
        return None
    return moduleids

api.setup(blocking=True)
wheels = find_modules()[0]

# --- Helper ---
def find_color_center(frame, lower_lab, upper_lab):
    """Finds the center (x, y) of a specified color in a camera frame (LAB)."""
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab_frame, lower_lab, upper_lab)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Filter small noise
        if cv2.contourArea(c) < 10:
            print(cv2.contourArea(c), "is too small, ignoring.")
            return None
        
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)
    return None

def get_system_state(frame):
    """Gets the full state (robot & target) from a camera frame."""
    # take a picture of frame and save it
    cv2.imwrite('frame.jpg', frame)
    target_pos = find_color_center(frame, GREEN_LOWER, GREEN_UPPER)
    blue_pos   = find_color_center(frame, BLUE_LOWER, BLUE_UPPER)
    red_pos    = find_color_center(frame, RED_LOWER, RED_UPPER)

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

def calculate_expert_action(state):
    """Calculates the expert motor commands based on the current state."""
    Rx, Ry, R_angle, Tx, Ty = state
    
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)

    if distance < 50:
        speed = 30
        turn = 15
    else:
        speed = BASE_SPEED
        turn = K_TURN
    
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

    # Calculate turn command based on the angle error
    turn_command = turn * error_angle

    # Combine a constant forward speed with the turn command
    raw_left_speed = speed - turn_command
    raw_right_speed = speed + turn_command

    return [raw_left_speed, raw_right_speed]

# --- MAIN SCRIPT ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Rx', 'Ry', 'R_angle', 'Tx', 'Ty', 'left_motor_speed', 'right_motor_speed']
    writer.writerow(header)

    print(f"Starting data collection for {NUM_TRIALS} trials...")

    for trial in range(NUM_TRIALS):
        print(f"\n--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
        input("Place the target at a new location and press Enter to start trial...")

        while True:
            print("Loop now")
            ret, frame = cap.read()
            if not ret:
                break

            state = get_system_state(frame)

            if state is None:
                api.setSpinSpeed(0, 0, wheels)
                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            action = calculate_expert_action(state)

            log_row = state + action
            writer.writerow(log_row)

            api.setSpinSpeed(-action[0], action[1], wheels)
            
            distance_to_target = math.sqrt((state[3] - state[0])**2 + (state[4] - state[1])**2)
            if distance_to_target < STOPPING_DISTANCE:
                print("Target reached!")
                api.setSpinSpeed(0, 0, wheels)
                time.sleep(1)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("\nData collection complete.")
api.setSpinSpeed(0, 0, wheels)
api.terminate()
cap.release()
cv2.destroyAllWindows()
