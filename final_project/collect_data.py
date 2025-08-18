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
FRAME_HEIGHT = 480
BASE_SPEED = 60
STOPPING_DISTANCE = 25
NOISE_LEVEL = 5.0

# --- Parameters ---
K_TURN = 30

# --- LAB Color Ranges ---
GREEN_LOWER = np.array([0,   0,   0])
GREEN_UPPER = np.array([255, 106, 255])
RED_LOWER   = np.array([97,  151, 111])
RED_UPPER   = np.array([212, 213, 219])
BLUE_LOWER  = np.array([0,   0, 0])
BLUE_UPPER  = np.array([137, 170, 97])

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
        if cv2.contourArea(c) < 10:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)
    return None

def get_system_state(frame):
    """Gets the full state (robot & target) from a camera frame."""
    target_pos = find_color_center(frame, GREEN_LOWER, GREEN_UPPER)
    blue_pos   = find_color_center(frame, BLUE_LOWER, BLUE_UPPER)
    red_pos    = find_color_center(frame, RED_LOWER, RED_UPPER)

    if not all([target_pos, blue_pos, red_pos]):
        print(f"Red: {red_pos}, Blue: {blue_pos}, Target: {target_pos}")
        return None # Can't see all objects

    Rx = (blue_pos[0] + red_pos[0]) // 2
    Ry = (blue_pos[1] + red_pos[1]) // 2
    R_angle = math.atan2(blue_pos[1] - red_pos[1], blue_pos[0] - red_pos[0])
    Tx, Ty = target_pos

    return [Rx, Ry, R_angle, Tx, Ty]

def calculate_expert_action(allocentric_state):
    """
    Calculates the expert motor commands and the egocentric state.
    Returns: ([left_speed, right_speed], [distance, error_angle])
    """
    Rx, Ry, R_angle, Tx, Ty = allocentric_state
    
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)

    if distance < 50:
        speed, turn = 30, 15
    else:
        speed, turn = BASE_SPEED, K_TURN
    
    if distance < STOPPING_DISTANCE:
        return [0, 0], [distance, 0]

    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    
    if error_angle > math.pi: error_angle -= 2 * math.pi
    elif error_angle < -math.pi: error_angle += 2 * math.pi

    turn_command = turn * error_angle
    raw_left_speed = speed - turn_command
    raw_right_speed = speed + turn_command

    return [raw_left_speed, raw_right_speed], [distance, error_angle]

# --- MAIN SCRIPT ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['distance', 'error_angle', 'left_motor_speed', 'right_motor_speed']
    writer.writerow(header)

    print(f"Starting data collection for {NUM_TRIALS} trials...")

    for trial in range(NUM_TRIALS):
        print(f"\n--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
        input("Place the robot and target, then press Enter to start trial...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Get the global (allocentric) state from the camera
            allocentric_state = get_system_state(frame)

            # If we can't see all markers, stop the robot and wait
            if allocentric_state is None:
                api.setSpinSpeed(0, 0, wheels)
                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # 2. Calculate the expert's ideal action AND the egocentric state
            expert_action, egocentric_state = calculate_expert_action(allocentric_state)
            
            # 3. Log the simple egocentric state and the ideal expert action
            log_row = egocentric_state + expert_action
            writer.writerow(log_row)

            # 4. Add noise to the action to drive the robot for better data
            noisy_action = [
                expert_action[0] + random.uniform(-NOISE_LEVEL, NOISE_LEVEL),
                expert_action[1] + random.uniform(-NOISE_LEVEL, NOISE_LEVEL)
            ]
            api.setSpinSpeed(-noisy_action[0], noisy_action[1], wheels)
            
            # 5. Check if the trial is over
            distance_to_target = egocentric_state[0] # Use the already calculated distance
            if distance_to_target < STOPPING_DISTANCE:
                print("Target reached!")
                api.setSpinSpeed(0, 0, wheels)
                time.sleep(1) # Pause before the next trial
                break

            # Allow quitting with 'q'
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Another quit check between trials
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("\nData collection complete.")
api.setSpinSpeed(0, 0, wheels)
api.terminate()
cap.release()
cv2.destroyAllWindows()
