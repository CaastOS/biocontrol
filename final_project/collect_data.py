import cv2
import numpy as np
import math
import csv
import time
import random
from FableAPI.fable_init import api

# --- Configuration ---
NUM_TRIALS = 100
OUTPUT_FILE = 'mlp_data_corrected.csv'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- PD Controller Parameters ---
KP = 80.0
KD = 40.0
BASE_SPEED = 40         # The robot's constant forward speed.
MAX_SPEED = 100         # The maximum speed for any motor.
STOPPING_DISTANCE = 25  # How close (in pixels) to get before stopping.

# --- HSV Color Ranges ---
GREEN_LOWER = np.array([0,184, 97])
GREEN_UPPER = np.array([179, 255, 153])
BLUE_LOWER = np.array([0, 167, 194])
BLUE_UPPER = np.array([160, 255, 255])
RED_LOWER = np.array([117, 158, 36])
RED_UPPER = np.array([179, 255, 255])


# --- PD Controller Class ---
class PDController:
    """A simple Proportional-Derivative controller."""
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def calculate(self, error, dt):
        """Calculates the control output based on error and time delta."""
        # Proportional term
        p_term = self.kp * error
        
        # Derivative term (accounts for rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        self.prev_error = error
        return p_term + d_term

# --- Fable Module Setup ---
def find_modules():
    """Discover modules and return the first moduleID (or None)."""
    moduleids = api.discoverModules()
    if not moduleids:
        print("No Fable modules found.")
        return None
    print(f"Found Fable module: {moduleids[0]}")
    return moduleids

api.setup(blocking=True)
wheels = find_modules()[0]

# --- Helper Functions ---
def find_color_center(frame, lower_hsv, upper_hsv):
    """Finds the center (x, y) of a specified color in a camera frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 20:
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
        print(f"Missing: {'Target' if not target_pos else ''} {'Blue' if not blue_pos else ''} {'Red' if not red_pos else ''}")
        return None

    Rx = (blue_pos[0] + red_pos[0]) // 2
    Ry = (blue_pos[1] + red_pos[1]) // 2
    R_angle = math.atan2(blue_pos[1] - red_pos[1], blue_pos[0] - red_pos[0])
    Tx, Ty = target_pos
    return [Rx, Ry, R_angle, Tx, Ty]

def calculate_expert_action(state, controller, dt):
    """Calculates expert motor commands using a PD controller for stable movement."""
    Rx, Ry, R_angle, Tx, Ty = state
    
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
    
    if distance < STOPPING_DISTANCE:
        return [0, 0]

    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    
    while error_angle > math.pi: error_angle -= 2 * math.pi
    while error_angle < -math.pi: error_angle += 2 * math.pi

    # Use the PD controller to get a steering correction value
    steering_correction = controller.calculate(error_angle, dt)
    left_motor_speed = BASE_SPEED - steering_correction
    right_motor_speed = BASE_SPEED + steering_correction

    # Clamp the motor speeds to the maximum allowed values
    left_motor_speed = max(-MAX_SPEED, min(MAX_SPEED, left_motor_speed))
    right_motor_speed = max(-MAX_SPEED, min(MAX_SPEED, right_motor_speed))

    return [left_motor_speed, right_motor_speed]


if wheels is None:
    print("Exiting script because no Fable module was found.")
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    pd_controller = PDController(KP, KD)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Rx', 'Ry', 'R_angle', 'Tx', 'Ty', 'left_motor_speed', 'right_motor_speed']
        writer.writerow(header)

        print(f"Starting data collection for {NUM_TRIALS} trials...")
        
        quit_flag = False
        for trial in range(NUM_TRIALS):
            if quit_flag: break
            
            print(f"\n--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
            input("Place the robot and target, then press Enter to start trial...")
            
            last_time = time.time()
            while True:
                # Calculate delta time (dt) for the PD controller's derivative term
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                ret, frame = cap.read()
                if not ret: break

                state = get_system_state(frame)

                if state is None:
                    api.setSpinSpeed(0, 0, wheels)
                    cv2.imshow("Live Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        quit_flag = True
                        break
                    continue

                action = calculate_expert_action(state, pd_controller, dt)
                
                log_row = state + action
                writer.writerow(log_row)
                api.setSpinSpeed(-action[1], action[0], wheels) # TODO: fix and check with physical robot which motor is left/right
                
                # --- Visualization for debugging ---
                Rx, Ry, R_angle, Tx, Ty = state
                cv2.circle(frame, (int(Rx), int(Ry)), 10, (255, 255, 0), 2)
                cv2.circle(frame, (int(Tx), int(Ty)), 10, (0, 255, 0), 2)
                cv2.line(frame, (int(Rx), int(Ry)), (int(Rx + 50 * math.cos(R_angle)), int(Ry + 50 * math.sin(R_angle))), (255, 0, 0), 2)
                cv2.imshow("Live Feed", frame)
                
                distance_to_target = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
                if distance_to_target < STOPPING_DISTANCE:
                    print("Target reached!")
                    api.setSpinSpeed(0, 0, wheels)
                    time.sleep(1) # Pause before starting the next trial prompt
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit_flag = True
                    break

    # --- Cleanup ---
    print("\nData collection complete.")
    api.setSpinSpeed(0, 0, wheels)
    api.terminate()
    cap.release()
    cv2.destroyAllWindows()
