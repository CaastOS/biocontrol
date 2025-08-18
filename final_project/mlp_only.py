import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import joblib

from FableAPI.fable_init import api

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STOPPING_DISTANCE = 25
MODEL_PATH = 'robot_controller.pth'
SCALER_PATH = 'data_scaler.gz'

# --- LAB Color Ranges ---
GREEN_LOWER = np.array([0,   0,   0])
GREEN_UPPER = np.array([255, 106, 255])
RED_LOWER   = np.array([97,  151, 111])
RED_UPPER   = np.array([212, 213, 219])
BLUE_LOWER  = np.array([0,   0, 0])
BLUE_UPPER  = np.array([137, 170, 97])

# --- Define the MLP Model Architecture ---
class RobotControllerMLP(nn.Module):
    def __init__(self):
        super(RobotControllerMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

# --- Module Discovery ---
def find_modules():
    """Discover modules and return the first moduleID (or None)."""
    moduleids = api.discoverModules()
    if not moduleids:
        return None
    return moduleids

# --- Helper Functions ---
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

def calculate_egocentric_state(allocentric_state):
    """Calculates the egocentric state from the allocentric state."""
    Rx, Ry, R_angle, Tx, Ty = allocentric_state
    
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    
    if error_angle > math.pi: error_angle -= 2 * math.pi
    elif error_angle < -math.pi: error_angle += 2 * math.pi
    
    return [distance, error_angle]

# --- Setup ---
api.setup(blocking=True)
wheels = find_modules()[0]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# --- Load the Trained Model and Scaler ---
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RobotControllerMLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set the model to evaluation mode

    scaler = joblib.load(SCALER_PATH)
    print("MLP model and scaler loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print("Please ensure 'robot_controller.pth' and 'data_scaler.gz' are in the same directory.")
    api.terminate()
    cap.release()
    exit()


# --- Control Loop ---
print("\n--- Starting Robot Control ---")
input("Place the robot and target, then press Enter to start...")

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

    # 2. Calculate the egocentric state for the model input
    egocentric_state = calculate_egocentric_state(allocentric_state)
    
    # 3. Use the MLP to predict the motor commands
    #    a. Reshape and scale the input data
    input_data = np.array([egocentric_state]) # Shape to (1, 2)
    scaled_input = scaler.transform(input_data)
    
    #    b. Convert to a tensor and make a prediction
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_action = model(input_tensor).cpu().numpy()[0]
    
    left_speed, right_speed = predicted_action
        
    # 4. Drive the robot with the predicted action
    #    Convert speeds to integers to ensure API compatibility
    int_left_speed = int(left_speed)
    int_right_speed = int(right_speed)
    api.setSpinSpeed(-int_left_speed, int_right_speed, wheels)
    
    # 5. Check if the trial is over
    distance_to_target = egocentric_state[0]
    if distance_to_target < STOPPING_DISTANCE:
        print("Target reached!")
        api.setSpinSpeed(0, 0, wheels)
        input("Target reached. Reposition and press Enter to run again, or 'q' in the window to quit.")


    # Allow quitting with 'q'
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\nExiting program.")
api.setSpinSpeed(0, 0, wheels)
api.terminate()
cap.release()
cv2.destroyAllWindows()
