import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import joblib
from FableAPI.fable_init import api

# Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STOPPING_DISTANCE = 25
MODEL_PATH = 'robot_controller.pth'
SCALER_PATH = 'data_scaler.gz'
CMAC_WEIGHTS_FILE = 'cmac_weights.npy'
BASE_SPEED = 60
K_TURN = 30

# LAB Color Ranges
GREEN_LOWER = np.array([0,   0,   0]); GREEN_UPPER = np.array([255, 106, 255])
RED_LOWER   = np.array([97,  151, 111]); RED_UPPER   = np.array([212, 213, 219])
BLUE_LOWER  = np.array([0,   0, 0]); BLUE_UPPER  = np.array([137, 170, 97])

# CMAC Implementation
class CMAC:
    def __init__(self, n_inputs, n_outputs, memory_size, generalization_factor, learning_rate):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.memory_size = memory_size
        self.generalization_factor = generalization_factor
        self.learning_rate = learning_rate
        self.weights = np.zeros((memory_size, n_outputs))
        self.input_ranges = np.array([[0, 800], [-np.pi, np.pi]])
        self.quantization_levels = (memory_size // generalization_factor)
        self.quantization_step = (self.input_ranges[:, 1] - self.input_ranges[:, 0]) / self.quantization_levels

    def _get_active_cell_indices(self, inputs):
        indices = []
        for i in range(self.generalization_factor):
            displaced_inputs = inputs - (i * self.quantization_step / self.generalization_factor)
            quantized_inputs = np.floor((displaced_inputs - self.input_ranges[:, 0]) / self.quantization_step).astype(int)
            hashed_index = sum(quantized_inputs * (self.quantization_levels ** np.arange(self.n_inputs)))
            indices.append(hashed_index % self.memory_size)
        return np.array(indices)

    def predict(self, inputs):
        active_indices = self._get_active_cell_indices(inputs)
        return np.sum(self.weights[active_indices], axis=0) / self.generalization_factor

    def train(self, inputs, target_error):
        active_indices = self._get_active_cell_indices(inputs)
        current_prediction = self.predict(inputs)
        cmac_training_error = target_error - current_prediction
        update_value = (self.learning_rate * cmac_training_error) / self.generalization_factor
        self.weights[active_indices] += update_value

# MLP Model Architecture
class RobotControllerMLP(nn.Module):
    def __init__(self):
        super(RobotControllerMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.layers(x)

# Helper and Ground Truth Functions
def find_modules():
    moduleids = api.discoverModules(); return moduleids[0] if moduleids else None

def find_color_center(frame, lower, upper):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB); mask = cv2.inRange(lab, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c); return (x + w // 2, y + h // 2)
    return None

def get_system_state(frame):
    target_pos = find_color_center(frame, GREEN_LOWER, GREEN_UPPER)
    blue_pos   = find_color_center(frame, BLUE_LOWER, BLUE_UPPER)
    red_pos    = find_color_center(frame, RED_LOWER, RED_UPPER)
    if not all([target_pos, blue_pos, red_pos]): return None
    Rx = (blue_pos[0] + red_pos[0]) // 2; Ry = (blue_pos[1] + red_pos[1]) // 2
    R_angle = math.atan2(blue_pos[1] - red_pos[1], blue_pos[0] - red_pos[0])
    Tx, Ty = target_pos
    return [Rx, Ry, R_angle, Tx, Ty]

def calculate_expert_action(allocentric_state):
    Rx, Ry, R_angle, Tx, Ty = allocentric_state
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
    speed, turn = (30, 15) if distance < 50 else (BASE_SPEED, K_TURN)
    if distance < STOPPING_DISTANCE: return np.array([0, 0])
    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    if error_angle > math.pi: error_angle -= 2 * math.pi
    elif error_angle < -math.pi: error_angle += 2 * math.pi
    turn_command = turn * error_angle
    return np.array([speed - turn_command, speed + turn_command])

def calculate_egocentric_state(allocentric_state):
    Rx, Ry, R_angle, Tx, Ty = allocentric_state
    distance = math.sqrt((Tx - Rx)**2 + (Ty - Ry)**2)
    target_angle = math.atan2(Ty - Ry, Tx - Rx)
    error_angle = target_angle - R_angle
    if error_angle > math.pi: error_angle -= 2 * math.pi
    elif error_angle < -math.pi: error_angle += 2 * math.pi
    return np.array([distance, error_angle])

# Setup
api.setup(blocking=True); wheels = find_modules()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Load MLP Model and Scaler
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobotControllerMLP().to(device); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval()
    scaler = joblib.load(SCALER_PATH)
    print("MLP model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model and scaler files are present."); exit()

# Initialize CMAC
cmac = CMAC(n_inputs=2, n_outputs=2, memory_size=2048, generalization_factor=16, learning_rate=0.1)
print("CMAC initialized for training.")

# Training Loop
input("\nPlace robot and target. Press Enter to start training...")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    allocentric_state = get_system_state(frame)
    if allocentric_state is None:
        api.setSpinSpeed(0, 0, wheels); cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # Get state for controllers
    egocentric_state = calculate_egocentric_state(allocentric_state)
    
    # Get MLP's prediction
    input_tensor = torch.tensor(scaler.transform([egocentric_state]), dtype=torch.float32).to(device)
    with torch.no_grad(): mlp_prediction = model(input_tensor).cpu().numpy()[0]
    
    # Get CMAC's correction prediction
    cmac_correction = cmac.predict(egocentric_state)

    # Combine predictions to drive the robot
    final_action = mlp_prediction + cmac_correction
    api.setSpinSpeed(-int(final_action[0]), int(final_action[1]), wheels)
    
    # Get 'ground truth' to calculate error for training
    perfect_action = calculate_expert_action(allocentric_state)
    
    # Calculate the MLP's error (this is the CMAC's target)
    mlp_error = perfect_action - mlp_prediction
    
    # Train the CMAC online
    cmac.train(egocentric_state, mlp_error)
    
    # Display feed and handle exit
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        np.save(CMAC_WEIGHTS_FILE, cmac.weights)
        break
    elif key == ord('s'):
        np.save(CMAC_WEIGHTS_FILE, cmac.weights)
        print(f"\nCMAC weights saved to {CMAC_WEIGHTS_FILE}")
        break

# Cleanup
print("\nExiting training script.")
api.setSpinSpeed(0, 0, wheels); api.terminate()
cap.release(); cv2.destroyAllWindows()
