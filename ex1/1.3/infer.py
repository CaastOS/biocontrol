import cv2
import torch
import torch.nn as nn
import numpy as np
from FableAPI.fable_init import api
from get_training_data import setup_api, get_first_module, move_smooth_to

# === Constants ===
LIMIT_X_LOW = -85
LIMIT_X_HIGH = 85
LIMIT_Y_LOW = 0
LIMIT_Y_HIGH = 85
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# === Model Definition ===
class RobotArmNet(nn.Module):
    def __init__(self):
        super(RobotArmNet, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.layer_stack(x)

# === Normalization and Denormalization ===
def normalize_pixel_coords(x: int, y: int) -> np.ndarray:
    return np.array([x / CAMERA_WIDTH, y / CAMERA_HEIGHT])

def denormalize_joint_angles(normalized_angles: np.ndarray) -> np.ndarray:
    min_vals = np.array([LIMIT_X_LOW, LIMIT_Y_LOW])
    max_vals = np.array([LIMIT_X_HIGH, LIMIT_Y_HIGH])
    return normalized_angles * (max_vals - min_vals) + min_vals

# === Global ===
target_coords = None
current_x, current_y = 0, 0

def get_click_coords(event, x, y, flags, param):
    global target_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        target_coords = (x, y)
        print(f"New target set at pixel coordinates: {target_coords}")

# === Main Inference Loop ===
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    try:
        print("Setting up Fable API...")
        setup_api(blocking=True)
        arm = get_first_module()
        if arm is None:
            print("No Fable module found. Exiting.")
            exit()

        print("Loading trained model...")
        model = RobotArmNet()
        model.load_state_dict(torch.load('robot_arm_model.pth'))
        model.eval()
        print("Model loaded successfully.")

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        window_name = 'Fable Robot Control - Click to Set Target'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, get_click_coords)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw marker and process if target is set
            if target_coords is not None:
                # Draw marker
                cv2.circle(frame, target_coords, 10, (0, 255, 0), 2)
                cv2.putText(frame, "Target", (target_coords[0] + 15, target_coords[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Normalize input
                norm_coords = normalize_pixel_coords(*target_coords)
                input_tensor = torch.tensor([norm_coords], dtype=torch.float32)

                # Predict
                with torch.no_grad():
                    normalized_output = model(input_tensor).squeeze().numpy()

                # Denormalize output
                predicted_angles = denormalize_joint_angles(normalized_output)
                theta_x, theta_y = predicted_angles

                print(f"Predicted Angles → Theta_X: {theta_x:.2f}, Theta_Y: {theta_y:.2f}")

                # Move the arm
                move_smooth_to(current_x, current_y, theta_x, theta_y, arm)
                current_x, current_y = theta_x, theta_y

                # Clear the target
                target_coords = None

            # Show the camera frame
            cv2.imshow(window_name, frame)

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        api.terminate()
        print("Application closed.")

# Questions
# 1. What happens if you change the relative position of the camera and the robot after starting your code?
# The model will still attempt to predict joint angles based on the pixel coordinates, but the accuracy may be affected if the camera's perspective changes significantly.
# The model was trained with a specific camera setup, so any deviation could lead to incorrect predictions. However, re-running the get_training_data.py script to collect
# new data with the new camera position would help retrain the model for better accuracy.

# 2. In your solution, is learning active all the time?
# No, learning is not active all the time. The model is trained once using the training data collected from the robot arm's movements.

# 3.If not, could you imagine a way to change your solution to have ”active” (online) learning? Would it work?
# Yes, active learning could be implemented by continuously collecting new data during the operation
# of the robot arm and periodically retraining the model with this new data. This would allow the model to adapt to changes in the environment.
