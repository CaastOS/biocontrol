import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os

# === Normalization Limits ===
LIMIT_X_LOW = -85
LIMIT_X_HIGH = 85
LIMIT_Y_LOW = 0
LIMIT_Y_HIGH = 85

# === Camera Dimensions ===
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

# === Normalization Functions ===
def normalize_pixel_coords(coords: np.ndarray) -> np.ndarray:
    return coords / np.array([CAMERA_WIDTH, CAMERA_HEIGHT])

def normalize_joint_angles(angles: np.ndarray) -> np.ndarray:
    min_vals = np.array([LIMIT_X_LOW, LIMIT_Y_LOW])
    max_vals = np.array([LIMIT_X_HIGH, LIMIT_Y_HIGH])
    return (angles - min_vals) / (max_vals - min_vals)

# === Training Script ===
if __name__ == '__main__':
    print("Loading data...")

    pixel_coords = np.load('X_data_grid.npy')
    joint_angles = np.load('y_data_grid.npy')

    pixel_coords = normalize_pixel_coords(pixel_coords)
    joint_angles = normalize_joint_angles(joint_angles)

    # Convert to tensors
    X_tensor = torch.tensor(pixel_coords, dtype=torch.float32)
    y_tensor = torch.tensor(joint_angles, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Split: 100% training, 0% validation (validation is not needed as it's a grid sampling)
    train_size = int(1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # === Model Setup ===
    print("Initializing model...")
    model = RobotArmNet()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    epochs = 1000
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: N/A")

    torch.save(model.state_dict(), 'robot_arm_model.pth')
    print("\n✅ Training complete. Model saved to 'robot_arm_model.pth'")
