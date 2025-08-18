import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- Configuration ---
DATA_FILE = 'mlp_data.csv'
MODEL_SAVE_PATH = 'robot_controller.pth' # PyTorch model extension
SCALER_SAVE_PATH = 'data_scaler.gz'
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# --- 1. Custom Dataset for PyTorch ---
class RobotDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. Define the MLP Model Architecture ---
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

# --- 3. Load and Prepare the Data ---
print("Loading and preparing data...")
try:
    df = pd.read_csv(DATA_FILE)
    X = df[['distance', 'error_angle']].values
    y = df[['left_motor_speed', 'right_motor_speed']].values
except FileNotFoundError:
    print(f"Error: The data file '{DATA_FILE}' was not found.")
    exit()

# --- 4. Normalize and Split Data ---
print("Normalizing and splitting data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Data scaler saved to '{SCALER_SAVE_PATH}'")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- 5. Create Datasets and DataLoaders ---
train_dataset = RobotDataset(X_train, y_train)
val_dataset = RobotDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data split into {len(X_train)} training samples and {len(X_val)} validation samples.")

# --- 6. Initialize Model, Loss, and Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = RobotControllerMLP().to(device)
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model Architecture:")
print(model)

# --- 7. Training Loop with Early Stopping ---
print("\nStarting model training...")
best_val_loss = float('inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    total_train_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # Validation loop
    model.eval() # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Early stopping and model checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model state
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Validation loss improved. Saving model to '{MODEL_SAVE_PATH}'")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
        break

print("Model training complete.")

# --- 8. Visualize Training History ---
print("Plotting training history...")
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.show()
print("Training plot saved to 'training_history.png'")
