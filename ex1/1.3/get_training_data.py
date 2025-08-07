from FableAPI.fable_init import api
import time
import numpy as np
import camera_tools as ct
from typing import Optional, Tuple, List

# === Constants ===
LIMIT_X_LOW = -85
LIMIT_X_HIGH = 85
LIMIT_Y_LOW = 0
LIMIT_Y_HIGH = 85

STEP_SIZE = 7
STEP_DELAY = 0.005
MOVE_POLL_INTERVAL = 0.05

# === Data containers ===
X_all = []
y_all = []

# === API Setup ===
def setup_api(blocking: bool = True) -> None:
    api.setup(blocking=blocking)

def discover_modules() -> Optional[List]:
    module_ids = api.discoverModules()
    return module_ids if module_ids else None

def get_first_module() -> Optional[int]:
    modules = discover_modules()
    return modules[0] if modules else None

# === Helper Functions ===
def generate_grid_points(x_low, x_high, y_low, y_high, step) -> List[Tuple[int, int]]:
    xs = list(range(x_low, x_high + 1, step))
    ys = list(range(y_low, y_high + 1, step))
    points = [(x, y) for x in xs for y in ys]
    return points

def move_smooth_to(x1: int, y1: int, x2: int, y2: int, module_id: int, num_steps: int = 100) -> None:
    step_x = (x2 - x1) / num_steps
    step_y = (y2 - y1) / num_steps

    for i in range(1, num_steps + 1):
        current_x = x1 + step_x * i
        current_y = y1 + step_y * i
        api.setPos(current_x, current_y, module_id)
        time.sleep(STEP_DELAY)

def wait_until_stopped(module_id: int) -> bool:
    time.sleep(0.25)  # wait for arm to settle, API status unreliable
    return True

def move_to_point_and_collect(module_id: int, cam, target_x: int, target_y: int) -> bool:
    print(f"Target chosen: x={target_x}, y={target_y}")

    move_smooth_to(0, 0, target_x, target_y, module_id)
    wait_until_stopped(module_id)

    img = ct.capture_image(cam)
    x, y = ct.locate(img)

    if x != 0 and y != 0:
        X_all.append([x, y])
        y_all.append([target_x, target_y])

    move_smooth_to(target_x, target_y, 0, 0, module_id)
    wait_until_stopped(module_id)
    print("Movement complete.\n")
    return True

# === Main Execution ===
if __name__ == "__main__":
    try:
        setup_api(blocking=True)
        arm = get_first_module()
        if arm is None:
            raise SystemExit("No modules found. Exiting.")

        print("Found arm module:", arm)

        cam = ct.prepare_camera()
        print("Waiting for camera to stabilize...")
        while True:
            img = ct.capture_image(cam)
            x, y = ct.locate(img)
            if x is not None:
                break
        print("Camera is ready!")

        grid_points = generate_grid_points(LIMIT_X_LOW, LIMIT_X_HIGH, LIMIT_Y_LOW, LIMIT_Y_HIGH, STEP_SIZE)
        print(f"Total points to sample: {len(grid_points)}")

        for i, (tx, ty) in enumerate(grid_points):
            print(f"--- Iteration {i + 1} of {len(grid_points)} ---")
            move_to_point_and_collect(arm, cam, tx, ty)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Saving collected data...")

        output_prefix = "data_grid"
        np.save(f'X_{output_prefix}.npy', np.array(X_all))
        np.save(f'y_{output_prefix}.npy', np.array(y_all))

        try:
            cam.release()
        except:
            pass

        api.terminate()
        print("API closed. Data collection complete.")
