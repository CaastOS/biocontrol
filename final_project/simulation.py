import pygame
import math
import random
import numpy as np

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Colors (Light Theme)
BACKGROUND_COLOR = (240, 245, 250)
TEXT_COLOR = (20, 20, 20)
TARGET_COLOR = (40, 200, 120)
TARGET_ACCENT = (255, 255, 255)
ROBOT_BODY = (80, 120, 220)
ROBOT_ACCENT = (255, 190, 0)
WHEEL_COLOR = (50, 50, 50)
WHEEL_HUB_COLOR = (150, 150, 150)
WHITE = (255, 255, 255)


# Robot properties
ROBOT_WIDTH = 40
ROBOT_HEIGHT = 35 # Slightly taller for the new design
WHEEL_RADIUS = 9
MAX_MOTOR_SPEED = 100.0  # Corresponds to 100%
SPEED_CONVERSION_FACTOR = 1 # Converts motor speed (0-100) to simulation velocity

# Simulation physics
FRICTION_COEFFICIENT = 0.03 # Simple linear friction
POSITION_NOISE = 0.5       # Max random offset per frame for position
ANGLE_NOISE = 0.01         # Max random offset per frame for angle

# --- PD Controller Class ---
class PDController:
    """A simple Proportional-Derivative controller."""
    def __init__(self, kp, kd):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0.0

    def calculate(self, error, dt):
        """Calculates the control output."""
        p_term = self.kp * error
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        self.prev_error = error
        return p_term + d_term

# --- Robot Class ---
class Robot:
    """Represents the two-wheeled robot."""
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle  # In radians
        self.width = ROBOT_WIDTH
        self.height = ROBOT_HEIGHT
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

    def update(self, left_motor_speed, right_motor_speed, dt):
        """Updates the robot's state based on motor speeds and physics."""
        # --- CHANGE: Removed redundant clamping from here. It's now handled in the main loop.
        v_left = left_motor_speed * SPEED_CONVERSION_FACTOR
        v_right = right_motor_speed * SPEED_CONVERSION_FACTOR

        self.linear_velocity = (v_left + v_right) / 2.0
        self.angular_velocity = (v_right - v_left) / self.width

        self.linear_velocity *= (1.0 - FRICTION_COEFFICIENT)
        self.angular_velocity *= (1.0 - FRICTION_COEFFICIENT)

        dx = self.linear_velocity * math.cos(self.angle) * dt
        dy = self.linear_velocity * math.sin(self.angle) * dt
        d_angle = self.angular_velocity * dt

        self.x += dx
        self.y += dy
        self.angle += d_angle
        
        self.x += random.uniform(-POSITION_NOISE, POSITION_NOISE)
        self.y += random.uniform(-POSITION_NOISE, POSITION_NOISE)
        self.angle += random.uniform(-ANGLE_NOISE, ANGLE_NOISE)

        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

# --- Perspective Projection Function ---
def project(x, y, z=0):
    """
    Applies a simple perspective projection to 2D coordinates.
    The 'y' coordinate is treated as depth.
    """
    vanishing_point_y = 0
    scale = (y / SCREEN_HEIGHT) * 1.5 + 0.5
    scale = max(0.1, min(2.0, scale))
    projected_x = x
    projected_y = vanishing_point_y + (y - vanishing_point_y) * (1 + (y / SCREEN_HEIGHT) * 0.6)
    return (projected_x, projected_y), scale

# --- Drawing Functions ---
def draw_robot(screen, robot):
    """Draws the visually updated robot with perspective."""
    (proj_x, proj_y), scale = project(robot.x, robot.y)
    
    scaled_width = robot.width * scale
    scaled_height = robot.height * scale
    scaled_wheel_radius = WHEEL_RADIUS * scale

    # Define a cooler chassis shape (points relative to robot center)
    chassis_points = [
        (scaled_height * 0.6, 0),             # Front point
        (scaled_height * 0.2, -scaled_width * 0.5), # Front left corner
        (-scaled_height * 0.4, -scaled_width * 0.55), # Rear left corner
        (-scaled_height * 0.6, 0),             # Rear center
        (-scaled_height * 0.4, scaled_width * 0.55),  # Rear right corner
        (scaled_height * 0.2, scaled_width * 0.5)  # Front right corner
    ]

    # Rotate and translate chassis points
    rotated_chassis = []
    for x, y in chassis_points:
        # Note: Swapped x/y to align shape with robot's forward direction
        rx = y * math.cos(robot.angle) - x * math.sin(robot.angle) + proj_x
        ry = y * math.sin(robot.angle) + x * math.cos(robot.angle) + proj_y
        rotated_chassis.append((rx, ry))

    # Draw wheels first (so they appear underneath the chassis)
    wheel_offset_x = -scaled_height * 0.1
    wheel_offset_y = scaled_width * 0.55
    
    # Left wheel
    lx_rel = wheel_offset_y * math.cos(robot.angle) - wheel_offset_x * math.sin(robot.angle)
    ly_rel = wheel_offset_y * math.sin(robot.angle) + wheel_offset_x * math.cos(robot.angle)
    pygame.draw.circle(screen, WHEEL_COLOR, (int(proj_x - lx_rel), int(proj_y - ly_rel)), int(scaled_wheel_radius))
    pygame.draw.circle(screen, WHEEL_HUB_COLOR, (int(proj_x - lx_rel), int(proj_y - ly_rel)), int(scaled_wheel_radius * 0.5))

    # Right wheel
    pygame.draw.circle(screen, WHEEL_COLOR, (int(proj_x + lx_rel), int(proj_y + ly_rel)), int(scaled_wheel_radius))
    pygame.draw.circle(screen, WHEEL_HUB_COLOR, (int(proj_x + lx_rel), int(proj_y + ly_rel)), int(scaled_wheel_radius * 0.5))

    # Draw chassis
    pygame.draw.polygon(screen, ROBOT_BODY, rotated_chassis)
    pygame.draw.polygon(screen, (0,0,0), rotated_chassis, 2) # Black outline

    # Draw front sensor/light
    front_light_offset = scaled_height * 0.5
    light_x = proj_x + front_light_offset * math.cos(robot.angle)
    light_y = proj_y + front_light_offset * math.sin(robot.angle)
    pygame.draw.circle(screen, ROBOT_ACCENT, (int(light_x), int(light_y)), int(scale * 5))


def draw_target(screen, target_pos):
    """Draws the visually updated target with perspective."""
    (proj_x, proj_y), scale = project(target_pos[0], target_pos[1])
    radius = int(15 * scale)
    pygame.draw.circle(screen, TARGET_COLOR, (int(proj_x), int(proj_y)), radius)
    pygame.draw.circle(screen, TARGET_ACCENT, (int(proj_x), int(proj_y)), int(radius * 0.6))
    pygame.draw.circle(screen, TEXT_COLOR, (int(proj_x), int(proj_y)), radius, 2)


# --- Main Simulation Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot PD Control Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    robot_start_x = random.randint(100, SCREEN_WIDTH - 100)
    robot_start_y = random.randint(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - 100)
    robot_start_angle = random.uniform(-math.pi, math.pi)
    robot = Robot(robot_start_x, robot_start_y, robot_start_angle)

    target_pos = (random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT // 2))

    pd_controller = PDController(kp=80.0, kd=40.0)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                target_pos = pygame.mouse.get_pos()

        # --- Control Loop ---
        dx = target_pos[0] - robot.x
        dy = target_pos[1] - robot.y
        distance_to_target = math.sqrt(dx**2 + dy**2)

        angle_to_target = math.atan2(dy, dx)

        error = angle_to_target - robot.angle
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        left_motor_speed = 0
        right_motor_speed = 0

        if distance_to_target > 20:
            steering_correction = pd_controller.calculate(error, dt)
            base_speed = 60
            left_motor_speed = base_speed - steering_correction
            right_motor_speed = base_speed + steering_correction

            left_motor_speed = max(0, min(MAX_MOTOR_SPEED, left_motor_speed))
            right_motor_speed = max(0, min(MAX_MOTOR_SPEED, right_motor_speed))
        
        robot.update(left_motor_speed, right_motor_speed, dt)

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        draw_target(screen, target_pos)
        draw_robot(screen, robot)
        
        # Display info text
        info_text = [
            f"Target: ({int(target_pos[0])}, {int(target_pos[1])})",
            f"Robot Pos: ({int(robot.x)}, {int(robot.y)})",
            f"Angle Error: {error:.2f} rad",
            f"Distance: {distance_to_target:.1f} pixels",
            f"L Motor: {int(left_motor_speed)} | R Motor: {int(right_motor_speed)}",
            "Click to set a new target"
        ]
        for i, line in enumerate(info_text):
            text_surf = font.render(line, True, TEXT_COLOR)
            screen.blit(text_surf, (10, 10 + i * 25))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
