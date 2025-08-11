import numpy as np
import matplotlib.pyplot as plt
import time
from SimFunctions import SimulationFunctions

# Movement duration
T = 0.6
# Time step
dt = 0.01
# Simulation duration
L = 6.0

# Proportional parameter
kp = 2
# Derivative parameter
kd = 10
# Noise coefficient
noise_coeff = 0.0
# Delay in milliseconds
delay_ms = 40

# Upper arm length
le1 = 0.3
# Lower arm length
le2 = 0.3
# Upper arm mass
m1 = 3
# Lower arm mass
m2 = 3
# Gravity
g = -9.8

## Functions
Var = [T, dt, L, kp, kd, le1, le2, m1, m2, g]
Sim = SimulationFunctions(Var)

## Variables
# Joint angles [shoulder elbow]  [rad]
ang = [-np.pi / 4, np.pi]
ang_rec = np.zeros((int(L / dt + 1), 2))
ang_rec[0,:] = ang

# Joint velocity [shoulder elbow] [rad/s]
vel = [0, 0]
vel_rec = np.zeros((int(L / dt + 1), 2))

# Joint acceleration [shoulder elbow]  [rad/s^2]
acc = [0, 0]
acc_rec = np.zeros((int(L / dt + 1), 2))

# Jerk [shoulder elbow]
jerk_rec = np.zeros((int(L / dt + 1), 2))

# Shoulder position
shoulder_pos = [0, 0]
# Elbow position
elbow_pos_rec = np.zeros((int(L / dt) + 1, 2))
# Wrist position
wrist_pos = [0, 0]
wrist_pos_rec = np.zeros((int(L / dt + 1), 2))
elbow_pos, wrist_pos = Sim.fkinematics(ang)
wrist_pos_rec[0, :] = wrist_pos
elbow_pos_rec[0, :] = elbow_pos

# Initial wrist position for current movement
init_wrist_pos = wrist_pos
# Desired wrist position
final_wrist_pos = [[0.3, 0.0], [0.0, 0.0], [.3 * np.cos(np.pi / 4), .3 * np.sin(np.pi / 4)], [0.0, 0.0],
                   [0.0, .3], [0.0, 0.0], [.3 * np.cos(3 * np.pi / 4), .3 * np.sin(3 * np.pi / 4)], [0.0, 0.0]]

# Current target index
curr_target = 0
# Movement start_time
start_t = 0

# Calculate delay in time steps
delay_steps = int(delay_ms / (dt * 1000))

## Simulation
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title(f'Arm Reaching (kp={kp}, kd={kd}, delay={delay_ms}ms, noise_coeff={noise_coeff})')
ax.set_xlabel('meters')
ax.set_ylabel('meters')
ax.set_xlim([-0.5, .5])
ax.set_ylim([-0.5, .5])
ax.set_aspect('equal', adjustable='box')

Time = time.time()
for t in np.arange(0, int(L), dt):
    
    current_idx = round(t/dt)

    # Update records for the beginning of the step
    ang_rec[current_idx + 1, :] = ang
    vel_rec[current_idx + 1, :] = vel
    acc_rec[current_idx + 1, :] = acc
    if t > 0:
        jerk_rec[current_idx + 1, :] = acc - acc_rec[current_idx, :]

    ## Current wrist target
    current_wrist_target = final_wrist_pos[curr_target][:]

    if curr_target <= 7:
        ## Planner
        # Get desired position from planner
        if t - start_t < T:
            desired_pos = Sim.minjerk(init_wrist_pos, current_wrist_target, t - start_t)
        
        ## Inverse kinematics
        # Get desired angle from inverse kinematics
        desired_ang = np.real(Sim.invkinematics(desired_pos)).ravel()

        ## Inverse dynamics (Feedback Control)
        
        # Get delayed angles and velocities for feedback
        delayed_idx = (current_idx + 1) - delay_steps
        if delayed_idx >= 1:
            feedback_ang = ang_rec[delayed_idx, :]
            feedback_vel = vel_rec[delayed_idx, :]
        else:
            # Before enough history is recorded, use current (undelayed) feedback
            feedback_ang = ang
            feedback_vel = vel
            
        # Get desired torque from PD controller using (potentially delayed) feedback
        desired_torque = Sim.pdcontroller(desired_ang, feedback_ang, feedback_vel)
            
        ## Forward dynamics
        
        # Define noise using a Gaussian distribution, scaled by the coefficient
        noise = noise_coeff * np.random.randn(2)
        
        # Add noise to the desired torque
        noisy_torque = desired_torque + noise
        
        # Pass the final torque (with noise) to the plant
        ang, vel, acc = Sim.plant(ang, vel, acc, noisy_torque)

        ## Forward kinematics
        # Calculate new joint positions
        elbow_pos, wrist_pos = Sim.fkinematics(ang)
        elbow_pos_rec[current_idx + 1, :] = elbow_pos
        wrist_pos_rec[current_idx + 1, :] = wrist_pos

        ## Next target
        if (t - start_t >= T + 0.02) and (curr_target < 7):
            curr_target = curr_target + 1
            init_wrist_pos = wrist_pos
            start_t = t

elapsed = time.time() - Time
print(f"Simulation finished in: {elapsed:.2f} seconds")

# Plot final trajectory
ax.plot(wrist_pos_rec[:-1, 0], wrist_pos_rec[:-1, 1], '--', color='red', linewidth=1.0, label='Wrist Path')
ax.scatter(np.array(final_wrist_pos)[:, 0], np.array(final_wrist_pos)[:, 1], color='green', s=100, zorder=5, label='Targets')
ax.legend()
plt.show()

# Plot Kinematics
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
[A, B] = plt.plot(np.arange(0, L, dt), vel_rec[:int(L/dt), 0], np.arange(0, L, dt), vel_rec[:int(L/dt), 1])
plt.legend([A, B], ['Shoulder', 'Elbow'])
plt.title('Kinematics')
plt.ylabel('Velocity (rad/s)')

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, L, dt), acc_rec[:int(L/dt), 0], np.arange(0, L, dt), acc_rec[:int(L/dt), 1])
plt.ylabel('Acceleration (rad/s^2)')

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, L, dt), jerk_rec[:int(L/dt), 0], np.arange(0, L, dt), jerk_rec[:int(L/dt), 1])
plt.xlabel('Time (s)')
plt.ylabel('Jerk (rad/s^3)')

plt.tight_layout()
plt.show()
