import numpy as np
import matplotlib.pyplot as plt

# Initialization
simlen = 30
target = 0.0
Ks = [0.2, 0.5, 1, 1.5, 2]
delays = [0, 1, 2, 3]

# Create a plot
plt.figure(figsize=(10, 6))

# Loop over different delay values
for K in Ks:
    for delta in delays:
        # Output array for each simulation
        y = np.zeros(simlen)
        y[0] = 1.0

        # Simulation loop
        for t in range(simlen - 1):
            # Determine the time-delayed output for the controller
            if t < delta:
                delayed_y = y[0]
            else:
                delayed_y = y[t - delta]

            # Compute control input with delay
            u = K * (target - delayed_y)

            # System dynamics
            y[t + 1] = 0.5 * y[t] + 0.4 * u

        # Plot the output for the current delay
        plt.plot(range(simlen), y, label=f'delay = {delta}, K={K}')
    plt.xlabel('time step')
    plt.ylabel('y')
    plt.title('System Output')
    plt.legend()
    plt.grid(True)
    plt.show()
