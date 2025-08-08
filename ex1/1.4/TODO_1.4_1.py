import numpy as np
import matplotlib.pyplot as plt

# Point 1.1
def LIF(um_0, I, T):
    Rm = 10*1e6 # Mega Ohm
    Cm = 1*1e-9 # Nano Farad
    u_thresh = -50*1e-3 # milli Volt
    u_rest = -65*1e-3 # milli Volt

    delta_t = 1e-5 
    um_t = np.zeros((int(T//delta_t)))
    um_t[0] = um_0

    dum_dt = lambda um_t: (u_rest - um_t + Rm*I)/(Rm*Cm)
   
    # TODO Calculate the um_t from T = 0 until T in steps of delta-t
    
    refactory_period = int(10e-3 / delta_t)
    refactory_count = 0

    for t in range(0, len(um_t)-1):
        if refactory_count > 0:
            um_t[t+1] = u_rest
            refactory_count -= 1
        
        else:
            um_t[t+1] = um_t[t] + dum_dt(um_t[t]) * delta_t
            if um_t[t+1] >= u_thresh:
                um_t[t+1] = u_rest
                refactory_count = refactory_period
              

    return um_t


# Point 1.2
# TODO: Calculate the membrane potential using the LIF function from Point 1.1
membrane_potential = LIF(um_0=-65*1e-3, I=1.6e-9, T=0.1)

plt.figure(figsize=(7,5))
plt.plot(list(range(int(0.1//1e-5))), membrane_potential)
plt.show()


# Point 1.3
# TODO: Define a function to calculate the interspike intervals
def calculate_isi(um_t, threshold=0.02):
    # Detect spikes by checking for drastic changes (running average difference)
    spikes = []
    window = 5  # samples for running average
    for i in range(window, len(um_t)):
        avg_prev = np.mean(um_t[i-window:i])
        avg_curr = np.mean(um_t[i-window+1:i+1])
        if avg_curr - avg_prev < -threshold:  # drastic drop indicates spike (reset)
            spikes.append(i)
    # Calculate interspike intervals
    isis = [(spikes[i] - spikes[i-1]) * 1e-5 for i in range(1, len(spikes))]
    return isis

isis = calculate_isi(membrane_potential)
print("Interspike intervals:", isis)

# TODO: Define a function to calculate the spiking frequency of a whole experiment
#spiking_frequency =

# membrane_potential2 = LIF(TODO)
#plt.figure(figsize=(7,5))
#plt.plot(list(range(int(0.1//1e-5))), membrane_potential2)
#plt.show()


# Point 1.4
plt.figure(figsize=(7,5))
spikes = []
# TODO write the code to accumulate the spikes
#for current in ...: #Solution here

plt.plot(list(np.arange(0,5.5e-9, 0.5e-9)), spikes)
plt.xlabel('Constant current')
plt.ylabel('Spiking frequency')
plt.show()