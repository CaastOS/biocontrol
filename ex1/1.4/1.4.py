import numpy as np
import matplotlib.pyplot as plt

# Point 1.1
def LIF(um_0, I, T):
    Rm = 10e6      # Ohm
    Cm = 1e-9      # Farad
    u_thresh = -50e-3  # Volt
    u_rest = -65e-3    # Volt

    delta_t = 1e-5
    time = np.arange(0, T, delta_t)
    um_t = np.zeros_like(time)
    um_t[0] = um_0

    tau_m = Rm * Cm
    spike_times = []

    for t in range(len(time) - 1):
        du_dt = (u_rest - um_t[t] + Rm * I) / tau_m
        um_t[t+1] = um_t[t] + du_dt * delta_t
        if um_t[t+1] >= u_thresh:
            um_t[t+1] = u_rest
            spike_times.append(time[t+1])

    return time, um_t, spike_times


# Point 1.2
time, membrane_potential, spikes_1nA = LIF(um_0=-65e-3, I=1e-9, T=0.1)

plt.figure(figsize=(7,5))
plt.plot(time, membrane_potential)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (V)")
plt.title("LIF neuron with I = 1 nA")
plt.show()

def produces_spike(I):
    _, _, spikes = LIF(-65e-3, I, 0.2)
    return len(spikes) > 0

I_test = np.arange(0, 5e-9, 0.05e-9)
min_current = None
for I in I_test:
    if produces_spike(I):
        min_current = I
        break
print(f"Minimum current for spike: {min_current*1e9:.2f} nA")

# Point 1.3

def calculate_isi(spike_times):
    if len(spike_times) < 2:
        return []
    return np.diff(spike_times)

def calculate_frequency(spike_times):
    isis = calculate_isi(spike_times)
    if len(isis) == 0:
        return 0
    return 1 / np.mean(isis)

isis = calculate_isi(spikes_1nA)
freq = calculate_frequency(spikes_1nA)
print("Interspike intervals (s):", isis)
print("Spiking frequency (Hz):", freq)

currents = np.arange(0, 5.5e-9, 0.5e-9)
spikes_freq = []

for I in currents:
    _, _, u_spikes = LIF(-65e-3, I, 0.1)
    spikes_freq.append(calculate_frequency(u_spikes))

plt.figure(figsize=(7,5))
plt.plot(currents*1e9, spikes_freq, marker='o')
plt.xlabel('Constant current (nA)')
plt.ylabel('Spiking frequency (Hz)')
plt.title('F-I curve of LIF neuron')
plt.grid(True)
plt.show()
