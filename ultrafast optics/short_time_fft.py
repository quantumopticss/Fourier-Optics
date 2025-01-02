import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# Generate a sample signal with changing frequencies (e.g., a chirp signal)
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs)  # Time vector (10 seconds of data)
f_start = 10  # Start frequency of the chirp (Hz)
f_end = 200  # End frequency of the chirp (Hz)
signal = np.exp(-((t-5)/1.5)**2)*np.exp(-1j*0.5*t**2)  # Chirp signal

# Compute the Short-Time Fourier Transform (STFT)
f, t_stft, Zxx = stft(signal, fs, nperseg=100)  # nperseg is the window length

# Plot the magnitude of the STFT
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='auto')
plt.title("STFT of the Chirp Signal")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (seconds)")
plt.colorbar(label="Magnitude")
plt.ylim(0, 300)  # Limit frequency range for better visualization
plt.show()
