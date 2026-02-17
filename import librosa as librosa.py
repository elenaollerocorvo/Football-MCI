# Test script for audio analysis using Librosa
# Loads an audio file and visualizes its waveform

# Import libraries
import librosa as librosa  # For audio processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa.display  # For audio-specific visualizations
from IPython.display import Audio  # To play audio in notebooks

# Path to the WAV audio file
path_to_sound = f'C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session1\\jetsonCatch02_002604.wav'

# Load audio file
# y: numpy array with audio samples (time series)
# sr: sampling rate in Hz
y, sr = librosa.load(path_to_sound, sr=32000)

# Create time array corresponding to each sample
time = np.arange(len(y)) / sr  # Time in seconds

# Print information about the loaded audio
print("The sampled audio is returned as a numpy array (time series) and has ", y.shape, " number of samples")
print("The 10 randomly picked consequitive samples of the audio are: ", y[3000:3010])

# Visualize audio waveform
plt.figure(figsize=(10, 4))
plt.plot(time, y)  # Amplitude vs time plot
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


