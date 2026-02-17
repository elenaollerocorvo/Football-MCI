# Script de prueba para análisis de audio usando Librosa
# Carga un archivo de audio y visualiza su forma de onda

# Importación de bibliotecas
import librosa as librosa  # Para procesamiento de audio
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa.display  # Para visualizaciones específicas de audio
from IPython.display import Audio  # Para reproducir audio en notebooks

# Ruta al archivo de audio WAV
path_to_sound = f'C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session1\\jetsonCatch02_002604.wav'

# Cargar archivo de audio
# y: array numpy con las muestras de audio (serie temporal)
# sr: tasa de muestreo (sampling rate) en Hz
y, sr = librosa.load(path_to_sound, sr=32000)

# Crear array de tiempo correspondiente a cada muestra
time = np.arange(len(y)) / sr  # Tiempo en segundos

# Imprimir información sobre el audio cargado
print("The sampled audio is returned as a numpy array (time series) and has ", y.shape, " number of samples")
print("The 10 randomly picked consequitive samples of the audio are: ", y[3000:3010])

# Visualizar forma de onda del audio
plt.figure(figsize=(10, 4))
plt.plot(time, y)  # Gráfico de amplitud vs tiempo
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

