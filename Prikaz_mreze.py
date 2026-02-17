# Script para visualizar arquitectura de red neuronal
# Crea una imagen visual de la estructura de la red usando visualkeras

# Importación de bibliotecas
import tensorflow as tf
from tensorflow.keras.models import Sequential  # Modelo secuencial
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, Bidirectional
import visualkeras  # Para visualización de arquitectura de redes
from collections import defaultdict

# Clase personalizada para wrapper de capas Dense (no usada actualmente)
class CustomDense(tf.keras.layers.Dense):
    """
    Wrapper personalizado para capas Dense.
    Permite personalizar propiedades de visualización.
    """
    def _init_(self, units, activation=None, **kwargs):
        super()._init_(units, activation=activation, **kwargs)
    
    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.units)

# DEFINICIÓN DEL MODELO A VISUALIZAR
# Modelo con LSTM bidireccional
model = Sequential()

# Capa 1: Densa con 128 neuronas, entrada de shape (50, 6)
# 50 = ventana temporal, 6 = características
model.add(Dense(128, activation='relu', input_shape=(50, 6)))

# Capa 2: LSTM bidireccional con 64 unidades
# Procesa la secuencia en ambas direcciones (adelante y atrás)
model.add(Bidirectional(LSTM(64, activation='relu')))

# Capa 3: Salida con 3 neuronas y activación softmax
model.add(Dense(3, activation='softmax'))

# CONFIGURACIÓN DE COLORES para visualización
color_map = defaultdict(dict)
# Asignar color turquesa a capas Bidirectional
color_map[Bidirectional] = {'fill': 'turquoise'}

# Visualizar el modelo y mostrarlo en pantalla
visualkeras.layered_view(model, legend=True, color_map=color_map).show()

# Guardar visualización como imagen PNG
visualkeras.layered_view(model, legend=True, color_map=color_map, to_file='C:\\Users\\Domagoj\\Desktop\\Official\\Diplomski\\Bidirectional_network.png')