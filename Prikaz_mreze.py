# Script to visualize neural network architecture
# Creates a visual image of the network structure using visualkeras

# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential  # Sequential model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, Bidirectional
import visualkeras  # For neural network architecture visualization
from collections import defaultdict

# Custom class for Dense layer wrapper (currently unused)
class CustomDense(tf.keras.layers.Dense):
    """
    Custom wrapper for Dense layers.
    Allows customizing visualization properties.
    """
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(units, activation=activation, **kwargs)
    
    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.units)

# DEFINITION OF THE MODEL TO VISUALIZE
# Model with Bidirectional LSTM
model = Sequential()

# Layer 1: Dense with 128 neurons, input shape (50, 6)
# 50 = temporal window, 6 = features
model.add(Dense(128, activation='relu', input_shape=(50, 6)))

# Layer 2: Bidirectional LSTM with 64 units
# Processes the sequence in both directions (forward and backward)
model.add(Bidirectional(LSTM(64, activation='relu')))

# Layer 3: Output with 3 neurons and softmax activation
model.add(Dense(3, activation='softmax'))

# COLOR CONFIGURATION for visualization
color_map = defaultdict(dict)
# Assign turquoise color to Bidirectional layers
color_map[Bidirectional] = {'fill': 'turquoise'}

# Visualize the model and show it on screen
visualkeras.layered_view(model, legend=True, color_map=color_map).show()

# Save visualization as PNG image
visualkeras.layered_view(model, legend=True, color_map=color_map, to_file='C:\\Users\\Domagoj\\Desktop\\Official\\Diplomski\\Bidirectional_network.png')
