# Script de diseño y entrenamiento de red neuronal profunda
# Crea una red híbrida (Dense + Conv1D + LSTM) para clasificación de toques de balón

# Importación de bibliotecas de TensorFlow/Keras
import tensorflow as tf
from keras.models import Sequential  # Modelo secuencial
import keras as keras
from keras.layers import *  # Todas las capas de red neuronal
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping  # Para detener entrenamiento temprano
from keras.callbacks import CSVLogger  # Para guardar log de entrenamiento
from keras.models import model_from_json
from tensorflow.keras import optimizers, losses, metrics

# DIVISIÓN DE DATOS: Train/Test split estratificado
# Mantiene la proporción de clases en ambos conjuntos
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=9)
for i,(train_index, test_index) in enumerate(sss.split(np.zeros(balanced_data_outputs.shape[0]), balanced_data_outputs)):
    pass  # Solo necesitamos los índices de la primera (y única) división

# Conjunto de entrenamiento
X_train_tf = balanced_data_featrues[train_index, :, :]  # Características de entrenamiento
y_train_tf = balanced_data_outputs[train_index, :]     # Etiquetas de entrenamiento

# Conjunto de prueba
X_test_tf = balanced_data_featrues[test_index, :, :]   # Características de prueba
y_test_tf = balanced_data_outputs[test_index, :]       # Etiquetas de prueba

type(X_test_tf)

# VISUALIZACIÓN: Distribución de clases en train y test sets
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

names=["Legs", "Chest", "Other_F","Other_NF"]  # Nombres de las 4 clases

# Gráfico de barras para conjunto de entrenamiento
axes[0].bar(names, np.sum(y_train_tf,axis=0))
axes[0].set_title('Train Set')
axes[0].set_xlabel('Labels')
axes[0].set_ylabel('No_of_Instances')

# Gráfico de barras para conjunto de prueba
axes[1].bar(names, np.sum(y_test_tf,axis=0))
axes[1].set_title('Test Set')
axes[1].set_xlabel('Labels')
axes[1].set_ylabel('No_Instances')

plt.tight_layout()
plt.show()


# DISEÑO DE LA RED NEURONAL
# Arquitectura híbrida: Dense -> Conv1D -> LSTM -> Dense
model = Sequential()

# Capa 1: Densa con 128 neuronas
model.add(Dense(128, activation='relu', input_shape=(X_train_tf.shape[1], X_train_tf.shape[2])))

# Capa 2: Densa con 64 neuronas
model.add(Dense(64, activation='relu'))

# Capa 3: Convolucional 1D con 64 filtros y kernel de tamaño 21
# Extrae características temporales locales
model.add(Conv1D(64, kernel_size=21, activation="relu"))

# Capa 4: LSTM con 24 unidades
# Captura dependencias temporales de largo alcance
model.add(LSTM(24, activation='relu'))

# Capa 5: Densa con 64 neuronas
model.add(Dense(64, activation='relu'))

# Capa 6: Densa con 32 neuronas
model.add(Dense(32, activation='relu'))

# Capa de salida: 4 neuronas (una por clase) con activación sigmoid
model.add(Dense(4, activation='sigmoid'))

# Compilar modelo
optimizer = optimizers.Adam(learning_rate=0.001)  # Optimizador Adam
model.compile(loss=keras.losses.CategoricalCrossentropy(), 
              optimizer=optimizer, 
              metrics=[keras.metrics.CategoricalAccuracy()])

# Callback para detener entrenamiento si no mejora después de 10 épocas
es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

# Logger para guardar historial de entrenamiento en CSV
now = datetime.datetime.now()
csv_logger = CSVLogger('/content/drive/MyDrive/Master Thesis/Log/log'+now.strftime("%m_%d_%Y_%H:%M:%S")+'.csv', append=True, separator=';')

# ENTRENAMIENTO DEL MODELO
startTime = timer()
history = model.fit(X_train_tf, y_train_tf, 
                   epochs=1000,  # Máximo 1000 épocas (puede terminar antes por early stopping)
                   callbacks=[es_callback,csv_logger], 
                   validation_data=(X_test_tf, y_test_tf), 
                   batch_size=1000, 
                   verbose=1)
endTime = timer()

# Imprimir tiempo de entrenamiento
print("Model trained in {:f}s.".format(endTime - startTime))
print("This is {:f}min.".format((endTime - startTime)/60))

# Guardar modelo entrenado
model.save('/content/drive/MyDrive/Master Thesis/Best Model/original_deep_network_Chest_350.h5')