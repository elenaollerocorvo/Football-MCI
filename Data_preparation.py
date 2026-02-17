# Script de preparación de datos para entrenamiento de red neuronal
# Realiza interpolación, reducción de clases, normalización y creación de ventanas deslizantes

# Importación de bibliotecas
import matplotlib.pyplot as plt  # Para gráficos
import pandas as pd  # Para manipulación de DataFrames
import numpy as np  # Para operaciones numéricas
from sklearn.preprocessing import MinMaxScaler  # Para normalización
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit  # Para división de datos
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime
from timeit import default_timer as timer

# Ruta al archivo CSV combinado con todos los datos
path = "/content/drive/MyDrive/Master Thesis/combined_file_V5.csv"

# Cargar datos en DataFrame
location = pd.read_csv(path)
df = pd.DataFrame(location)

# Definir columnas de características (distancias y ratio de alturas)
feature_columns = df.columns[2:8]
# Definir columnas de clases (codificación one-hot original)
class_columns = df.columns[8:13]

# Interpolación de valores faltantes para cada clase
# Los valores 820 y 0 indican datos faltantes que se interpolan linealmente
for class_col in class_columns:
    # Identificar filas donde esta clase está activa (valor one-hot = 1)
    class_active_indices = df.index[df[class_col] == 1].tolist()
    i = 0

    # Agrupar índices consecutivos (frames consecutivos de la misma clase)
    while i < len(class_active_indices):
      group_indices = [class_active_indices[i]]
      i += 1
      # Continuar agrupando mientras los índices sean consecutivos
      while i<len(class_active_indices) and class_active_indices[i] == class_active_indices[i - 1] + 1:
        group_indices.append(class_active_indices[i])
        i +=1
      
      # Extraer características para este grupo de frames
      class_df = df.loc[group_indices, feature_columns]
      # Reemplazar valores de datos faltantes (820 y 0) con NaN
      class_df[feature_columns] = class_df[feature_columns].replace(820, np.nan)
      class_df[feature_columns] = class_df[feature_columns].replace(0, np.nan)

      # Interpolar valores faltantes usando interpolación lineal
      # limit_direction='both' permite interpolar en ambas direcciones
      interpolated_class_df = class_df.interpolate(method='linear',limit_direction = 'both',  axis=0)

      # Actualizar el DataFrame original con los valores interpolados
      df.loc[group_indices, feature_columns] = interpolated_class_df

# Convertir tipos de datos a los tipos apropiados
df = df.astype({"distance_to_RF": int, "distance_to_LF": int, "distance_to_RT": int, "distance_to_LT": int, "distance_to_CH": int, "person_ball_H_rt": float})


# Calcular distribución de clases para visualización
np_data=df.to_numpy()
distribution=np.sum(np_data[:,8:],axis=0)  # Suma de cada clase
names=["RF","LF","RT","LT","Chest","Other"]
plt.bar(names,distribution)  # Gráfico de barras con la distribución

# REDUCCIÓN DE CLASES: Combinar toques de extremidades inferiores en una sola clase "Legs"
# Verificar si hay algún toque en las columnas de extremidades inferiores
has_data = df.iloc[:, 8:12].apply(lambda row: (row.notnull() & (row != 0)).any(), axis=1)
df['Legs'] = has_data.astype("float32")
df.insert(8, 'Legs', df.pop('Legs'))  # Insertar columna Legs al inicio de las clases
df.drop(columns=['RightF','LeftF','RightT','LeftT'], inplace=True)  # Eliminar columnas individuales

# DIVISIÓN DE CLASE "OTHER": Separar en dos subclases
# Other_found: cuando hay datos de distancias disponibles (distance != 820)
# Other_not_found: cuando NO hay datos de distancias (distance == 820)
df['Other_found'] = df['Other'].where((df['distance_to_RF'] != 820) & (df['Other'] == 1), 0.0)
df['Other_not_found'] = df['Other'].where((df['distance_to_RF'] == 820) & (df['Other'] == 1), 0.0)
df.drop(columns=["Other"], inplace=True)  # Eliminar columna Other original

# NORMALIZACIÓN: Escalar todas las características al rango [0, 1]
scaler = MinMaxScaler()
df[df.columns[2:8]] = scaler.fit_transform(df[df.columns[2:8]])
df.describe()  # Mostrar estadísticas descriptivas

def create_sliding_windows(df, window_size, step_size):
    """
    Crea ventanas deslizantes sobre el DataFrame.
    
    Args:
        df: DataFrame de entrada
        window_size: Tamaño de cada ventana (número de frames)
        step_size: Paso entre ventanas consecutivas
        
    Returns:
        Lista de DataFrames, cada uno representando una ventana
    """
    arr = df.values
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(window_size, arr.shape[1]))[::step_size, 0]
    return [pd.DataFrame(window, columns=df.columns) for window in windows]


# CREACIÓN DE VENTANAS DESLIZANTES
# Se crean ventanas de tamaño fijo para capturar contexto temporal
all_windows = []
windowsize=50  # Tamaño de ventana: 50 frames
video_ids = df['video_id'].unique()  # Lista de IDs únicos de video

# Crear ventanas por video (no mezclar frames de diferentes videos)
for video_id in video_ids:
    video_data = df[df['video_id'] == video_id]  # Datos del video actual
    sliding_windows = create_sliding_windows(video_data, windowsize, 1)  # Step size = 1
    for window in sliding_windows:
        all_windows.append(window)

# Separar características (X) y etiquetas (y)
features = []  # Secuencias de características (ventanas)
outputs = []   # Etiquetas (clase del último frame de cada ventana)

for window in all_windows:
    features.append( window.iloc[:, 2:8].values)  # Características: columnas 2-7
    outputs.append(window.iloc[-1, 8:].values)    # Etiqueta: último frame, columnas 8+

# Convertir a arrays de numpy
features_array = np.array(features)  # Shape: (num_ventanas, windowsize, num_características)
outputs_array = np.array(outputs)    # Shape: (num_ventanas, num_clases)

# BALANCEO DE CLASES
# La clase "Chest" es la menos frecuente, usarla como referencia
column_names = ['Legs', 'Chest', 'Other_found', 'Other_not_found']
windowed_df = pd.DataFrame(outputs_array, columns=column_names)

# Obtener índices de cada clase
indices_of_Other_nf = windowed_df[windowed_df['Other_not_found'] == 1].index
indices_of_Other_f = windowed_df[windowed_df['Other_found'] == 1].index
indices_of_Legs = windowed_df[windowed_df['Legs'] == 1].index
indices_of_Chest = windowed_df[windowed_df['Chest'] == 1].index

# Fijar semilla para reproducibilidad
np.random.seed(42) 

# Muestrear aleatoriamente el mismo número de muestras de cada clase
# (igualar a la cantidad de muestras de "Chest", la clase minoritaria)
random_idx_Other_nf = np.random.choice(indices_of_Other_nf, len(indices_of_Chest), replace=False)
random_idx_Other_f = np.random.choice(indices_of_Other_f, len(indices_of_Chest), replace=False)
random_idx_Legs = np.random.choice(indices_of_Legs, len(indices_of_Chest), replace=False)
random_idx_Chest = indices_of_Chest.to_numpy()

# Combinar todas las clases balanceadas en un solo conjunto de datos
balanced_data_featrues=np.vstack([features_array[random_idx_Legs, :, :],features_array[random_idx_Chest, :, :],features_array[random_idx_Other_nf, :, :],features_array[random_idx_Other_f, :, :]])

balanced_data_outputs=np.vstack([windowed_df.to_numpy()[random_idx_Legs, :],windowed_df.to_numpy()[random_idx_Chest, :],windowed_df.to_numpy()[random_idx_Other_nf, :],windowed_df.to_numpy()[random_idx_Other_f, :]])


