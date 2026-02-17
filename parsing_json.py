# Script para procesar y limpiar archivos JSON de etiquetas
# Elimina campos innecesarios y normaliza los datos en un DataFrame

# Importación de bibliotecas
import pandas as pd  # Para manipulación de datos
import json  # Para leer/escribir JSON
import os  # Para operaciones de sistema de archivos

# Ruta a la carpeta con archivos JSON de etiquetas
folder_path = "C:\\Users\\Domagoj\\Desktop\\Session 2 json"

# Lista para almacenar datos de todos los JSON
json_data_list = []

def rectify_json():
    """
    Elimina campos innecesarios de todos los archivos JSON en la carpeta.
    Los campos eliminados son: project_name, labeler, method, decision_basis
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Leer JSON
        f = open(file_path, 'r')
        data = json.load(f)
        
        # Eliminar campos no necesarios para el procesamiento posterior
        del data['project_name']
        del data['labeler']
        del data['method']
        del data['decision_basis']
        f.close()
        
        # Guardar JSON modificado
        f = open(file_path, 'w')
        json.dump(data, f, indent=4)


# Cargar todos los archivos JSON en una lista
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path) as f:
            data = json.load(f)
            
            # Si no hay toques en frames, limpiar también el campo de milisegundos
            if data['button_presses'] == '':
                data['button_presses_ms'] = ''
                print(data['button_presses_ms'])
            
            json_data_list.append(data)

# Ejecutar función de limpieza
rectify_json()

# Normalizar lista de JSON a DataFrame de pandas
cijela_lista  = pd.json_normalize(json_data_list)

# Configurar pandas para mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas


