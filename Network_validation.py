# Script de validación del modelo entrenado
# Evalúa el rendimiento del modelo en diferentes conjuntos de datos

# Importación de bibliotecas
from tensorflow.keras.models import load_model  # Para cargar modelo guardado
from sklearn import metrics  # Para métricas de evaluación

# Cargar modelo entrenado desde archivo
best_model = load_model("/content/drive/MyDrive/Master Thesis/Best Model/original_deep_network_Chest_350.h5")
classes = ['Legs', "Chest", 'OtherF','OtherNF']  # Nombres de las clases

# EVALUACIÓN EN CONJUNTO DE PRUEBA
# Realizar predicciones (salida softmax)
pred_raw = best_model.predict(X_test_tf)

# Convertir probabilidades a etiquetas one-hot (argmax)
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calcular accuracy
score_acc = metrics.accuracy_score(y_test_tf, pred)
print(f"acc: {score_acc:.3f}")

# Generar reporte de clasificación (precision, recall, f1-score por clase)
class_rep = metrics.classification_report(y_test_tf, pred, target_names=classes)

# Crear y visualizar matriz de confusión normalizada
cm = metrics.confusion_matrix(y_test_tf.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar por filas
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()


# EVALUACIÓN EN CONJUNTO COMPLETO (todas las ventanas)
pred_raw = best_model.predict(features_array)

# Convertir probabilidades a etiquetas one-hot
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calcular accuracy en conjunto completo
score_acc = metrics.accuracy_score(outputs_array, pred)
print(f"acc: {score_acc:.3f}")

# Generar reporte de clasificación
class_rep = metrics.classification_report(outputs_array, pred, target_names=classes)

# Matriz de confusión para conjunto completo
cm = metrics.confusion_matrix(outputs_array.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()

# EVALUACIÓN EN UN VIDEO ESPECÍFICO
windowsize=50
video_id=5  # ID del video a analizar
video_data = df[df['video_id'] == video_id]  # Filtrar datos del video
sliding_window = create_sliding_windows(video_data, windowsize, 1)  # Crear ventanas
sliding_window_np=np.array(sliding_window)

# Separar características y etiquetas para este video
features_session=sliding_window_np[:,:,2:8]  # Características
outputs_session =sliding_window_np[:,-1,8:]  # Etiquetas

# Realizar predicciones para este video
pred_raw = best_model.predict(features_session)

# Convertir a etiquetas one-hot
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calcular accuracy para este video
score_acc = metrics.accuracy_score(outputs_session, pred)
print(f"acc: {score_acc:.3f}")

# Reporte de clasificación para este video
class_rep = metrics.classification_report(outputs_session, pred, target_names=classes)

# Matriz de confusión para este video
cm = metrics.confusion_matrix(outputs_session.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()

# VISUALIZACIÓN TEMPORAL: Comparar predicciones vs etiquetas reales frame por frame
fig, (ax0, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

frameindex=np.arange(50,len(pred)+50)  # Índices de frames

# Gráfico para clase "Legs"
ax0.plot(frameindex, outputs_session[:,0], "b-", label="Real")  # Etiqueta real (azul)
ax0.plot(frameindex, pred[:,0], "r-", label="Predicción")  # Predicción (rojo)
ax0.set_xlabel('Frameindex')
ax0.set_ylabel('Legs')

# Gráfico para clase "OtherF"
ax2.plot(frameindex, outputs_session[:,1], label="Real")
ax2.plot(frameindex, pred[:,1], label="Predicción")
ax2.set_xlabel('Frameindex')
ax2.set_ylabel('OtherF')

# Gráfico para clase "OtherNF"
ax3.plot(frameindex, outputs_session[:,2], label="Real")
ax3.plot(frameindex, pred[:,2], label="Predicción")
ax3.set_xlabel('Frameindex')
ax3.set_ylabel('OtherNF')