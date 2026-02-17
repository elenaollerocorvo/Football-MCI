# Trained model validation script
# Evaluates model performance on different datasets

# Import libraries
from tensorflow.keras.models import load_model  # To load saved model
from sklearn import metrics  # For evaluation metrics
import numpy as np
import matplotlib.pyplot as plt

# Load trained model from file
best_model = load_model("/content/drive/MyDrive/Master Thesis/Best Model/original_deep_network_Chest_350.h5")
classes = ['Legs', "Chest", 'OtherF','OtherNF']  # Class names

# TEST SET EVALUATION
# Make predictions (softmax output)
pred_raw = best_model.predict(X_test_tf)

# Convert probabilities to one-hot labels (argmax)
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calculate accuracy
score_acc = metrics.accuracy_score(y_test_tf, pred)
print(f"acc: {score_acc:.3f}")

# Generate classification report (precision, recall, f1-score per class)
class_rep = metrics.classification_report(y_test_tf, pred, target_names=classes)

# Create and visualize normalized confusion matrix
cm = metrics.confusion_matrix(y_test_tf.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by rows
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()


# FULL DATASET EVALUATION (all windows)
pred_raw = best_model.predict(features_array)

# Convert probabilities to one-hot labels
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calculate accuracy on full dataset
score_acc = metrics.accuracy_score(outputs_array, pred)
print(f"acc: {score_acc:.3f}")

# Generate classification report
class_rep = metrics.classification_report(outputs_array, pred, target_names=classes)

# Confusion matrix for full dataset
cm = metrics.confusion_matrix(outputs_array.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()

# SPECIFIC VIDEO EVALUATION
windowsize = 50
video_id = 5  # ID of the video to analyze
video_data = df[df['video_id'] == video_id]  # Filter video data
sliding_window = create_sliding_windows(video_data, windowsize, 1)  # Create windows
sliding_window_np = np.array(sliding_window)

# Separate features and labels for this video
features_session = sliding_window_np[:,:,2:8]  # Features
outputs_session = sliding_window_np[:,-1,8:]  # Labels

# Make predictions for this video
pred_raw = best_model.predict(features_session)

# Convert to one-hot labels
pred = np.zeros_like(pred_raw)
pred[np.arange(len(pred_raw)), pred_raw.argmax(1)] = 1

# Calculate accuracy for this video
score_acc = metrics.accuracy_score(outputs_session, pred)
print(f"acc: {score_acc:.3f}")

# Classification report for this video
class_rep = metrics.classification_report(outputs_session, pred, target_names=classes)

# Confusion matrix for this video
cm = metrics.confusion_matrix(outputs_session.argmax(1), pred.argmax(1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = metrics.ConfusionMatrixDisplay(cm_normalized, display_labels=classes)
disp.plot()

# TEMPORAL VISUALIZATION: Compare predictions vs real labels frame by frame
fig, (ax0, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

frameindex = np.arange(50, len(pred) + 50)  # Frame indices

# Plot for "Legs" class
ax0.plot(frameindex, outputs_session[:,0], "b-", label="Real")  # Real label (blue)
ax0.plot(frameindex, pred[:,0], "r-", label="Prediction")  # Prediction (red)
ax0.set_xlabel('Frameindex')
ax0.set_ylabel('Legs')

# Plot for "OtherF" class
ax2.plot(frameindex, outputs_session[:,1], label="Real")
ax2.plot(frameindex, pred[:,1], label="Prediction")
ax2.set_xlabel('Frameindex')
ax2.set_ylabel('OtherF')

# Plot for "OtherNF" class
ax3.plot(frameindex, outputs_session[:,2], label="Real")
ax3.plot(frameindex, pred[:,2], label="Prediction")
ax3.set_xlabel('Frameindex')
ax3.set_ylabel('OtherNF')
