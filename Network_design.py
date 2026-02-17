# Deep neural network design and training script
# Creates a hybrid network (Dense + Conv1D + LSTM) for ball touch classification

# TensorFlow/Keras libraries import
import tensorflow as tf
from keras.models import Sequential  # Sequential model
import keras as keras
from keras.layers import * # All neural network layers
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping  # For early stopping of training
from keras.callbacks import CSVLogger  # To save training log
from keras.models import model_from_json
from tensorflow.keras import optimizers, losses, metrics

# DATA SPLITTING: Stratified Train/Test split
# Maintains class proportion in both sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=9)
for i, (train_index, test_index) in enumerate(sss.split(np.zeros(balanced_data_outputs.shape[0]), balanced_data_outputs)):
    pass  # We only need the indices from the first (and only) split

# Training set
X_train_tf = balanced_data_features[train_index, :, :]  # Training features
y_train_tf = balanced_data_outputs[train_index, :]      # Training labels

# Test set
X_test_tf = balanced_data_features[test_index, :, :]   # Test features
y_test_tf = balanced_data_outputs[test_index, :]       # Test labels

type(X_test_tf)

# VISUALIZATION: Class distribution in train and test sets
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

names = ["Legs", "Chest", "Other_F", "Other_NF"]  # Names of the 4 classes

# Bar chart for training set
axes[0].bar(names, np.sum(y_train_tf, axis=0))
axes[0].set_title('Train Set')
axes[0].set_xlabel('Labels')
axes[0].set_ylabel('No_of_Instances')

# Bar chart for test set
axes[1].bar(names, np.sum(y_test_tf, axis=0))
axes[1].set_title('Test Set')
axes[1].set_xlabel('Labels')
axes[1].set_ylabel('No_Instances')

plt.tight_layout()
plt.show()

# NEURAL NETWORK DESIGN
# Hybrid architecture: Dense -> Conv1D -> LSTM -> Dense
model = Sequential()

# Layer 1: Dense with 128 neurons
model.add(Dense(128, activation='relu', input_shape=(X_train_tf.shape[1], X_train_tf.shape[2])))

# Layer 2: Dense with 64 neurons
model.add(Dense(64, activation='relu'))

# Layer 3: 1D Convolutional with 64 filters and kernel size 21
# Extracts local temporal features
model.add(Conv1D(64, kernel_size=21, activation="relu"))

# Layer 4: LSTM with 24 units
# Captures long-range temporal dependencies
model.add(LSTM(24, activation='relu'))

# Layer 5: Dense with 64 neurons
model.add(Dense(64, activation='relu'))

# Layer 6: Dense with 32 neurons
model.add(Dense(32, activation='relu'))

# Output layer: 4 neurons (one per class) with sigmoid activation
model.add(Dense(4, activation='sigmoid'))

# Compile model
optimizer = optimizers.Adam(learning_rate=0.001)  # Adam optimizer
model.compile(loss=keras.losses.CategoricalCrossentropy(), 
              optimizer=optimizer, 
              metrics=[keras.metrics.CategoricalAccuracy()])

# Callback to stop training if it doesn't improve after 10 epochs
es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

# Logger to save training history to CSV
now = datetime.datetime.now()
csv_logger = CSVLogger('/content/drive/MyDrive/Master Thesis/Log/log'+now.strftime("%m_%d_%Y_%H:%M:%S")+'.csv', append=True, separator=';')

# MODEL TRAINING
startTime = timer()
history = model.fit(X_train_tf, y_train_tf, 
                    epochs=1000,  # Maximum 1000 epochs (can finish earlier due to early stopping)
                    callbacks=[es_callback, csv_logger], 
                    validation_data=(X_test_tf, y_test_tf), 
                    batch_size=1000, 
                    verbose=1)
endTime = timer()

# Print training time
print("Model trained in {:f}s.".format(endTime - startTime))
print("This is {:f}min.".format((endTime - startTime)/60))

# Save trained model
model.save('/content/drive/MyDrive/Master Thesis/Best Model/original_deep_network_Chest_350.h5')
