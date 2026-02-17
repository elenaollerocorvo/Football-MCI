# Data preparation script for neural network training
# Performs interpolation, class reduction, normalization, and sliding window creation

# Import libraries
import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For DataFrame manipulation
import numpy as np  # For numerical operations
from sklearn.preprocessing import MinMaxScaler  # For normalization
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit  # For data splitting
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime
from timeit import default_timer as timer

# Path to the combined CSV file with all data
path = "/content/drive/MyDrive/Master Thesis/combined_file_V5.csv"

# Load data into DataFrame
location = pd.read_csv(path)
df = pd.DataFrame(location)

# Define feature columns (distances and height ratio)
feature_columns = df.columns[2:8]
# Define class columns (original one-hot encoding)
class_columns = df.columns[8:13]

# Interpolation of missing values for each class
# Values 820 and 0 indicate missing data which are linearly interpolated
for class_col in class_columns:
    # Identify rows where this class is active (one-hot value = 1)
    class_active_indices = df.index[df[class_col] == 1].tolist()
    i = 0

    # Group consecutive indices (consecutive frames of the same class)
    while i < len(class_active_indices):
        group_indices = [class_active_indices[i]]
        i += 1
        # Continue grouping as long as the indices are consecutive
        while i < len(class_active_indices) and class_active_indices[i] == class_active_indices[i - 1] + 1:
            group_indices.append(class_active_indices[i])
            i += 1
        
        # Extract features for this group of frames
        class_df = df.loc[group_indices, feature_columns]
        # Replace missing data values (820 and 0) with NaN
        class_df[feature_columns] = class_df[feature_columns].replace(820, np.nan)
        class_df[feature_columns] = class_df[feature_columns].replace(0, np.nan)

        # Interpolate missing values using linear interpolation
        # limit_direction='both' allows interpolating in both directions
        interpolated_class_df = class_df.interpolate(method='linear', limit_direction='both', axis=0)

        # Update the original DataFrame with the interpolated values
        df.loc[group_indices, feature_columns] = interpolated_class_df

# Convert data types to appropriate types
df = df.astype({"distance_to_RF": int, "distance_to_LF": int, "distance_to_RT": int, "distance_to_LT": int, "distance_to_CH": int, "person_ball_H_rt": float})

# Calculate class distribution for visualization
np_data = df.to_numpy()
distribution = np.sum(np_data[:, 8:], axis=0)  # Sum of each class
names = ["RF", "LF", "RT", "LT", "Chest", "Other"]
plt.bar(names, distribution)  # Bar chart with the distribution

# CLASS REDUCTION: Combine lower limb touches into a single "Legs" class
# Check if there is any touch in the lower limb columns
has_data = df.iloc[:, 8:12].apply(lambda row: (row.notnull() & (row != 0)).any(), axis=1)
df['Legs'] = has_data.astype("float32")
df.insert(8, 'Legs', df.pop('Legs'))  # Insert Legs column at the beginning of the classes
df.drop(columns=['RightF', 'LeftF', 'RightT', 'LeftT'], inplace=True)  # Drop individual columns

# "OTHER" CLASS SPLITTING: Separate into two subclasses
# Other_found: when distance data is available (distance != 820)
# Other_not_found: when distance data is NOT available (distance == 820)
df['Other_found'] = df['Other'].where((df['distance_to_RF'] != 820) & (df['Other'] == 1), 0.0)
df['Other_not_found'] = df['Other'].where((df['distance_to_RF'] == 820) & (df['Other'] == 1), 0.0)
df.drop(columns=["Other"], inplace=True)  # Drop original Other column

# NORMALIZATION: Scale all features to the [0, 1] range
scaler = MinMaxScaler()
df[df.columns[2:8]] = scaler.fit_transform(df[df.columns[2:8]])
df.describe()  # Show descriptive statistics

def create_sliding_windows(df, window_size, step_size):
    """
    Creates sliding windows over the DataFrame.
    
    Args:
        df: Input DataFrame
        window_size: Size of each window (number of frames)
        step_size: Step between consecutive windows
        
    Returns:
        List of DataFrames, each representing a window
    """
    arr = df.values
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(window_size, arr.shape[1]))[::step_size, 0]
    return [pd.DataFrame(window, columns=df.columns) for window in windows]

# SLIDING WINDOW CREATION
# Fixed-size windows are created to capture temporal context
all_windows = []
windowsize = 50  # Window size: 50 frames
video_ids = df['video_id'].unique()  # List of unique video IDs

# Create windows per video (do not mix frames from different videos)
for video_id in video_ids:
    video_data = df[df['video_id'] == video_id]  # Current video data
    sliding_windows = create_sliding_windows(video_data, windowsize, 1)  # Step size = 1
    for window in sliding_windows:
        all_windows.append(window)

# Separate features (X) and labels (y)
features = []  # Feature sequences (windows)
outputs = []   # Labels (class of the last frame of each window)

for window in all_windows:
    features.append(window.iloc[:, 2:8].values)  # Features: columns 2-7
    outputs.append(window.iloc[-1, 8:].values)    # Label: last frame, columns 8+

# Convert to numpy arrays
features_array = np.array(features)  # Shape: (num_windows, windowsize, num_features)
outputs_array = np.array(outputs)    # Shape: (num_windows, num_classes)

# CLASS BALANCING
# The "Chest" class is the least frequent, use it as a reference
column_names = ['Legs', 'Chest', 'Other_found', 'Other_not_found']
windowed_df = pd.DataFrame(outputs_array, columns=column_names)

# Get indices of each class
indices_of_Other_nf = windowed_df[windowed_df['Other_not_found'] == 1].index
indices_of_Other_f = windowed_df[windowed_df['Other_found'] == 1].index
indices_of_Legs = windowed_df[windowed_df['Legs'] == 1].index
indices_of_Chest = windowed_df[windowed_df['Chest'] == 1].index

# Set seed for reproducibility
np.random.seed(42) 

# Randomly sample the same number of samples for each class
# (equal to the number of samples of "Chest", the minority class)
random_idx_Other_nf = np.random.choice(indices_of_Other_nf, len(indices_of_Chest), replace=False)
random_idx_Other_f = np.random.choice(indices_of_Other_f, len(indices_of_Chest), replace=False)
random_idx_Legs = np.random.choice(indices_of_Legs, len(indices_of_Chest), replace=False)
random_idx_Chest = indices_of_Chest.to_numpy()

# Combine all balanced classes into a single dataset
balanced_data_features = np.vstack([
    features_array[random_idx_Legs, :, :],
    features_array[random_idx_Chest, :, :],
    features_array[random_idx_Other_nf, :, :],
    features_array[random_idx_Other_f, :, :]
])

balanced_data_outputs = np.vstack([
    windowed_df.to_numpy()[random_idx_Legs, :],
    windowed_df.to_numpy()[random_idx_Chest, :],
    windowed_df.to_numpy()[random_idx_Other_nf, :],
    windowed_df.to_numpy()[random_idx_Other_f, :]
])
