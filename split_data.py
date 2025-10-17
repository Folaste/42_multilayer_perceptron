import os
import pandas as pd
import numpy as np
import random

def split_data(input_path, ratio=0.2, random_seed=None, normalization_method='z-score'):

    training_path = 'data/training_data.csv'
    validation_path = 'data/validation_data.csv'

    # Load the data
    df = pd.read_csv(input_path, header=None)

    result_col = 1  # colonne du diagnostic (M/B)

    # Set the random seed
    random.seed(random_seed)

    # Replace diagnosis column by binary classes
    df[result_col] = df[result_col].map({'M': 1, 'B': 0})

    # Group data by class (to have balanced classes)
    groups = {}
    for label, group in df.groupby(result_col):
        groups[label] = group.sample(frac=1, random_state=random_seed)  # Internal shuffle

    training_parts = []
    validation_parts = []

    # Split the balanced classes data into training and validation sets
    for label, group in groups.items():
        split_point = int(len(group) * (1 - ratio))
        training_parts.append(group.iloc[:split_point])
        validation_parts.append(group.iloc[split_point:])

    # Concatenate groups in each set
    train_data = pd.concat(training_parts).sample(frac=1, random_state=random_seed)
    validation_data = pd.concat(validation_parts).sample(frac=1, random_state=random_seed)

    # Create directories if they don't exist
    for path in [training_path, validation_path]:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    # Save the complete data
    train_data.to_csv(training_path, index=False, header=False)
    validation_data.to_csv(validation_path, index=False, header=False)

    # Separate data from result
    result_train_set = train_data[[result_col]]
    result_validation_set = validation_data[[result_col]]
    data_train_set = train_data.drop([0, 1], axis=1)
    data_validation_set = validation_data.drop([0, 1], axis=1)

    # Normalize data
    data_train_set, data_validation_set = normalize_data(data_train_set, data_validation_set, normalization_method)

    training_dir = os.path.dirname(training_path)
    training_file = os.path.basename(training_path)
    validation_dir = os.path.dirname(validation_path)
    validation_file = os.path.basename(validation_path)

    # Save normalized data
    data_train_set.to_csv(os.path.join(training_dir, 'X_' + training_file), index=False, header=False)
    data_validation_set.to_csv(os.path.join(validation_dir, 'X_' + validation_file), index=False, header=False)

    # Save raw labels (binary)
    result_train_set.to_csv(os.path.join(training_dir, 'y_' + training_file), index=False, header=False)
    result_validation_set.to_csv(os.path.join(validation_dir, 'y_' + validation_file), index=False, header=False)

    # === 🧠 Convert to one-hot encoding for MLP with 2 outputs ===
    y_train = result_train_set.values.flatten().astype(int)
    y_val = result_validation_set.values.flatten().astype(int)

    # One-hot encode: M → [1,0], B → [0,1]
    Y_train = np.zeros((len(y_train), 2))
    Y_val = np.zeros((len(y_val), 2))
    Y_train[np.arange(len(y_train)), 1 - y_train] = 1  # 1->M, 0->B
    Y_val[np.arange(len(y_val)), 1 - y_val] = 1

    # Save one-hot encoded labels
    pd.DataFrame(Y_train).to_csv(os.path.join(training_dir, 'y_onehot_' + training_file), index=False, header=False)
    pd.DataFrame(Y_val).to_csv(os.path.join(validation_dir, 'y_onehot_' + validation_file), index=False, header=False)


def normalize_data(train_set, validation_set, normalization_method='z-score'):
    if normalization_method == 'z-score':
        mean = train_set.mean()
        std = train_set.std()
        train_set = (train_set - mean) / std
        validation_set = (validation_set - mean) / std

    elif normalization_method == 'min-max':
        min_val = train_set.min()
        max_val = train_set.max()
        train_set = (train_set - min_val) / (max_val - min_val)
        validation_set = (validation_set - min_val) / (max_val - min_val)

    return train_set, validation_set
