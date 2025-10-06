import os
import pandas as pd
import random

def split_data(input_path, ratio=0.2, random_seed=None, normalization_method='z-score'):

    training_path = 'data/training_data.csv'
    validation_path = 'data/validation_data.csv'

    # Load the data
    df = pd.read_csv(input_path, header=None)

    result_col = 1

    # Set the random seed
    random.seed(random_seed)

    # Replace diagnosis column by binary classes
    df[result_col] = df[result_col].map({'M' : 1, 'B': 0})

    # Group data by class (to have balanced classes)
    groups = {}
    for label, group in df.groupby(result_col):
        groups[label] = group.sample(frac=1, random_state=random_seed) # Internal shuffle

    training_parts = []
    validation_parts = []

    # Split the balanced classes data into training and validation sets
    for label, group in groups.items():
        split_point = int(len(group) * (1 - ratio))
        training_parts.append(group.iloc[:split_point])
        validation_parts.append(group.iloc[split_point:])

    # Concatenate groups in each set
    train_data = pd.concat(training_parts).sample(frac=1, random_state=random_seed) # Internal shuffle
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

    # Save the data with X_ and y_ prefixes, preserving directory structure
    data_train_set.to_csv(os.path.join(training_dir, 'X_' + training_file), index=False, header=False)
    data_validation_set.to_csv(os.path.join(validation_dir, 'X_' + validation_file), index=False, header=False)
    result_train_set.to_csv(os.path.join(training_dir, 'y_' + training_file), index=False, header=False)
    result_validation_set.to_csv(os.path.join(validation_dir, 'y_' + validation_file), index=False, header=False)



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