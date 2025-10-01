import os
import pandas as pd
import random

def split_data(input_path, training_path, validation_path, ratio=0.2, random_seed=None):
    # Load the data
    df = pd.read_csv(input_path, header=None)

    label_col = 1

    # Set the random seed
    if random_seed:
        random.seed(random_seed)

    # Group data by class (to have balanced classes)
    groups = {}
    for label, group in df.groupby(label_col):
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

    # Save the data
    train_data.to_csv(training_path, index=False)
    validation_data.to_csv(validation_path, index=False)

    print(f"Training dataset saved to {training_path}")
    print(train_data[label_col].value_counts(normalize=True))
    print(f"Validation dataset saved to {validation_path}")
    print(validation_data[label_col].value_counts(normalize=True))

if __name__ == "__main__":
    split_data("data.csv", "data/train_data.csv", "data/validation_data.csv", random_seed=42)
