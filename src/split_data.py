import os
import pandas as pd
import numpy as np


def split_data(
        input_path,
        dev_ratio=0.15,
        test_ratio=0.15,
        random_seed=None,
        normalization_method='z-score'
):
    """
    Sépare les données en ensembles d'entraînement, développement et test,
    puis normalise et sauvegarde les données.

    Args:
        input_path: Chemin vers le fichier de données d'entrée
        dev_ratio: Proportion de données pour l'ensemble de développement
        test_ratio: Proportion de données pour l'ensemble de test
        random_seed: Graine aléatoire pour la reproductibilité
        normalization_method: Méthode de normalisation ('z-score' ou 'min-max')
    """
    # Chemins de sortie
    training_path = 'data/training_data.csv'
    dev_path = 'data/dev_data.csv'
    test_path = 'data/test_data.csv'

    # Chargement des données
    df = pd.read_csv(input_path, header=None)

    # Colonne de diagnostic (M/B)
    RESULT_COL = 1

    # Préparation des données
    df[RESULT_COL] = df[RESULT_COL].map({'M': 1, 'B': 0})

    # Séparation stratifiée par classe
    train_data, dev_data, test_data = _stratified_split(
        df, RESULT_COL, dev_ratio, test_ratio, random_seed
    )

    # Création des répertoires et sauvegarde des données complètes
    _create_directories([training_path, dev_path, test_path])
    train_data.to_csv(training_path, index=False, header=False)
    dev_data.to_csv(dev_path, index=False, header=False)
    test_data.to_csv(test_path, index=False, header=False)

    # Séparation features/labels
    X_train, y_train = _separate_features_labels(train_data, RESULT_COL)
    X_dev, y_dev = _separate_features_labels(dev_data, RESULT_COL)
    X_test, y_test = _separate_features_labels(test_data, RESULT_COL)

    # Normalisation des features
    X_train, X_dev, X_test = normalize_data(
        X_train, X_dev, X_test, normalization_method
    )

    # Sauvegarde des données normalisées et labels
    _save_datasets(
        X_train, y_train, training_path,
        X_dev, y_dev, dev_path,
        X_test, y_test, test_path
    )


def _stratified_split(df, result_col, dev_ratio, test_ratio, random_seed):
    """Effectue une séparation stratifiée des données par classe."""
    groups = {
        label: group.sample(frac=1, random_state=random_seed)
        for label, group in df.groupby(result_col)
    }

    training_parts, dev_parts, test_parts = [], [], []

    for group in groups.values():
        n = len(group)
        train_end = int(n * (1 - dev_ratio - test_ratio))
        dev_end = int(n * (1 - test_ratio))

        training_parts.append(group.iloc[:train_end])
        dev_parts.append(group.iloc[train_end:dev_end])
        test_parts.append(group.iloc[dev_end:])

    # Mélange final de chaque ensemble
    train_data = pd.concat(training_parts).sample(frac=1, random_state=random_seed)
    dev_data = pd.concat(dev_parts).sample(frac=1, random_state=random_seed)
    test_data = pd.concat(test_parts).sample(frac=1, random_state=random_seed)

    return train_data, dev_data, test_data


def _separate_features_labels(data, result_col):
    """Sépare les features et les labels."""
    y = data[[result_col]]
    X = data.drop([0, result_col], axis=1)
    return X, y


def _create_directories(paths):
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    for path in paths:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)


def _save_datasets(X_train, y_train, train_path,
                   X_dev, y_dev, dev_path,
                   X_test, y_test, test_path):
    """Sauvegarde tous les datasets (features, labels binaires et one-hot)."""
    datasets = [
        (X_train, y_train, train_path),
        (X_dev, y_dev, dev_path),
        (X_test, y_test, test_path)
    ]

    for X, y, path in datasets:
        directory = os.path.dirname(path)
        filename = os.path.basename(path)

        # Sauvegarde des features normalisées
        X.to_csv(os.path.join(directory, f'X_{filename}'), index=False, header=False)

        # Sauvegarde des labels binaires
        y.to_csv(os.path.join(directory, f'y_{filename}'), index=False, header=False)

        # Conversion et sauvegarde des labels one-hot
        y_binary = y.values.flatten().astype(int)
        y_onehot = np.eye(2)[1 - y_binary]  # M (1) -> [1,0], B (0) -> [0,1]
        pd.DataFrame(y_onehot).to_csv(
            os.path.join(directory, f'y_onehot_{filename}'),
            index=False,
            header=False
        )


def normalize_data(train_set, dev_set, test_set, normalization_method='z-score'):
    """
    Normalise les données en utilisant les statistiques de l'ensemble d'entraînement.

    Args:
        train_set: Ensemble d'entraînement
        dev_set: Ensemble de développement
        test_set: Ensemble de test
        normalization_method: 'z-score' ou 'min-max'

    Returns:
        Tuple des trois ensembles normalisés
    """
    if normalization_method == 'z-score':
        mean, std = train_set.mean(), train_set.std()
        train_set = (train_set - mean) / std
        dev_set = (dev_set - mean) / std
        test_set = (test_set - mean) / std

    elif normalization_method == 'min-max':
        min_val, max_val = train_set.min(), train_set.max()
        train_set = (train_set - min_val) / (max_val - min_val)
        dev_set = (dev_set - min_val) / (max_val - min_val)
        test_set = (test_set - min_val) / (max_val - min_val)

    return train_set, dev_set, test_set