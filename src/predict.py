import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.neural_network import forward_propagation


def binary_cross_entropy(y_true, y_pred):
    """
    Calculates binary cross-entropy loss

    Args:
        y_true: true labels (0 or 1)
        y_pred: predicted probabilities for class 1

    Returns:
        average loss
    """
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def final_prediction(model, x_path, y_path, show_samples_predictions=False, show_confusion_matrix=False):
    """
    Prediction program

    Args:
        model: path to .npz file containing model parameters
        x_path: path to CSV file containing features
        y_path: path to CSV file containing labels
        show_samples_predictions: if True, displays sample predictions
        show_confusion_matrix: if True, displays confusion matrix
    """
    # --- Load parameters ---
    if (not model.endswith(".npz")) or (not os.path.isfile(model)):
        raise ValueError(
            f"Model file '{model}' does not exist or is not a valid .npz file."
        )
    npz_file = np.load(model)
    parameters = {key: npz_file[key] for key in npz_file.files}

    # --- Load CSV data as float ---
    if (not x_path.endswith(".csv")) or (not os.path.isfile(x_path)):
        raise ValueError("x_path must be a valid CSV file.")
    x_data = np.loadtxt(x_path, delimiter=',', dtype=float).T

    y_data = None
    if y_path is not None:
        y_data = np.loadtxt(y_path, delimiter=',', dtype=float)
        if y_data.ndim > 1:
            y_data = np.argmax(y_data, axis=1)

    if x_data is None or model is None or y_data is None:
        raise ValueError("Invalid input data.")

    # --- Forward pass ---
    activations = forward_propagation(x_data, parameters)

    # --- Last layer ---
    a_final = activations[f"A{len(parameters) // 2}"]

    # --- Predictions ---
    predictions = np.argmax(a_final, axis=0)

    if show_samples_predictions:
        print_sample_predictions(a_final, predictions, y_data)

    result = {
        "probabilities": a_final,
        "predictions": predictions
    }

    # Accuracy
    accuracy = np.mean(predictions == y_data)
    result["accuracy"] = accuracy

    # Binary Cross-Entropy Loss
    # Get probability for class 1 (M)
    y_pred_proba = a_final[1, :]
    loss = binary_cross_entropy(y_data, y_pred_proba)
    result["loss"] = loss

    print(f"\n{'=' * 50}")
    print(f"EVALUATION METRICS")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Binary Cross-Entropy Loss: {loss:.6f}")
    print(f"{'=' * 50}\n")

    # --- Confusion Matrix ---
    if show_confusion_matrix:
        plot_confusion_matrix(y_data, predictions)


def print_sample_predictions(a_final, predictions, y_true):
    """
    Displays for each sample:
    - prob M
    - prob B
    - true label
    - prediction
    """
    if y_true is None:
        return

    n = a_final.shape[1]

    print(f"\n{'=' * 50}")
    print(f"SAMPLE PREDICTIONS")
    print(f"{'=' * 50}")

    for i in range(n):
        prob_b = a_final[0, i]
        prob_m = a_final[1, i]
        pred = predictions[i]
        truth = y_true[i]

        print(f"Sample {i + 1}: "
              f"P(B)={prob_b:.4f}, P(M)={prob_m:.4f}, "
              f"Predicted={'M' if pred == 1 else 'B'}, "
              f"Actual={'M' if truth == 1 else 'B'}")

    print(f"{'=' * 50}\n")


def plot_confusion_matrix(y_true, y_pred):
    """
    Creates and displays a confusion matrix

    Args:
        y_true: true labels
        y_pred: predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['B', 'M'],
                yticklabels=['B', 'M'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Print detailed metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'=' * 50}")
    print(f"CONFUSION MATRIX DETAILS")
    print(f"{'=' * 50}")
    print(f"True Negatives (B predicted as B): {tn}")
    print(f"False Positives (B predicted as M): {fp}")
    print(f"False Negatives (M predicted as B): {fn}")
    print(f"True Positives (M predicted as M): {tp}")

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"{'=' * 50}\n")