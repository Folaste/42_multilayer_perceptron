import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score


def initialize(dimensions):

    parameters = {}
    l = len(dimensions)

    np.random.seed(1)

    for l in range(1, l):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    return parameters


def forward_propagation(x, parameters):
    activations = {'A0': x}
    nb_dimensions = len(parameters) // 2

    for l in range(1, nb_dimensions+1):
        z = parameters['W' + str(l)].dot(activations['A' + str(l-1)]) + parameters['b' + str(l)]
        if l < nb_dimensions:
            activations['A' + str(l)] = 1 / (1 + np.exp(-z))  # Sigmoid
        else:
            # --- Softmax stable ---
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            activations['A' + str(l)] = exp_z / np.sum(exp_z, axis=0, keepdims=True)

    return activations


def back_propagation(y, parameters, activations):
    nb_samples = y.shape[1]
    l = len(parameters) // 2

    d_z = activations['A' + str(l)] - y
    gradients = {}

    for l in reversed(range(1, l+1)):
        gradients['dW' + str(l)] = 1 / nb_samples * d_z.dot(activations['A' + str(l-1)].T)
        gradients['db' + str(l)] = 1 / nb_samples * np.sum(d_z, axis=1, keepdims=True)
        if l > 1:
            d_z = np.dot(parameters['W' + str(l)].T, d_z) * (activations['A' + str(l-1)] * (1 - activations['A' + str(l-1)]))

    return gradients

def update(gradients, parameters, learning_rate):
    l = len(parameters) // 2

    for l in range(1, l+1):
        parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]

    return parameters

def predict(x, parameters):
    activations = forward_propagation(x, parameters)
    return activations['A' + str(len(parameters) // 2)]

def build_model_filename(hidden_layers, learning_rate, batch_size, epochs):
    hl = "-".join(str(h) for h in hidden_layers)
    return f"model_h[{hl}]_lr{learning_rate}_bs{batch_size}_ep{epochs}.npz"

def save_model(parameters, layers, learning_rate, batch_size, epochs):
    # 1) Création du dossier models/
    os.makedirs("../models", exist_ok=True)

    # 2) Construire le nom du fichier
    filename = build_model_filename(layers, learning_rate, batch_size, epochs)
    filepath = os.path.join("../models", filename)

    # 3) Filtrer W* et b*
    filtered = {k: v for k, v in parameters.items() if k.startswith(("W", "b"))}

    # 4) Sauvegarde
    np.savez(filepath, **filtered)

    print(f"Model saved to: {filepath}")
    return filepath

def deep_neural_network(x_train_path, y_train_path, x_valid_path, y_valid_path, hidden_layers, learning_rate, epochs, batch_size):

    if not isinstance(hidden_layers, (list, tuple)) or not all(isinstance(x, int) for x in hidden_layers):
        raise ValueError("hidden_layers must be a list of integers")

    try:
        x_train = np.genfromtxt(x_train_path, delimiter=",")
        y_train = np.genfromtxt(y_train_path, delimiter=",")
        x_valid = np.genfromtxt(x_valid_path, delimiter=",")
        y_valid = np.genfromtxt(y_valid_path, delimiter=",")
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")

    # Format attendu : (n_features, n_samples)
    x_train = x_train.T
    x_valid = x_valid.T

    # y_train et y_valid doivent être (n_classes, n_samples)
    y_train = y_train.T
    y_valid = y_valid.T

    print('dimensions de X_train:', x_train.shape)
    print('dimensions de y_train:', y_train.shape)

    dimensions = list(hidden_layers)
    dimensions.insert(0, x_train.shape[0])  # input layer
    dimensions.append(2)  # output layer (2 classes)
    print(f"Network dimensions: {dimensions}")

    parameters = initialize(dimensions)

    training_history = np.zeros((epochs, 3))

    pbar = tqdm(range(epochs), desc="Training")

    for i in pbar:

        if batch_size == 0:
            activations = forward_propagation(x_train, parameters)
            gradients = back_propagation(y_train, parameters, activations)
            parameters = update(gradients, parameters, learning_rate)

        else:
            # Shuffle
            permutation = np.random.permutation(x_train.shape[1])
            x_shuffled = x_train[:, permutation]
            y_shuffled = y_train[:, permutation]

            nb_batches = x_train.shape[1] // batch_size

            for b in range(nb_batches):
                start = b * batch_size
                end = (b + 1) * batch_size

                x_batch = x_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]

                activations = forward_propagation(x_batch, parameters)
                gradients = back_propagation(y_batch, parameters, activations)
                parameters = update(gradients, parameters, learning_rate)

        # === Full train eval ===
        af = predict(x_train, parameters)
        af = np.clip(af, 1e-15, 1 - 1e-15)

        y_train_t = y_train.T
        af_t = af.T

        train_loss = float(log_loss(y_train_t, af_t))
        train_acc = float(accuracy_score(y_train_t.argmax(axis=1), af_t.argmax(axis=1)))

        # Validation
        y_pred_valid = predict(x_valid, parameters)
        valid_acc = float(accuracy_score(y_valid.T.argmax(axis=1), y_pred_valid.T.argmax(axis=1)))

        # Save history
        training_history[i, 0] = train_loss
        training_history[i, 1] = train_acc
        training_history[i, 2] = valid_acc

        # Update tqdm display
        pbar.set_postfix({
            "loss": f"{train_loss:.3f}",
            "train_acc": f"{train_acc:.3f}",
            "valid_acc": f"{valid_acc:.3f}",
        })

    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs} : "
              f"train loss : {round(float(training_history[i, 0]), 3)} - "
              f"train acc : {round(float(training_history[i, 1]), 3)} - "
              f"valid acc : {round(float(training_history[i, 2]), 3)}"
        )

    save_model(parameters, dimensions, learning_rate, batch_size, epochs)


    # === Graphiques ===
    plt.figure(figsize=(12, 4))
    plt.suptitle("Model evolution", fontsize=20)
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(training_history[:, 0], label='train loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(training_history[:, 1], label='train acc')
    plt.plot(training_history[:, 2], label='valid acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return training_history