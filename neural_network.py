import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score


def initialize(dimensions):

    parameters = {}
    L = len(dimensions)

    np.random.seed(1)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    return parameters


def forward_propagation(X, parameters):
    activations = {'A0': X}
    L = len(parameters) // 2

    for l in range(1, L+1):
        Z = parameters['W' + str(l)].dot(activations['A' + str(l-1)]) + parameters['b' + str(l)]
        if l < L:
            activations['A' + str(l)] = 1 / (1 + np.exp(-Z))  # Sigmoid
        else:
            # --- Softmax stable ---
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            activations['A' + str(l)] = expZ / np.sum(expZ, axis=0, keepdims=True)

    return activations


def back_propagation(y, parameters, activations):
    nb_samples = y.shape[1]
    L = len(parameters) // 2

    dZ = activations['A' + str(L)] - y
    gradients = {}

    for l in reversed(range(1, L+1)):
        gradients['dW' + str(l)] = 1 / nb_samples * dZ.dot(activations['A' + str(l-1)].T)
        gradients['db' + str(l)] = 1 / nb_samples * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dZ = np.dot(parameters['W' + str(l)].T, dZ) * (activations['A' + str(l-1)] * (1 - activations['A' + str(l-1)]))

    return gradients

def update(gradients, parameters, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]

    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    return activations['A' + str(len(parameters) // 2)]

def deep_neural_network(X_train_path, y_train_path, X_valid_path, y_valid_path, hidden_layers, learning_rate, epochs, batch_size):

    if not isinstance(hidden_layers, (list, tuple)) or not all(isinstance(x, int) for x in hidden_layers):
        raise ValueError("hidden_layers must be a list of integers")

    try:
        X_train = np.genfromtxt(X_train_path, delimiter=",")
        y_train = np.genfromtxt(y_train_path, delimiter=",")
        X_valid = np.genfromtxt(X_valid_path, delimiter=",")
        y_valid = np.genfromtxt(y_valid_path, delimiter=",")
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")

    # Format attendu : (n_features, n_samples)
    X_train = X_train.T
    X_valid = X_valid.T

    # y_train et y_valid doivent être (n_classes, n_samples)
    y_train = y_train.T
    y_valid = y_valid.T

    print('dimensions de X_train:', X_train.shape)
    print('dimensions de y_train:', y_train.shape)

    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])  # input layer
    dimensions.append(2)  # output layer (2 classes)
    print(f"Network dimensions: {dimensions}")

    parameters = initialize(dimensions)
    # L = len(parameters) // 2

    training_history = np.zeros((epochs, 3))

    for i in tqdm(range(epochs)):

        if batch_size == 0:

            activations = forward_propagation(X_train, parameters)
            gradients = back_propagation(y_train, parameters, activations)
            parameters = update(gradients, parameters, learning_rate)

        else:
            # Shuffling data
            permutation = np.random.permutation(X_train.shape[1])
            X_shuffled = X_train[:, permutation]
            y_shuffled = y_train[:, permutation]

            nb_batches = X_train.shape[1] // batch_size

            for b in range(nb_batches):
                start = b * batch_size
                end = (b + 1) * batch_size

                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]

                activations = forward_propagation(X_batch, parameters)
                gradients = back_propagation(y_batch, parameters, activations)
                parameters = update(gradients, parameters, learning_rate)

        # ========= IMPORTANT =========
        # Recalcul complet pour le train set
        Af = predict(X_train, parameters)
        Af = np.clip(Af, 1e-15, 1 - 1e-15)

        # Metrics
        y_train_T = y_train.T
        Af_T = Af.T

        training_history[i, 0] = log_loss(y_train_T, Af_T)
        training_history[i, 1] = accuracy_score(y_train_T.argmax(axis=1), Af_T.argmax(axis=1))

        # Validation
        y_pred_valid = predict(X_valid, parameters)
        training_history[i, 2] = accuracy_score(y_valid.T.argmax(axis=1), y_pred_valid.T.argmax(axis=1))

        print(
            f"Epoch {i + 1}/{epochs} : loss = {round(float(training_history[i, 0]), 3)}, "
            f"accuracy train = {round(float(training_history[i, 1]), 3)}, accuracy valid = {round(float(training_history[i, 2]), 3)}"
        )

    # === Graphiques ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.plot(training_history[:, 2], label='valid acc')
    plt.legend()
    plt.show()

    return training_history