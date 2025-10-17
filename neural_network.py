import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score


def initialize(dimensions):

    parameters = {}
    L = len(dimensions)

    np.random.seed(1)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * 0.01
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

def deep_neural_network(X_path, y_path, hidden_layers, learning_rate, epochs):

    if not isinstance(hidden_layers, (list, tuple)) or not all(isinstance(x, int) for x in hidden_layers):
        raise ValueError("hidden_layers must be a list of integers")

    try:
        X = np.genfromtxt(X_path, delimiter=",")
        y = np.genfromtxt(y_path, delimiter=",")
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")

    X = X.T
    # y = y.reshape((1, y.shape[0]))
    y = y.T

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(2)
    print(f"Network dimensions: {dimensions}")

    parameters = initialize(dimensions)
    # for e in parameters:
    #     print(f"{e}: {parameters[e].shape}")

    L = len(parameters) // 2

    training_history = np.zeros((epochs, 2))

    for i in tqdm(range(epochs)):
        activations = forward_propagation(X, parameters)
        gradients = back_propagation(y, parameters, activations)
        parameters = update(gradients, parameters, learning_rate)

        Af = activations['A' + str(L)]
        Af = np.clip(Af, 1e-15, 1 - 1e-15)  # stabilité

        # Transposées pour sklearn
        Af_T = Af.T
        y_T = y.T

        training_history[i, 0] = log_loss(y_T, Af_T)
        training_history[i, 1] = accuracy_score(y_T.argmax(axis=1), Af_T.argmax(axis=1))


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return training_history