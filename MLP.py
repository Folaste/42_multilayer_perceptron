import numpy as np


class MLP:
    # output_activation_method, loss_function,
    def __init__(self, hidden_layer_sizes, epochs, learning_rate, batch_size, activation_method, weights_initializers, random_seed):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = 2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation_method = activation_method
        # self.output_activation_method = output_activation_method
        # self.loss_function = loss_function
        self.weights_initializers = weights_initializers
        self.random_seed = random_seed

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        # parameters = {}

        np.random.seed(self.random_seed)

        dimensions = self.hidden_layer_sizes + [self.output_layer_size]
        layers = len(dimensions)

        print(dimensions)
        print(layers)




