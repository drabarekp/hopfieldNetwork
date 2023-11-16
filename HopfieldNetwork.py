import numpy as np
# for visualization
import matplotlib.pyplot as plt
from constants import HEBB, OJA, SYNC, ASYNC


class HopfieldNetwork:  # network class

    def __init__(self, size, learning):

        self.neurons = np.zeros(size)
        self.weights = np.zeros((size, size))
        self.patterns = []

        if learning == HEBB:
            self.learning_func = lambda x: self.hebb(x)
        elif learning == OJA:
            self.learning_func = lambda x: self.oja(x)
        else:
            raise ValueError("pick hebb or oja ")


    def load_patterns_and_learn(self, patterns):
        self.patterns = patterns
        self.learning_func(patterns)

    def hebb(self, patterns):
        patterns = np.array(patterns)
        self.weights = (patterns.T @ patterns) / len(patterns)
        for i in range(len(self.neurons)):
            self.weights[i][i] = 0.0

    def oja(self, patterns):
        self.hebb(patterns)

        V = np.sum(self.weights * self.neurons)
        eta = 0.01
        iteration_count = 100

        for i in range(iteration_count):
            neurons_matrix = np.array([np.full(len(self.neurons), self.neurons[i]) for i in range(len(self.neurons))])
            self.weights = self.weights + eta * V * (neurons_matrix - V * self.weights)

    def update_one(self):
        index = np.random.randint(0, len(self.neurons))

        activation = np.sum(self.weights[index, :] * self.neurons)

        if activation > 0:
            self.neurons[index] = 1
        else:
            self.neurons[index] = -1

    def update_sync(self):
        new_neurons = self.neurons.copy()
        for i in range(len(self.neurons)):
            activation = np.dot(self.weights[i, :],
                                self.neurons)
            if activation > 0:
                new_neurons[i] = 1
            else:
                new_neurons[i] = -1
        self.neurons = new_neurons

    # updates fraction of neurons given by fraction_updated
    def update_async(self, fraction_updated):
        numbers = len(self.neurons) * fraction_updated
        for _ in range(int(numbers)):
            self.update_one()

    def draw_neurons(self, rows_count, columns_count):
        self.draw(self.neurons, rows_count, columns_count)

    def draw_weights(self):
        image = self.weights.flatten()
        image = image.reshape(len(self.weights[0]), len(self.weights[0]))
        image = np.expand_dims(image, axis=-1)
        plt.imshow(image, cmap=plt.get_cmap('viridis'))
        plt.show()
    def draw(self, target, rows_count, columns_count):
        image = target.flatten()
        image = image.reshape(rows_count, columns_count)
        image = np.expand_dims(image, axis=-1)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.show()