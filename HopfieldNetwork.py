import copy

import numpy as np

from dataManager import *


# network class
class HopfieldNetwork:
    def __init__(self, size, learning):
        self.size = size  # 2-tuple
        self.length = size[0] * size[1]
        self.patterns = []  # np.full((0, 0), 0.0)
        self.neurons = np.zeros(self.length, dtype='float16')
        self.weights = np.zeros((self.length, self.length), dtype='float16')

        self.energies = []
        self.calc_energy()
        self.scores = []
        self.calc_score()
        self.states = [copy.deepcopy(self.neurons)]

        if learning == HEBB:
            self.learning_func = lambda x, y, z: self.hebb(x, None, None)
        elif learning == OJA:
            self.learning_func = lambda x, y, z: self.oja(x, y, z)
        else:
            raise ValueError("Pick Hebb or Oja!")

    # no parameters reset implemented - use this function one time only!
    def load_patterns_and_learn(self, patterns, eta=0.001, iterations=100):
        self.patterns = np.array(patterns)
        self.learning_func(self.patterns, eta, iterations)

    def hebb(self, patterns, eta=None, iterations=None):  # , collect=True, plot=True):
        # patterns = np.array(patterns)
        self.weights = (patterns.T @ patterns) / len(patterns)  # self.length
        # print('finished')
        for i in range(self.length):
            self.weights[i][i] = 0.0

        # if collect:
        #     self.learning.append(copy.deepcopy(self.weights))
        # if plot:
        #     self.draw_learning()

    def oja(self, patterns, eta, iterations):  # , collect=True, plot=True):
        # self.hebb(patterns)  # , collect=True, plot=False)
        # self.weights = np.zeros((self.length, self.length))
        # new_weights = np.zeros((self.length, self.length))

        self.weights = np.float16(np.random.uniform(-1, 1, size=(self.length, self.length)))
        for it in range(iterations):
            old_weights = copy.deepcopy(self.weights)

            for i in range(self.length):
                for j in range(self.length):
                    self.weights[i][j] = (1 - eta) * self.weights[i][j] + eta * np.sum(
                        patterns[:, j] * (patterns[:, i] - patterns[:, j] * self.weights[i][j])) / len(patterns)
                # print(i)
                self.weights[i][i] = 0.0

            n_sc = np.linalg.norm(old_weights - self.weights)
            # print(n_sc)
            if n_sc < 1E-10:
                break
        # V = np.sum(self.weights * self.neurons)
        #
        # for i in range(iterations):
        #     neurons_matrix = np.array([np.full(self.length, self.neurons[i]) for i in range(self.length)])
        #     self.weights = self.weights + eta * V * (neurons_matrix - V * self.weights)

        # if collect:
        #     self.learning.append(copy.deepcopy(self.weights))
        # if plot:
        #     self.draw_learning()

    def update_one(self):
        index = np.random.randint(0, self.length)
        activation = np.sum(self.weights[index, :] * self.neurons)

        if activation > 0:
            self.neurons[index] = 1
        else:
            self.neurons[index] = -1

    def update_all(self):
        new_neurons = np.zeros(self.length, dtype='float16')
        for index in range(self.length):
            activation = np.sum(self.weights[index, :] * self.neurons)
            if activation > 0:
                new_neurons[index] = 1
            else:
                new_neurons[index] = -1
        self.neurons = new_neurons

    def update_sync(self):
        new_neurons = self.neurons.copy()
        for i in range(self.length):
            activation = np.dot(self.weights[i, :], self.neurons)
            if activation > 0:
                new_neurons[i] = 1
            else:
                new_neurons[i] = -1
        self.neurons = new_neurons

        self.states.append(copy.deepcopy(self.neurons))
        # plot_dataset(self.states, self.size, enum=True)

    # updates fraction of neurons given by fraction_updated
    def update_async(self, fraction_updated):
        for iteration in range(int(fraction_updated)):
            for _ in range(self.length):
                self.update_one()

            self.states.append(copy.deepcopy(self.neurons))
        # plot_dataset(self.states, self.size, enum=True)
        # numbers = self.length * fraction_updated
        # for _ in range(int(numbers)):
        #     self.update_one()

    def calc_energy(self):
        self.energies.append(-np.dot(np.dot(self.neurons.T, self.weights), self.neurons) / 2)

    def calc_score(self):
        visualisations = []
        neurons_copy = copy.deepcopy(self.neurons)
        score = []

        for pattern in self.patterns:
            self.neurons = copy.deepcopy(pattern)
            self.update_all()

            score.append(np.mean(pattern == self.neurons))
            visualisations.append(self.neurons)

        self.neurons = neurons_copy
        return score, visualisations

    def report_score(self):
        score, visualisations = self.calc_score()
        stabilities = np.count_nonzero(np.abs(np.subtract(score, 1)) < 1E-8)
        avg_score = np.mean(score)

        return score, avg_score, stabilities, visualisations

    def draw_neurons(self):
        plot_pattern(self.neurons, self.size)

    def draw_patterns(self):
        plot_dataset(self.patterns, self.size, enum=True)

    def draw_weights(self):
        plt.figure(figsize=[20, 20])
        plt.imshow(np.array(self.weights).reshape(self.length, self.length), cmap='viridis')

    # def draw_learning(self):
    #     plt.figure(figsize=[20, 5 * int(np.ceil(len(self.learning) / 4))])
    #     for it in range(len(self.learning)):
    #         plt.subplot(int(np.ceil(len(self.learning) / 4)), 4, it + 1)
    #         plt.title(str(it))
    #         plt.imshow(np.array(self.learning[it] * -1).reshape((self.length, self.length)), cmap='viridis')
