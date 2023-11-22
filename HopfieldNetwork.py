import copy

from dataManager import *


# network class
class HopfieldNetwork:
    def __init__(self, size, learning):
        self.size = size  # 2-tuple
        self.length = size[0] * size[1]
        self.patterns = []  # np.full((0, 0), 0.0)
        self.neurons = np.zeros(self.length, dtype='float16')
        self.weights = np.zeros((self.length, self.length), dtype='float16')

        self.energies = [self.calc_energy()]
        self.states = [copy.deepcopy(self.neurons)]

        if learning == HEBB:
            self.learning_func = lambda x, y, z: self.hebb(x, None, None)
        elif learning == OJA:
            self.learning_func = lambda x, y, z: self.oja(x, y, z)
        else:
            raise ValueError("Pick Hebb or Oja!")

    def reset_neurons(self):
        self.neurons = np.zeros(self.length, dtype='float16')
        self.energies = [np.inf]
        self.states = [copy.deepcopy(self.neurons)]

    # no parameters reset implemented - use this function one time only!
    def load_patterns_and_learn(self, patterns, eta=0.001, iterations=100):
        self.patterns = np.array(patterns)
        self.learning_func(self.patterns, eta, iterations)

    def hebb(self, patterns, eta=None, iterations=None):
        self.weights = (patterns.T @ patterns) / len(patterns)  # self.length
        for i in range(self.length):
            self.weights[i][i] = 0.0

    def oja(self, patterns, eta, iterations):

        self.weights = np.float16(np.random.uniform(-1, 1, size=(self.length, self.length)))
        for it in range(iterations):
            old_weights = copy.deepcopy(self.weights)

            for i in range(self.length):
                for j in range(self.length):
                    self.weights[i][j] = (1 - eta) * self.weights[i][j] + eta * np.sum(
                        patterns[:, j] * (patterns[:, i] - patterns[:, j] * self.weights[i][j])) / len(patterns)
                self.weights[i][i] = 0.0

            n_sc = np.linalg.norm(old_weights - self.weights)
            if n_sc < 1E-10:
                break

    def update_one(self):
        index = np.random.randint(0, self.length)
        activation = np.sum(self.weights[index, :] * self.neurons)

        if activation > 0:
            self.neurons[index] = 1
        else:
            self.neurons[index] = -1

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
        self.energies.append(self.calc_energy())

    # updates fraction of neurons given by fraction_updated
    def update_async(self, fraction_updated):
        for iteration in range(int(fraction_updated)):
            for _ in range(self.length):
                self.update_one()

            self.states.append(copy.deepcopy(self.neurons))
            self.energies.append(self.calc_energy())

    def calc_energy(self):
        energy = 0
        # loop overcomes float16 precision and memory problems
        for i in range(self.length):
            energy += np.sum(np.float64(self.weights[i]) * np.float64(self.neurons) * np.float64(self.neurons[i]))
        return energy / 2

    def calc_score(self, max_iter, error, rands):
        visualisations = []
        score = []

        if rands > 0:
            for _ in range(rands):
                self.reset_neurons()
                self.neurons = np.float16(np.random.choice([1, -1], size=self.length))
                self.energies = [self.calc_energy()]
                self.states = [copy.deepcopy(self.neurons)]

                self.update_sync()
                for i in range(max_iter - 1):
                    if (np.array_equal(self.states[i + 1], self.states[i])
                            and self.energies[i] - self.energies[i + 1] < 1E-2):
                        break
                    self.update_sync()

                score.append(np.max([np.mean(self.patterns[i] == self.neurons) for i in range(len(self.patterns))]))
                visualisations.append(self.states)
        else:
            for pattern in self.patterns:
                self.reset_neurons()
                if error > 0:
                    self.neurons = damage_pattern(pattern, error)
                else:
                    self.neurons = copy.deepcopy(pattern)
                self.energies = [self.calc_energy()]
                self.states = [copy.deepcopy(self.neurons)]

                self.update_sync()
                for i in range(max_iter - 1):
                    # network cant 'over-train'
                    if (np.array_equal(self.states[i + 1], self.states[i])
                            and self.energies[i] - self.energies[i + 1] < 1E-2):
                        break
                    self.update_sync()

                score.append(np.mean(pattern == self.neurons))
                visualisations.append(self.states)

        self.reset_neurons()
        return score, visualisations

    def report_score(self, max_iter=100, error=0, rands=0):
        score, visualisations = self.calc_score(max_iter, error, rands)
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
