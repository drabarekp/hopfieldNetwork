import csv

import numpy as np

path = 'patterns/'


def load(filename):
    result = []
    with open(path + filename, newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            result.append(np.array(row))
    return result
