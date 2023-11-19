import csv
import matplotlib.pyplot as plt
import numpy as np
import png

from constants import *


def load_csv(filename):
    result = []
    with open(PATTERN_PATH + filename, newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            result.append(np.array(row))
    return result


def export_pictures(in_path, in_names, out_path, out_name):
    csv_lists = []
    for name in in_names:
        image = png.Reader(filename='{}/{}'.format(in_path, name)).asDirect()

        values = []
        pixels = [list(row) for row in image[2]]
        for row in pixels:
            for byte in row:
                values.append(-1) if byte == 255 else values.append(1)
        csv_lists.append(values)

    with open('{}/{}'.format(out_path, out_name), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(csv_lists)


def damage_pattern(pattern, error):
    noise = np.random.choice([1, -1], size=len(pattern), p=[1 - error, error])
    return np.multiply(pattern, noise)


def plot_pattern(pattern, size):
    plt.figure(figsize=[5, 5])
    plt.imshow(np.array(pattern * -1).reshape(size[0], size[1]), cmap='gray')


def plot_dataset(dataset, size, enum=False):
    plt.figure(figsize=[20, 5 * int(np.ceil(len(dataset) / 4))])
    for pattern in range(len(dataset)):
        plt.subplot(int(np.ceil(len(dataset) / 4)), 4, pattern + 1)
        if enum:
            plt.title(str(pattern))
        plt.imshow(np.array(dataset[pattern] * -1).reshape(size), cmap='gray')
