import numpy as np
from numpy import genfromtxt
import sys


def z_score_normalization(samples):
    # z = (x - m) / s
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return (samples - mean) / std


def shuffle(samples, labels):
    samples_and_labels = np.concatenate((samples, labels), axis=1)
    np.random.shuffle(samples_and_labels)
    samples = samples_and_labels[:, :-1]
    labels = samples_and_labels[:, -1]
    return samples, labels


class Perceptron:
    def __init__(self):
        self.weights = []
        self.epochs = 500
        self.eta = 0.1

    def find_y_hat(self, sample):
        return np.argmax(np.dot(self.weights, sample))

    def fit(self, train_x, train_y):
        n_samples, n_features = train_x.shape
        # weights = matrix of zeros size 5x3
        self.weights = np.zeros(3 * n_features).reshape(3, n_features)
        for e in range(self.epochs):
            # for every 100 epochs dividing the eta by 2
            if e % 100 == 0:
                self.eta /= 2
            for x, y in zip(train_x, train_y):
                y_hat = self.find_y_hat(x)
                if y != y_hat:
                    self.weights[y_hat] -= self.eta * x
                    self.weights[y] += self.eta * x

    def predict(self, test_x):
        return [self.find_y_hat(x) for x in test_x]


if __name__ == '__main__':

    # Program's parameters
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    out_fname = sys.argv[4]

    train_x = genfromtxt(sys.argv[1], delimiter=',').astype(float)
    train_y = genfromtxt(sys.argv[2], delimiter=',').astype(int)
    train_y = train_y.reshape(train_y.shape[0], 1)
    test_x = genfromtxt(sys.argv[3], delimiter=',').astype(float)

    # Normalizing the data with z score normalization
    train_x = z_score_normalization(train_x)
    test_x = z_score_normalization(test_x)

    # adding bias to the data
    bias_train = [1] * len(train_x)
    bias_train = np.asarray(bias_train).reshape(len(train_x), 1)
    train_x = np.append(train_x, bias_train, axis=1)
    bias_test = [1] * len(test_x)
    bias_test = np.asarray(bias_test).reshape(len(test_x), 1)
    test_x = np.append(test_x, bias_test, axis=1)

    # Shuffling the data
    samples, labels = shuffle(train_x, train_y)
    # Converting the labels to int
    labels = [int(y) for y in labels]

    perceptron = Perceptron()
    perceptron.fit(samples, labels)
    predictions = perceptron.predict(test_x)

    out_f = open(out_fname, "w")
    for i in range(len(test_x)):
        out_f.write(f"{predictions[i]}\n")
    out_f.close()
