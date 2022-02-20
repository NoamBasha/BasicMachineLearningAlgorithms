import numpy as np
from numpy import genfromtxt
import sys


def z_score_normalization(samples):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return (samples - mean) / std


def shuffle(samples, labels):
    samples_and_labels = np.concatenate((samples, labels), axis=1)
    np.random.shuffle(samples_and_labels)
    samples = samples_and_labels[:, :-1]
    labels = samples_and_labels[:, -1]
    return samples, labels


class KNN:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = [self.predict_sample(sample) for sample in x_test]
        return np.asarray(predictions)

    def predict_sample(self, sample):
        # computing the distances between the sample and all the training samples
        dists = [np.linalg.norm(sample - train_sample) for train_sample in self.x_train]
        # finding the k indices of the samples with the smallest distance
        k_indices = np.argsort(dists)[0:self.k]
        # finding the k labels of those samples
        k_labels = [self.y_train[i] for i in k_indices]
        # finding the most common label from the k labels
        k_labels = np.asarray(k_labels)
        k_labels = np.ndarray.flatten(k_labels)
        frequencies = {}
        for element in k_labels:
            if element not in frequencies:
                frequencies[element] = 1
            else:
                frequencies[element] += 1
        prediction = max(frequencies, key=frequencies.get)
        return prediction


if __name__ == '__main__':

    # Program's parameters
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    out_fname = sys.argv[4]
    k = int(sys.argv[5])

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

    knn = KNN(k)
    knn.fit(samples, labels)
    predictions = knn.predict(test_x)

    out_f = open(out_fname, "w")
    for i in range(len(test_x)):
        out_f.write(f"{predictions[i]}\n")
    out_f.close()
