from __future__ import print_function
from __future__ import division

import numpy as np
import json

import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def sgm(x, der=False):
    """Logistic sigmoid function.
    Use der=True for the derivative."""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)


@np.vectorize
def relu(x, der=False):
    """Rectifier activation function.
    Use der=True for the derivative."""
    if not der:
        return np.maximum(0, x)
    else:
        if x <= 0:
            return 0
        else:
            return 1


class NeuralNetwork(object):
    """Neural Network class.

    Args:
        shape (list): shape of the network. First element is the input layer, last element
        is the output layer.
        activation (optional): pass the activation function. Defaults to sigmoid.
    """

    WRONGTYPE_MESSAGE = "The network should be initialized with either a list or a string"
    MEMORYERROR_MESSAGE = "Not enough memory to initialize the network"
    FILENOTFOUNDERROR_MESSAGE = "There specified file does not exist"
    WRONGSHAPE_MESSAGE = "There must be at least 2 layers in the network"

    # outputs: output of the layers (before the sigmoid)
    # activations: outputs after the sigmoid
    #注意此处的初始化过程，要从下一层的神经刀上一层的神经元来理解

    def _init_weights(self):
        print("初始化weights")
        self.weights = [np.random.randn(j, i)/np.sqrt(j) for i, j in zip(
            self.shape[:-1], self.shape[1:])]

    def _init_biases(self):
        print("初始化weights")
        self.biases = [np.random.randn(i, 1) for i in self.shape[1:]]

    def _init_activations(self, size=None):
        self.activations = [np.zeros((i, size))
                            for i in self.shape[1:]] if size else []

    def _init_outputs(self, size=None):
        self.outputs = [np.zeros((i, size))
                        for i in self.shape[1:]] if size else []

    def _init_deltas(self, size=None):
        self.deltas = [np.zeros((i, size))
                       for i in self.shape[1:]] if size else []

    def _init_dropout(self, size=None):
        self.dropout = [np.zeros((i, size))
                        for i in self.shape[1:]] if size else []


    def __init__(self, shape_or_file, activation=sgm, dropout=False):
        if isinstance(shape_or_file, str):
            try:
                self.load(shape_or_file)
            except FileNotFoundError:
                print(self.FILENOTFOUNDERROR_MESSAGE)
                raise
            except MemoryError:
                print(self.MEMORYERROR_MESSAGE)
                raise

        elif isinstance(shape_or_file, list):
            if len(shape_or_file) < 2:
                print(self.WRONGSHAPE_MESSAGE)
                raise ValueError

            try:
                self.shape = shape_or_file
                self.activation = activation
                self._init_weights()
                self._init_biases()
                self._init_activations()
                self._init_outputs()
                if dropout:
                    self._init_dropouts()
            except MemoryError:
                print(self.MEMORYERROR_MESSAGE)
                raise
        # 返回权重

    def get_weight(self):
        return self.weights

    # 返回bias
    def get_b(self):
        return self.biases
    def vectorize_output(self):
        """Tranforms a categorical label represented by an integer into a vector."""
        num_labels = np.unique(self.target).shape[0]
        num_examples = self.target.shape[1]
        result = np.zeros((num_labels, num_examples))
        for l, c in zip(self.target.ravel(), result.T):
            c[l] = 1
        self.target = result

    def labelize(self, data):
        """Tranform a matrix (where each column is a data) into an list that contains the argmax of each item."""
        return np.argmax(data, axis=0)

    def feed_forward(self, data, return_labels=False):
        """Given the input and, return the predicted value according to the current weights."""
        result = data
        # num examples in this batch = data.shape[1]

        # if z = w*a +b
        # then activations are \sigma(z)
        try:
            self._init_outputs()
            self._init_activations()
        except MemoryError:
            print(self.MEMORYERROR_MESSAGE)
            raise

        self.activations.append(data)
        self.outputs.append(data)

        for w, b in zip(self.weights, self.biases):
            result = np.dot(w, result) + b
            self.outputs.append(result)
            result = self.activation(result)
            self.activations.append(result)

        if return_labels:
            result = self.labelize(result)

        # the last level is the activated output
        return result

    def calculate_deltas(self, data, target):
        """ Given the input and the output (typically from a batch),
        it calculates the corresponding deltas.
        It is assumed that the network has just feed forwarded its batch.
        Deltas are stored in a (n, k) matrix, where n is the dimensions of the
        corresponding layer and k is the number of examples.
        """
        # num_examples = data.shape[1]
        # delta for the back propagation #初始化梯度
        try:
            self._init_deltas()
        except MemoryError:
            print(self.MEMORYERROR_MESSAGE)
            raise

        # calculate delta for the output level
        delta = np.multiply(
            self.activations[-1] - target,
            self.activation(self.outputs[-1], der=True)
        )
        self.deltas.append(delta)

        # since it's back propagation we start from the end #这个是先求最后一个的梯度，然后依次向前推
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                np.dot(
                    self.weights[l].T,
                    self.deltas[steps - l]
                ),
                self.activation(self.outputs[l], der=True)
            )
            self.deltas.append(delta)

        # delta[i] contains the delta for layer i+1
        #这样做的目的是为了之后的权重改变好从第一层开始
        self.deltas.reverse()

    def update_weights(self, total, learning_rate):
        """Use backpropagation to update weights"""
        self.weights = [w - (learning_rate / total) * np.dot(d, a.T)
                        for w, d, a in zip(self.weights, self.deltas, self.activations)]

    def update_biases(self, total, learning_rate):
        """Use backpropagation to update the biases"""
        # summing over the columns of d, as each column is a different example
        self.biases = [b - (learning_rate / total) * (np.sum(d, axis=1)).reshape(b.shape)
                       for b, d in zip(self.biases, self.deltas)]


    def train(
            self,
            train_data=None,
            train_labels=None,
            batch_size=100,
            epochs=20,
            learning_rate=.3,
            print_cost=False,
            classification=True,
            test_data=None,
            test_labels=None,
            plot=False,
            method='SGD'):
        """Train the network using the specified method"""
        if method is not 'SGD':
            print("This method is not supported at the moment")
            exit()
        self.classification = classification
        if plot:
            self.training_error = []

        # normalize inputs?
        # self.input = (np.array(input) / np.amax(input, axis = 0)).T
        # self.target = (np.array(target) / np.amax(target)).T

        if test_data is not None and test_labels is not None:
            self.test_data = np.array(test_data).T
            self.test_labels = np.array(test_labels).T
            self.testing_error = []

        for epoch in range(epochs):
            # for each epoch, we reshuffle the data and train the network

            print("Starting epoch:", epoch + 1, "/", epochs, end=" ")



            self.update_weights(batch_size, learning_rate)
            self.update_biases(batch_size, learning_rate)

            if print_cost:
                if self.classification:
                    cost = self.cost(
                        self.feed_forward(self.data, return_labels=True),
                        self.original_labels
                    )
                    if plot:
                        self.training_error.append(cost)
                    print("\terror or the training set is {0:.2f}%\n".format(
                        cost * 100), end='')
                    if test_data is not None and test_labels is not None:
                        cost = self.cost(
                            self.feed_forward(
                                self.test_data, return_labels=True),
                            self.test_labels
                        )
                        if plot:
                            self.testing_error.append(cost)
                        print(
                            "\terror or the test set is {0:.2f}%\n".format(
                                cost * 100))

                else:
                    forwarded = self.feed_forward(self.data)
                    print("error is \n", self.cost(forwarded, self.target))

        if plot:
            plotting_data = {"TrainingError": self.training_error}
            if test_data is not None and test_labels is not None:
                plotting_data["Testing Error"] = self.testing_error
            fig, ax = plt.subplots()
            errors = pd.DataFrame(plotting_data)
            errors.plot(ax=ax)
            plt.show()

    def predict(self, data):
        if isinstance(data, list):
            data = np.array(data).T
        return self.feed_forward(data)

    # def save(self, file_location):
    #     """Save network's data in a JSON file located in file_location"""
    #     data = {
    #         "shape": self.shape,
    #         "weights": [w.tolist() for w in self.weights],
    #         "biases": [b.tolist() for b in self.biases]
    #     }
    #     with open(file_location, 'w') as fp:
    #         json.dump(data, fp)
    #
    # def load(self, file_location):
    #     with open(file_location, 'r') as fp:
    #         data = json.load(fp)
    #     try:
    #         self.shape = data["shape"]
    #         self.weights = [np.array(w) for w in data["weights"]]
    #         self.biases = [np.array(b) for b in data["biases"]]
    #     except KeyError as e:
    #         print("Load failed, the json file does not contain the required key ", e)
    #         raise
