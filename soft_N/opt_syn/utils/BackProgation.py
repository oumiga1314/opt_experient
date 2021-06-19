from __future__ import print_function
from __future__ import division
import math
import numpy as np
import pandas as pd
import time
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
class backProgation:
       #初始化激活层
    def _init_activations(self, size=None):
        self.activations = [np.zeros((i, size))
                            for i in self.shape[1:]] if size else []
       #初始化输出层
    def _init_outputs(self, size=None):
        self.outputs = [np.zeros((i, size))
                        for i in self.shape[1:]] if size else []
    #初始化梯度
    def _init_deltas(self, size=None):
        self.deltas = [np.zeros((i, size))
                       for i in self.shape[1:]] if size else []

    def _init_dropout(self, size=None):
        self.dropout = [np.zeros((i, size))
                        for i in self.shape[1:]] if size else []

     # 就是进行one-hot编码
    def vectorize_output(self):
           """Tranforms a categorical label represented by an integer into a vector."""
           num_labels =self.shape[-1]
           num_examples = self.target.shape[1]
           result = np.zeros((num_labels, num_examples))
           for l, c in zip(self.target.ravel(), result.T):
               c[l] = 1
           self.target = result

    def vectorize_output_test(self, test_label):
           # 该方法就是把标签的代表的地方置为1，其余的地方置为0
           """Tranforms a categorical label represented by an integer into a vector."""
           num_labels = np.unique(test_label).shape[0]
           print(test_label.shape)
           num_examples = test_label.shape[1]
           result = np.zeros((num_labels, num_examples))
           for l, c in zip(test_label.ravel(), result.T):
               c[l] = 1
           self.test_target = result

    def labelize(self, data):
           """Tranform a matrix (where each column is a data) into an list that contains the argmax of each item."""

           return np.argmax(data, axis=0)

    def feed_forward(self, data, return_labels=False):
           """Given the input and, return the predicted value according to the current weights."""
           result = data
           # mu = np.mean(result, axis=0)
           # sigma = np.std(result, axis=0)
           # result = (data - mu) / sigma
           # num examples in this batch = data.shape[1]
           #result = data
           # if z = w*a +b
           # then activations are \sigma(z)
           try:
               self._init_outputs()
               self._init_activations()
           except MemoryError:
               print(self.MEMORYERROR_MESSAGE)
               raise

           self.activations.append(result)
           self.outputs.append(result)

           for w, b in zip(self.weights, self.biases):
               result = np.dot(w, result) + b
               self.outputs.append(result)
               result = self.activation(result)
               self.activations.append(result)

           if return_labels:
               result = self.labelize(result)
           # the last level is the activated output
           return result
    #计算梯度
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
           # 这样做的目的是为了之后的权重改变好从第一层开始
           self.deltas.reverse()

    def cost(self, predicted, target):

           """Calculate the cost function using the current weights and biases"""
           # the cost is normalized (divided by numer of samples)
           if self.classification:
               return np.sum(predicted == target) / len(predicted)
           # else:
           #     return (np.linalg.norm(np.abs(predicted - target))) # predicted.shape[1]

    def cost1(self, predicted, target):
           self.vectorize_output_test(target)
           cost_1=-np.sum(self.test_target*(np.log(predicted)))/predicted.shape[1]
           return cost_1

    def __init__(self, shape_or_file, activation=sgm, dropout=False):
        self.shape = shape_or_file
        #指定使用哪个激活函数
        self.activation = activation
        self._init_activations()
        self._init_outputs()
        if dropout:
            self._init_dropouts()

    def train(self, train_data=None,
                 train_labels=None,
                 batch_size=100,
                 classification=True,
                 weights=None,
                 bias=None,
                 method='SGD'
                 ):
           if method is not 'SGD':
               print("This method is not supported at the moment")
               exit()
           if train_data is None or train_labels is None:
               print("Both trainig data and training labels are required to start training")
               return
           self.classification = classification
           self.weights = weights
           self.biases = bias
           self.batch_size = batch_size

           # 对传入的训练数据进行处理
           train_data = np.array(train_data)
           train_labels = np.array(train_labels)

           # 把训练数据转置为了后面的计算方便
           self.data = train_data.T
           self.target = train_labels.T



           if self.classification:
               self.original_labels = self.target.ravel()

               self.vectorize_output()
           # 判断输入输出维度是否正确
           assert self.data.shape[0] == self.shape[0], \
               ('Input and shape of the network not compatible: ', self.data.shape[0], " != ", self.shape[0])
           assert self.target.shape[0] == self.shape[-1], \
               ('Output and shape of the network not compatible: ', self.target.shape[0], " != ", self.shape[-1])

           # 每次计算梯度重新初始化下
           self._init_outputs()
           self._init_activations()

           # feed forward the input 对输入的数据进行前项传播
           self.feed_forward(self.data)
           self.calculate_deltas(self.data, self.target)

           w_ = [np.dot(d, a.T)  for d, a in zip(self.deltas, self.activations)]
           b_ = [np.sum(d, axis=1).reshape(b.shape)  for b, d in zip(self.biases, self.deltas)]

           return w_, b_

       #提供一个预测值的方法

    def predict(self, test_data,test_labels,weigths,bias):
        self.weights=weigths
        self.biases=bias
        if test_data is not None and test_labels is not None:

            self.test_data = np.array(test_data).T
            self.test_origin=test_labels.ravel()
            self.test_labels = np.array(test_labels).T
           #第一计算准确率
            cost = self.cost(
                self.feed_forward(self.test_data, return_labels=True),self.test_origin)
            #d第二个计算损失函数
            cost1=self.cost1(self.feed_forward(self.test_data,return_labels=False),self.test_labels)

        return cost,cost1