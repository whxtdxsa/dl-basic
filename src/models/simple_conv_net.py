import sys, os
sys.path.append(os.pardir)

from layers import *
import numpy as np
from collections import OrderedDict
class SimpleConvNet:
    def __init__(self, input_dim, conv_param, hidden_size, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        pad = conv_param['pad']
        stride = conv_param['stride']

        input_size = input_dim[1]
        conv_output_size = int(1 + (input_size + 2 * pad - filter_size) / stride)

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(conv_output_size * conv_output_size * filter_num, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        #self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        #self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride, pad)
        self.layers['Relu1'] = Relu()
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        #self.layers['Relu2'] = Relu()
        #self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for l,layer in self.layers.items():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        l = self.last_layer.forward(y, t)
        return l

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1: t = np.argmax(t, axis = 1)
        return np.sum(y == t) / x.shape[0]
        
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'] .db
        #grads['W3'] = self.layers['Affine2'].dW
        #grads['b3'] = self.layers['Affine2'] .db


        return grads

