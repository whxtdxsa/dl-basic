import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = x < 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dout = dout * self.out * (1.0 - self.dout) 
        return dout

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.original_x_shape = None
        self.x = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = np.matmul(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.matmul(self.x.T, dout) 
        self.db = np.sum(dout, axis=0)
        dout = np.matmul(dout, self.W.T)
        dout = dout.reshape(*self.original_x_shape)
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None    
        self.t = None    
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: 
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

from util import im2col, col2im
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None 
        self.db = None 

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.matmul(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = self.x.shape

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) # (N, out_h, out_w, FN)
        
        self.db = np.sum(dout, axis = 0)
        
        self.dW = np.matmul(self.col.T, dout)
        self.dW = self.dW.reshape(C, FH, FW, FN).transpose(3, 0, 1, 2)

        dx = np.matmul(dout, self.col_W.T)
        dx = col2im(dx, self.x.shape, FH, FW, self.stride, self.pad)
        
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col =  col.reshape(-1, self.pool_h * self.pool_w)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        

        self.x = x
        self.arg_max = np.argmax(col, axis=1)
        
        del col 
        
        return out

    def backward(self, dout):

        dout = dout.transpose(0, 2, 3, 1).reshape(-1)
        dx = np.zeros((dout.size, self.pool_h * self.pool_w))
        dx[np.arange(self.arg_max.size), self.arg_max.reshape(-1)] = dout

        dx = col2im(dx, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

