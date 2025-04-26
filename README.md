# dl-study

# Dive To Deep Learning

Date: April 21, 2025
Multi-select: Vision
Select: Study
Status: In progress

# Overview

---

[https://github.com/whxtdxsa/dl-basic.git](https://github.com/whxtdxsa/dl-basic.git)

딥러닝 이론이 인공지능 학습에 실제로 어떻게 적용되는 지 알아보는 프로젝트입니다. 

이 프로젝트에는 두 가지 서브 목표가 있습니다. 

1. Theory2Numpy(딥러닝 이론을 코드로 구현)

먼저 Numpy를 사용하여 기본적인 신경망을 구현합니다. 이론이 어떻게 코드로 구현되는 지 학습합니다.

1. Numpy2Pytorch(Pytorch와 Numpy비교)

Numpy로 작성된 신경망을 Pytorch로 사용하여 재구성합니다. Pytorch가 Numpy 복잡한 연산을 어떻게 간소화했는지를 알아보고 Numpy에서는 지원하지 않는 GPU를 훈련을 공부합니다.

### References

---

딥러닝 이론과 코드는 아래의 자료를 참고했습니다. 

- DeepLearning(Ian Goodfellow, Yoshua Bengio, Aaron Courville · 2016)
- 밑바닥부터 시작하는 딥러닝(사이코 고키)

# Theory2Numpy

---

### Project Information

---

Task Type: Classification

Data: MNIST dataset

Model: Two Layer Net & CNN
Loss: Cross Entropy

### Directory Structure

---

proj/

|- - data/

|- - src/

      |- - models/

            |- - two_layer_net.py

      |- - layers.py

      |- - train.py

      |- - utils.py

      |- - functions.py

### Development Environment

---

![SmartSelect_20250422_235141_Notion.jpg](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250422_235141_Notion.jpg)

OS: Android 14 aarch64

Python 3.12.10

Numpy 2.2.4

# NLP

---

### Model: Two Layer Net

---

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()
```

- When an object is created, Parameters and Layers are saved as dictionary types.
- The weights are initialized close to zeros but not exact zeros.
    - A large initial weights weight causes gradient vanish.
    - Zero initial weights causes same gradient updates for all weights
- The last layer is saved seperately.
    - When a model is used in inferrence, it doesn’t use last layer specifically in softmax layer.
    - It is useful when the user want to customize the last layer.

![Structure of Two Layer Net](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250421_203928_Samsung_Notes.jpg)

Structure of Two Layer Net

```python
class TwoLayerNet:
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        return np.sum(y == t) / float('x.shape[0]')
        
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grad = {}
        grad['W1'] = self.layers['Affine1'].dW
        grad['b1'] = self.layers['Affine1'].db
        grad['W2'] = self.layers['Affine2'].dW
        grad['b2'] = self.layers['Affine2'].db

        return grad
```

- Meaning of axis=n in numpy
    - When shape is (D0, D1, …, Dk), the shape of result would be (D0, D1, …, Dk) without Dn.
    - Calculate in the dimension without n-th dim.
    

### Layers

---

```python
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
        self.W = w
        self.b = b

        self.original_x_shape = None
        self.x = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
 
        out = np.matmul(x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.matmul(self.x.T, dout) 
        self.db = np.sum(dout, axit=0)
        dout = np.matmul(dout, self.W.T)
        dout = dout.reshape(*self.original_x_shape)
        return dout
```

```python
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
```

- The reason for saving loss to self.loss: SoftmaxWithLoss is a layer so that it is general to have its states. We can refer to it whenever we need without computing it again.
- dx[np.arange(batch_size), self.t] -= 1: For each batches extract the true label from y.

![Computational Graph of backward propagation for getting gradient.](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250422_202004_Samsung_Notes.jpg)

Computational Graph of backward propagation for getting gradient.

![Computational Graph of Softmax with cross entropy](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250423_222815_Samsung_Notes.jpg)

Computational Graph of Softmax with cross entropy

### Training

---

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
iters_num = 10000
batch_size = 100
learning_rate = 0.1
iters_per_epoch = max(train_size // batch_size, 1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
	
		optimizer = SGD(learning_rate)
		optimizer.update(params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"epochs {i//iters_per_epoch}/{iters_num//iters_per_epoch} | train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}")

```

- Stochastic Gradient Descent
    - Get random batch masks for each iterations.
    - Define an epoch as comsuming all training data assumed that there is no same data between each batches.
    - We can calculate total epoch before training: iters_num / iters_per_epoch

![Loss graph for 10000 iterations. The variance is larger and loss is smaller during training. Accuracy for each epochs. It converged close to 1.0 accuracy during training.](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/training_result.png)

Loss graph for 10000 iterations. The variance is larger and loss is smaller during training. Accuracy for each epochs. It converged close to 1.0 accuracy during training.

- The usage of matplotlib for getting a graph
    
    ```python
    import matplotlib.pyplot as plt
    path = "/data/data/com.termux/files/home/storage/dcim/Graph"
    
    x1 = np.arange(len(train_loss_list))
    x2 = np.arange(len(train_acc_list))
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)  
    plt.plot(x1, train_loss_list, label='train loss')
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    
    plt.subplot(1, 2, 2) 
    plt.plot(x2, train_acc_list, label='train acc')
    plt.plot(x2, test_acc_list, label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    
    plt.tight_layout()  
    plt.savefig(path + '/training_result.png')
    plt.clf()
    
    ```
    

### Optimizer

---

```python
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, param in params.items():
                self.v[key] = np.zeros_like(param)

        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr):
        self.h = None
        self.lr = lr
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, param in params.items():
                self.h[key] = np.zeros_like(param) 

        for key in params.keys():
            self.h[key] = self.h[key] + grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

![Loss and accuracy graph of Momentum with lr = 0.05.](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/momentum_training_result.png)

Loss and accuracy graph of Momentum with lr = 0.05.

![Loss and accuracy graph of AdaGrad with lr = 0.03.](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/adagrad_training_result.png)

Loss and accuracy graph of AdaGrad with lr = 0.03.

# CNN

---

### Model: SimpleConvNet

```python
class SimpleConvNet:
    def __init__(self, input_dim, conv_param, hidden_size, output_size, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        pad = conv_param['pad']
        stride = conv_param['stride']
        
        c_size, h_size, w_size = input_dim
        input_size = c_size * h_size * w_size
        conv_output_size = int(1 + (h_size + 2 * pad - filter_size) / stride)
        # pool_output_size = int(1 + (conv_output_size - 2) / 2)
        #
        self.params = {}
        self.params['W1'] = np.sqrt(2/input_size) * np.random.randn(filter_num, c_size, filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        h2 = conv_output_size * conv_output_size * filter_num
        self.params['W2'] = np.sqrt(2/h2) * np.random.randn(h2, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.sqrt(2/hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride, pad)
        self.layers['Relu1'] = Relu()
        # self.layers['Pooling1'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

```

- By resorce issue of work environment, I should remove pooling layer. The pooling layer takes much resource during training.

### Layers

---

```python
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

```

```python
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
        
        return out

    def backward(self, dout):

        dout = dout.transpose(0, 2, 3, 1).reshape(-1)
        dx = np.zeros((dout.size, self.pool_h * self.pool_w))
        dx[np.arange(self.arg_max.size), self.arg_max.reshape(-1)] = dout

        dx = col2im(dx, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

```

![Structure of convolution layer. To get computational efficiency, we transform multi dimensional tensor to two dimensional matrix.](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250426_152747_Samsung_Notes.jpg)

Structure of convolution layer. To get computational efficiency, we transform multi dimensional tensor to two dimensional matrix.

![Structure of pooling layer. Simmiliar tricks with convolution layer](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/SmartSelect_20250426_152803_Samsung_Notes.jpg)

Structure of pooling layer. Simmiliar tricks with convolution layer

### Training

---

![training_result.png](Dive%20To%20Deep%20Learning%201dc5167b75fd80398b6be0bd5191072b/training_result%201.png)

- Higher accuracy than NLP
    - CNN can reflect the structure of neighborhood in data.
- Limit Point
    - The model is superficial so that it wouldn’t catch more feature of data.
    - There is no pooling layer that makes the model robust to small change of data.

# Numpy2Pytorch

---
