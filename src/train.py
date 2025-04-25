import sys, os
sys.path.append(os.pardir)

import numpy as np
# from models.two_layer_net import TwoLayerNet
from models.simple_conv_net import SimpleConvNet
from data.mnist import load_mnist
from optimizer import *
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
iters_num = 10000
batch_size = 100
learning_rate = 0.005
iters_per_epoch = max(train_size // batch_size, 1)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = SimpleConvNet(input_dim=(1, 28, 28), conv_param={'filter_num': 1, 'filter_size': 5, 'pad':0, 'stride':2}, hidden_size=10)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer = AdaGrad(learning_rate)
    optimizer.update(params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"epochs {i//iters_per_epoch}/{iters_num//iters_per_epoch} | train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}")

import matplotlib.pyplot as plt
path = "/data/data/com.termux/files/home/storage/dcim/Graph"

x1 = np.arange(len(train_loss_list))
x2 = np.arange(len(train_acc_list))

plt.figure(figsize=(10, 4))  # 도화지 크기 설정 (가로 10, 세로 4)

# 첫 번째 그래프: 손실
plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째
plt.plot(x1, train_loss_list, label='train loss')
plt.xlabel("iters")
plt.ylabel("loss")
plt.legend()
plt.title("Loss Curve")

# 두 번째 그래프: 정확도
plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()  # 그래프 간 여백 자동 조절
plt.savefig(path + '/adagrad_training_result.png')
plt.clf()
