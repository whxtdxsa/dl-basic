import sys, os
sys.path.append(os.pardir)

from data.mnist import load_mnist
from model.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

print(x_train.shape)

# network = TwoLayerNet() 
