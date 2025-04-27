from data.mnist import load_mnist
from model.conv_net import ConvNet
from dataloader import NumpyDataset
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 100
lr = 0.01
epochs = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

train_dataset = NumpyDataset(x_train, t_train)
test_dataset = NumpyDataset(x_test, t_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

network = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    network.train()
    running_loss = 0.0

    for x_batch, t_batch in train_loader:
        x_batch, t_batch = x_batch.to(device), t_batch.to(device)

        optimizer.zero_grad()
        outputs = network(x_batch)
        loss = criterion(outputs, t_batch)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()

    train_loss_list.append(running_loss / len(train_loader))

    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x_batch, t_batch in test_loader:
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            outputs = network(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == t_batch).sum().item()
            total += t_batch.size(0)
        test_acc = correct / total
        test_acc_list.append(test_acc)

        correct = 0
        total = 0
        for x_batch, t_batch in train_loader:
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            outputs = network(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == t_batch).sum().item()
            total += t_batch.size(0)
        train_acc = correct / total
        train_acc_list.append(train_acc)

    print(f"epoch {epoch}: train_loss {running_loss/len(train_loader):.4f}, train_acc {train_acc:.4f}, test_acc {test_acc:.4f}")


import matplotlib.pyplot as plt
#path = "/data/data/com.termux/files/home/storage/dcim/Graph"
path = "."
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
plt.savefig(path + '/training_result_cnn.png')
plt.clf()
