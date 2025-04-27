from data.mnist import load_mnist
from model.two_layer_net import TwoLayerNet
from dataloader import NumpyDataset
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 100
lr = 0.01
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

train_dataset = NumpyDataset(x_train, t_train)
test_dataset = NumpyDataset(x_test, t_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=10).to(device)
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
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, t_batch in test_loader:
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            outputs = network(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == t_batch).sum().item()
            total += t_batch.size(0)

    acc = correct / total
    test_acc_list.append(acc)

    print(f"epoch {epoch}: train_loss {running_loss/len(train_loader):.4f}, test_acc {acc:.4f}")
