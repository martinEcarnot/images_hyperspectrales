# Global import
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import classification_report


def load(use_path):
    """
    Creation of a dataframe with the paths of all the files in the folder and the class

    :param use_path: path of the folder containing all the files for a given task
    :return: a sorted dataframe
    """
    list_path = os.listdir(use_path)
    class_image = use_path.split("var", 1)[1][0]
    list_class = [class_image] * len(list_path)

    data = pd.DataFrame(list(zip(list_path, list_class)), columns=["path", "class"])

    return data.sort_values(by=['path'], ignore_index=True)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax

        # 200, 200, 10 -> # 200, 200, 30
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(5, 5), padding=(2, 2))
        # 200, 200, 30 -> # 100, 100, 30
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 100, 100, 30 -> # 100, 100, 60
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(5, 5), padding=(2, 2))
        # 100, 100, 60 -> # 50, 50, 60
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 50, 50, 60 -> # 50, 50, 90
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=90, kernel_size=(5, 5), padding=(2, 2))
        # 50, 50, 90 -> # 25, 25, 90
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 25, 25, 90 -> # 25, 25, 120
        self.conv4 = nn.Conv2d(in_channels=90, out_channels=120, kernel_size=(5, 5), padding=(2, 2))
        # 25, 25, 120 -> # 12, 12, 120
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # 12, 12, 120 -> # 12, 12, 150
        self.conv4 = nn.Conv2d(in_channels=120, out_channels=150, kernel_size=(5, 5), padding=(2, 2))
        # 12, 12, 150 -> # 6, 6, 150
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # 12, 12, 150 -> # 12, 12, 180
        self.conv5 = nn.Conv2d(in_channels=150, out_channels=180, kernel_size=(5, 5), padding=(2, 2))
        # 12, 12, 180 -> # 6, 6, 180
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        # 6, 6, 180 -> # 6, 6, 210
        self.conv6 = nn.Conv2d(in_channels=180, out_channels=200, kernel_size=(5, 5), padding=(2, 2))
        # 6, 6, 210 -> # 3, 3, 210
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(3 * 3 * 210, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear2 = nn.Linear(30, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print("x.shape 1: ", input_data.shape)
        x = self.conv1(input_data)
        x = self.relu(x)
        # print("x.shape 2: ", x.shape)
        x = self.pool1(x)

        # print("x.shape: ", x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        # print("x.shape: ", x.shape)
        x = self.pool2(x)

        # print("x.shape: ", x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        # print("x.shape: ", x.shape)
        x = self.pool3(x)

        # print("x.shape: ", x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        # print("x.shape: ", x.shape)
        x = self.pool4(x)

        # print("x.shape: ", x.shape)
        x = self.conv5(x)
        x = self.relu(x)
        # print("x.shape: ", x.shape)
        x = self.pool5(x)

        # print("x.shape: ", x.shape)
        x = self.conv6(x)
        x = self.relu(x)
        # print("x.shape: ", x.shape)
        x = self.pool6(x)

        # print("x.shape: ", x.shape)
        x = self.flatten(x)
        # print("x.shape: ", x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


movement_dir_train = 'C:/Users/kiera/Documents/EMA/3A/2IA/Deep Learning/Projet/data/project/train/'
movement_dir_valid = 'C:/Users/kiera/Documents/EMA/3A/2IA/Deep Learning/Projet/data/project/valid/'
movement_dir_test = 'C:/Users/kiera/Documents/EMA/3A/2IA/Deep Learning/Projet/data/project/test/'

train_set = Movement_Dataset(movement_dir_train)
valid_set = Movement_Dataset(movement_dir_valid)
test_set = Movement_Dataset(movement_dir_test)

epochs = 200
batch_size = 200

# Create data loaders
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
model = model.double()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

print(model)

for epoch in range(epochs):
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch_num, input_data in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = input_data

        x = x.to(device).double()
        y = y.to(device).double()

        output = model(x)
        s = nn.Softmax()
        pred = s(output)
        for k in range(len(pred)):
            if pred[k][0] > 0.5 and y[k][0] == 1:
                correct += 1
            elif pred[k][0] < 0.5 and y[k][1] == 1:
                correct += 1
            total += 1
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if batch_num % 1 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f | Accuracy %6.2f' % (
            epoch, batch_num, loss.item(), correct / total))
    print('Epoch %d | Loss %6.2f | Accuracy %6.2f' % (epoch, sum(losses) / len(losses), correct / total))

    model.eval()
    losses = []
    correct = 0
    total = 0
    for batch_num, input_data in enumerate(testloader):
        x, y = input_data
        x = x.to(device).double()
        y = y.to(device).double()

        output = model(x)
        s = nn.Softmax()
        pred = s(output)
        for k in range(len(pred)):
            if pred[k][0] > 0.5 and y[k][0] == 1:
                correct += 1
            elif pred[k][0] < 0.5 and y[k][1] == 1:
                correct += 1
            total += 1

        loss = criterion(output, y)
        losses.append(loss.item())

    print('Validation Epoch %d | Val_Loss %6.2f | Val_Accuracy %6.2f' % (epoch, sum(losses) / len(losses), correct / total))
