# Global import
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import walk
import spectral as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load(use_path):
    """
    Creation of a dataframe with the paths of all the files in all folders and the class
    :param use_path: path of the global folder
    :return: a sorted dataframe
    """
    folders = next(walk(use_path), (None, None, []))[1]  # Retrieve folders
    dataframe = pd.DataFrame(columns=["img", "hdr", "class"])
    for folder in folders:
        path = os.path.join(use_path, folder)
        list_path = os.listdir(path)  # Retrieve all files (no folder here)
        # Class
        class_image = folder.split("var", 1)[1][0]
        list_class = [class_image] * (len(list_path) // 2)
        # Path .img
        list_img = [x for x in list_path if "img" in x]
        list_img = [os.path.join(path, x) for x in list_img]
        # Path .hdr
        list_hdr = [x for x in list_path if "hdr" in x]
        list_hdr = [os.path.join(path, x) for x in list_hdr]

        data = pd.DataFrame(list(zip(list_img, list_hdr, list_class)), columns=["img", "hdr", "class"])
        dataframe = pd.concat([dataframe, data], ignore_index=True)
    return dataframe


class CustomDataset(Dataset):
    """
    Create dataset for the network
    """

    def __init__(self, df_path):
        """
        Initialisation of the dataset
        :param df_path: one of the three dataframe containing paths of files
        """
        super(CustomDataset, self).__init__()
        # self.paths_img = df_path["img"]
        self.paths_hdr = df_path["hdr"]
        self.labels = df_path["class"]

    def __len__(self):
        return len(self.paths_hdr)

    def __getitem__(self, idx):
        """
        Creation of the tensor and the label
        :param idx: indice to select the path in the list
        :return: tensor, label
        """
        label = self.labels[idx]
        if label == '1':
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])
        # path_img = self.paths_img[idx]
        path_hdr = self.paths_hdr[idx]
        img = sp.open_image(path_hdr)
        img_tensor = torch.tensor(img[:, :, :])

        return img_tensor, label


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax

        self.conv1 = nn.Conv2d(in_channels=10, out_channels=80, kernel_size=(7, 7), stride=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=(3, 3))
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=(3, 3))
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3*3*640, 300)
        self.dropout = nn.Dropout(0.4)
        # self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(300, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print("x.shape 0: ", input_data.shape)
        x = self.conv1(input_data)
        x = self.tanh(x)
        # print("x.shape 1: ", x.shape)
        x = self.pool1(x)
        # print("x.shape 1: ", x.shape)

        x = self.conv2(x)
        x = self.relu(x)
        # print("x.shape 2: ", x.shape)
        x = self.pool2(x)
        # print("x.shape 2: ", x.shape)

        x = self.conv3(x)
        x = self.relu(x)
        # print("x.shape 3: ", x.shape)
        x = self.pool3(x)
        # print("x.shape 3: ", x.shape)
        #
        x = self.conv4(x)
        x = self.relu(x)
        # print("x.shape 4: ", x.shape)
        # x = self.pool4(x)
        # print("x.shape 4: ", x.shape)
        # exit()
        # x = self.conv5(x)
        # x = self.relu(x)
        # # print("x.shape 5: ", x.shape)
        # x = self.pool5(x)
        # # print("x.shape 5: ", x.shape)
        # # exit()
        #
        # x = self.pool6(x)
        # print("x.shape 6: ", x.shape)
        # exit()
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

# use_path_train = "E:\\Etude technique\\raw\\train"
# use_path_test = "E:\\Etude technique\\raw\\test"
# use_path_model = "E:\\Etude technique\\model"


use_path_train = "D:\\Etude technique\\train"
use_path_test = "D:\\Etude technique\\test"
use_path_model = "D:\\Etude technique\\model.pth"


def train_model(train_path, verbose=False, show_result=True, epochs=20, batch_size=12):
    df_path_train = load(train_path)
    df_train, df_valid = train_test_split(df_path_train, test_size=0.2)
    train_set = CustomDataset(df_path_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    val_set = CustomDataset(df_valid)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    if verbose:
        print(model)

    all_losses = []
    all_accuracy = []
    all_val_losses = []
    all_val_accuracy = []

    for epoch in range(epochs):
        model.train()
        losses, corrects, val_losses, val_corrects = [], 0, [], 0

        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.permute(0, 3, 1, 2) / 64

            x = x.to(device).double()
            y = y.to(device).double()

            output = model(x)
            correct, total, nb_0, nb_1 = 0, 0, 0, 0

            for k in range(len(output)):
                if float(output[k][0].item()) > float(output[k][1].item()):
                    if int(y[k][0].item()) == 1:
                        correct += 1
                        corrects += 1
                elif float(output[k][1].item()) >= float(output[k][0].item()):
                    if int(y[k][1].item()) == 1:
                        correct += 1
                        corrects += 1
                total += 1

            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if verbose:
                if batch_num % 5 == 0:
                    print('\tEpoch %d | Batch %d | Loss %6.2f | Accuracy %6.2f' % (
                          epoch, batch_num, loss.item(), correct / total))

        for batch_num, input_data in enumerate(val_loader):
            x, y = input_data
            x = x.permute(0, 3, 1, 2) / 64

            x = x.to(device).double()
            y = y.to(device).double()

            output = model(x)

            for k in range(len(output)):
                if float(output[k][0].item()) > float(output[k][1].item()):
                    if int(y[k][0].item()) == 1:
                        val_corrects += 1
                elif float(output[k][1].item()) >= float(output[k][0].item()):
                    if int(y[k][1].item()) == 1:
                        val_corrects += 1

            loss = criterion(output, y)
            val_losses.append(loss.item())

        print('Epoch %d | Loss %6.2f | Accuracy %6.2f | Val_Loss %6.2f | Val_Accuracy %6.2f' % (
              epoch, sum(losses) / len(losses), corrects / len(train_set), sum(val_losses) / len(val_losses), val_corrects / len(val_set)))

        all_losses.append(sum(losses) / len(losses))
        all_accuracy.append(corrects / len(train_set))
        all_val_losses.append(sum(val_losses) / len(val_losses))
        all_val_accuracy.append(val_corrects / len(val_set))

    if show_result:
        plt.plot(range(len(all_val_losses)), all_val_losses)
        plt.plot(range(len(all_val_losses)), all_val_losses)
        plt.show()
        plt.plot(range(len(all_val_accuracy)), all_val_accuracy)
        plt.plot(range(len(all_val_accuracy)), all_val_accuracy)
        plt.show()

    return model


def test_model(model_, test_path, verbose=False, batch_size=12):
    df_path_test = load(test_path)
    test_set = CustomDataset(df_path_test)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_.eval()
    corrects, totals = 0, 0

    for batch_num, input_data in enumerate(testloader):
        x, y = input_data
        x = x.permute(0, 3, 1, 2) / 64

        x = x.to(device).double()
        y = y.to(device).double()

        output = model_(x)

        correct, total = 0, 0

        for k in range(len(output)):
            if float(output[k][0].item()) > float(output[k][1].item()):
                if int(y[k][0].item()) == 1:
                    correct += 1
                    corrects += 1
            elif float(output[k][1].item()) >= float(output[k][0].item()):
                if int(y[k][1].item()) == 1:
                    correct += 1
                    corrects += 1
            total += 1
            totals += 1

    print('Test Accuracy %6.2f' % (corrects / totals))


def save_model(model_, save_path):
    torch.save(model_, save_path)


def load_model(load_path):
    model_ = torch.load(load_path)
    return model_

# model = train_model(use_path_train, use_path_test)
# test_model(model, use_path_test)
#
# save_model(model, use_path_model)
# print('Done')
