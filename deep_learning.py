# Global import
import pandas as pd
import os
from os import walk
import spectral as sp
import torch
from torch.utils.data import Dataset


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
        # path_img = self.paths_img[idx]
        path_hdr = self.paths_hdr[idx]
        img = sp.open_image(path_hdr)
        img_tensor = torch.tensor(img[:, :, :])

        return img_tensor, label


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax

        # 55, 180, 1 -> #55, 60, 30
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(3, 9), stride=(1, 3), padding=(1, 4))
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=(1, 0))
        # 56, 60, 16 -> #28, 30, 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 26, 28, 32 -> #13, 14, 32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=(1, 0))
        # 11, 12, 64 -> #6, 6, 64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 6), stride=1)
        # 1, 1, 128 -> #1, 1, 128
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 30)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(30, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # print("x.shape 1: ", input_data.shape)
        x = self.conv1(input_data)
        x = self.tanh(x)
        # print("x.shape 2: ", x.shape)
        x = self.pool1(x)

        # print("x.shape: ", x.shape)
        x = self.conv2(x)
        x = self.tanh(x)
        # print("x.shape: ", x.shape)
        x = self.pool2(x)
        # print("x.shape: ", x.shape)
        x = self.conv3(x)
        x = self.tanh(x)
        # print("x.shape: ", x.shape)
        x = self.pool3(x)
        # print("x.shape: ", x.shape)
        x = self.conv4(x)
        x = self.tanh(x)
        # print("x.shape: ", x.shape)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
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

# test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
model = model.double()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss(torch.tensor([1, 4]))

print(model)

for epoch in range(epochs):
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch_num, input_data in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = input_data
        # print(y.shape)

        # y = y.reshape((len(x), 2))
        # print(x.shape, y.shape)

        x = x.to(device).double()
        y = y.to(device).double()
        # print('y', y)

        output = model(x)
        s = nn.Softmax()
        pred = s(output)
        for k in range(len(pred)):
            if pred[k][0] > 0.5 and y[k][0] == 1:
                correct += 1
            elif pred[k][0] < 0.5 and y[k][1] == 1:
                correct += 1
            total += 1
        # print('output', s(output))
        # print('y', y)
        loss = criterion(output, y)
        # print(loss)
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

    print('Validation Epoch %d | Val_Loss %6.2f | Val_Accuracy %6.2f' % (
    epoch, sum(losses) / len(losses), correct / total))


