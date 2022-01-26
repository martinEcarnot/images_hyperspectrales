import numpy as np
import pandas as pd
import os
from os import walk
import spectral as sp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report


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
        class_int = int(self.labels[idx])
        label = 0 if class_int == 1 else 1  # Can be improve for random couple of classes
        # path_img = self.paths_img[idx]
        path_hdr = self.paths_hdr[idx]
        img = sp.open_image(path_hdr)
        img_tensor = torch.tensor(img[:, :, :])

        return [img_tensor, torch.tensor(label)], torch.tensor(label)


class CNN(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax

        self.conv1_0 = nn.Conv2d(in_channels=dim_in, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(7, 7), stride=(3, 3))

        self.conv2_1 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(5, 5))

        self.conv3_1 = nn.Conv2d(in_channels=80, out_channels=120, kernel_size=(3, 3))
        self.conv3_2 = nn.Conv2d(in_channels=120, out_channels=160, kernel_size=(3, 3), padding=(1, 1))

        self.conv4_1 = nn.Conv2d(in_channels=160, out_channels=240, kernel_size=(3, 3))
        self.conv4_2 = nn.Conv2d(in_channels=240, out_channels=320, kernel_size=(3, 3), padding=(1, 1))

        self.conv5_1 = nn.Conv2d(in_channels=320, out_channels=400, kernel_size=(3, 3))
        self.conv5_2 = nn.Conv2d(in_channels=400, out_channels=500, kernel_size=(3, 3), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(in_channels=500, out_channels=640, kernel_size=(3, 3), padding=(1, 1))

        self.conv6_1 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=(3, 3))

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1 * 1 * 320, 30)
        self.dropout = nn.Dropout(0.2)
        # self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(31, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        image, variete = input_data[0], input_data[1]
        image = image.permute(0, 3, 1, 2).float()

        x = self.relu(self.conv1_0(image))
        x = self.pool(self.relu(self.conv1_1(x)))
        # print("x.shape 1: ", x.shape)

        x = self.pool(self.relu(self.conv2_1(x)))

        x = self.relu(self.conv3_1(x))
        x = self.pool(self.relu(self.conv3_2(x)))
        #
        x = self.relu(self.conv4_1(x))
        x = self.pool(self.relu(self.conv4_2(x)))
        x = self.pool(x)
        # print("x.shape 4: ", x.shape)
        # exit()

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = torch.cat((variete[..., None], x), -1)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def train_model(train_loader, val_loader, model, loss_fn, optimizer, verbose=False):
    if verbose:
        print(model)

    train_loss, correct = 0, 0
    size_train = len(train_loader.dataset)
    size_valid = len(val_loader.dataset)
    num_batches_train = len(train_loader)
    num_batches_valid = len(val_loader)

    # List to stock y and y pred to print a classification report
    list_y_pred = []
    list_y = []

    for batch_num, (image, labels) in enumerate(train_loader):
        model.train()

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image, labels = image.cuda(), labels.cuda()

        # Compute prediction error
        output = model(image)
        list_y_pred += output.argmax(1).tolist()
        list_y += labels.tolist()
        loss = loss_fn(output, labels)
        train_loss += loss
        correct += (output.argmax(1) == labels).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            if batch_num % 5 == 0:
                loss, current = loss.item(), batch_num * len(image)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size_train:>5d}]")

    # Determination of other metrics (they are just print but can be retrieve as a dictionnary if necessary)
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y_pred)))

    # Validation
    valid_loss, correct_valid = 0, 0
    list_y_pred_val = []
    list_y_val = []
    model.eval()
    for image_val, label_val in val_loader:

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image_val, label_val = image_val.cuda(), label_val.cuda()

        # Forward pass validation
        target = model(image_val)
        # Classification report
        list_y_pred_val += target.argmax(1).tolist()
        list_y_val += label_val.tolist()
        # Loss validation
        loss = loss_fn(target, label_val)
        # Calculate Loss
        valid_loss += loss.item()
        correct_valid += (target.argmax(1) == label_val).type(torch.float).sum().item()

    print(classification_report(list_y_val, list_y_pred_val, labels=np.unique(list_y_pred_val)))

    train_loss /= num_batches_train
    valid_loss /= num_batches_valid
    correct /= size_train
    correct_valid /= size_valid
    print(f"Results : \t Accuracy Train: {(100 * correct):>0.1f}%, Avg loss Train: {train_loss:>8f} \t Accuracy "
          f"Validation: {(100 * correct_valid):>0.1f}%, Avg loss Validation: {valid_loss:>8f}")

    return model


def test_model(test_loader, model, loss_fn):
    """
    Apply the trained model to test dataset

    :param test_loader: test dataloader
    :param model: model to use (ANN)
    :param loss_fn: the loss for training
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    list_y_pred = []
    list_y = []
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            output = model(images)
            list_y_pred += output.argmax(1).tolist()
            list_y += labels.tolist()
            test_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test : \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Determination of other metrics
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y_pred)))


def save_model(model_, save_path):
    torch.save(model_, save_path)


def load_model(load_path):
    model_ = torch.load(load_path)
    return model_


def main_loop(train_path, valid_path, test_path, model, loss_fn, optimizer, epochs=20, batch_size=12):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param train_path:
    :param valid_path:
    :param test_path:
    :param model: model to train
    :param loss_fn: loss used
    :param optimizer: optimizer for the gradient
    :param epochs: number of epochs used for training
    :param batch_size:
    :return:
    """

    if torch.cuda.is_available():
        model = model.cuda()

    df_path_train = load(train_path)
    train_set = CustomDataset(df_path_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_path_valid = load(valid_path)
    val_set = CustomDataset(df_path_valid)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_path_test = load(test_path)
    test_set = CustomDataset(df_path_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    print('training model')
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_model(train_loader, val_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)

    print("Saving model at ", use_path_model)
    save_model(model, use_path_model)

    print("Testing model")
    test_model(test_loader, model=model, loss_fn=loss_fn)

    print("Done!")


use_path_train = "E:\\Etude technique\\raw\\train"
use_path_test = "E:\\Etude technique\\raw\\test"
use_path_valid = "E:\\Etude technique\\raw\\valid"
use_path_model = "E:\\Etude technique\\model.pth"
# use_path_train = "D:\\Etude technique\\train"
# use_path_test = "D:\\Etude technique\\test"
# use_path_valid = "D:\\Etude technique\\valid"
# use_path_model = "D:\\Etude technique\\model.pth"

learning_rate = 1e-4

# Detection number of bands
path_band = os.path.join(use_path_train, os.listdir(use_path_train)[0])
file_band = [x for x in os.listdir(path_band) if "hdr" in x][0]
image_band = sp.open_image(os.path.join(path_band, file_band))
dim_in = image_band.shape[2]
# dim_in = 10

model = CNN(dim_in)  # .double()
# weight = torch.tensor().cuda() if torch.cuda.is_available() else torch.tensor([4., 2.])
loss_fn = nn.CrossEntropyLoss()#weight=weight)
# loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
epochs = 10
main_loop(use_path_train, use_path_valid, use_path_test, model, loss_fn, optimizer, epochs=epochs, batch_size=12)
