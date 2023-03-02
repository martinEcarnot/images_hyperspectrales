import numpy as np
import pandas as pd
import os
from os import walk
import spectral as sp
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
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
        """
        Function used by the Dataset to retrieve the size
        :return: len of the dataframe
        """
        return len(self.paths_hdr)

    def __getitem__(self, idx):
        """
        Creation of the tensor and the label
        :param idx: indice to select the path in the list
        :return: tensor, label
        """
        class_int = int(self.labels[idx])
        label = 0 if class_int == 1 else 1  # Can be improve for random couple of classes
        path_hdr = self.paths_hdr[idx]
        img = sp.open_image(path_hdr)
        img_tensor = torch.tensor(img[:, :, :])
       
        return img_tensor, torch.tensor(label)


class CNN(nn.Module):
    """
    Creation of the neural network
    """

    def __init__(self, dim_in):
        """
        Initialisation of the layers
        :param dim_in: dimension of the input image
        """
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
        self.linear1 = nn.Linear(2 * 2 * 320, 30)
        self.dropout = nn.Dropout(0.2)
        # self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(30, 2)
        # For multiple input: 30 + number of input (1 here)
        
        self.softmax = nn.Softmax(dim=1)
        print()
    def forward(self, input_data):
        """
        Order of the layers
        :param input_data: Input image
        :return: a tensor of size (1, 2) (Softmax)
        """
        # If multiple input:
        
        x = self.relu(self.conv1_0(input_data))
        x = self.pool(self.relu(self.conv1_1(x)))
        # print("x.shape 1: ", x.shape)

        x = self.pool(self.relu(self.conv2_1(x)))
        x = self.relu(self.conv3_1(x))
        x = self.pool(self.relu(self.conv3_2(x)))
        x = self.relu(self.conv4_1(x))
        x = self.pool(self.relu(self.conv4_2(x)))
        

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        # Add another input
        
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def train_model(train_loader, val_loader, device, model, loss_fn, optimizer, verbose=False):
    """
    Loop to train the deep learning model.

    :param train_loader: Dataloader of the training dataset
    :param val_loader: Dataloader of the validation dataset
    :param device: cpu or cuda
    :param model: the CNN model
    :param loss_fn: the loss to consider
    :param optimizer: optimizer (Adam)
    :param verbose: Set to True to display the model parameters and information during training
    :return: The trained model, accuracy training, accuracy validation, train_loss, valid_loss
    """
    if verbose:
        # from torchsummary import summary
        print(model)
        # summary(model, input_size=(21, 200, 200))  # 21 = nb_bands
        
    train_loss, correct = 0, 0
    size_train = len(train_loader.dataset)
    size_valid = len(val_loader.dataset)
    num_batches_train = len(train_loader)
    num_batches_valid = len(val_loader)

    # List to stock y and y pred to print a classification report
    list_y_pred = []
    list_y = []
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


    for batch_num, (image, labels) in enumerate(train_loader):
        model.train()

        # Transfer Data to GPU if available
        image, labels = image.to(device), labels.to(device)

        image = image.permute(0, 3, 1, 2).float()

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
        image_val, label_val = image_val.to(device), label_val.to(device)

        image_val = image_val.permute(0, 3, 1, 2).float()

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
    print(f"Results : \t Accuracy Train: {(100 * correct):>0.1f}% \t Avg loss Train: {train_loss:>8f} \t Accuracy "
          f"Validation: {(100 * correct_valid):>0.1f}% \t Avg loss Validation: {valid_loss:>8f}")

    return model, correct, correct_valid, train_loss, valid_loss


def test_model(test_loader, device, model, loss_fn):
    """
    Apply the trained model to test dataset

    :param test_loader: test dataloader
    :param device: cpu or cuda
    :param model: model to use (CNN)
    :param loss_fn: the loss to consider
    :return: Accuracy and loss of the test
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    list_y_pred = []
    list_y = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 3, 1, 2).float()
            output = model(images)
            list_y_pred += output.argmax(1).tolist()
            list_y += labels.tolist()
            test_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test : \n Accuracy: {(100 * correct):>0.1f}% \t Avg loss: {test_loss:>8f} \n")

    # Determination of other metrics
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y_pred)))
    return 100*correct, test_loss


def save_model(model_, save_path):
    """
    Save the model to a given path
    :param model_: Model to save
    :param save_path: Path to use
    """
    torch.save(model_, save_path)


def load_model(load_path):
    """
    Load a saved model
    :param load_path: location of the model
    :return: the model
    """
    model_ = torch.load(load_path)
    return model_


def display_save_figure(figure_path, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid, name_figure):
    """
    After the training is done, the results are displayed and saved

    :param figure_path: path to save the figure and the data
    :param list_accu_train: List of all accuracy during training
    :param list_accu_valid: List of all accuracy during validation
    :param list_loss_train: List of all loss during training
    :param list_loss_valid: List of all loss during validation
    :param name_figure: name to give to the figure
    """
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7))
    ax = axes.ravel()
    list_nb = range(len(list_accu_train))
    ax[0].plot(list_nb, list_accu_train, label='Train')
    for a, b in zip(list_nb, list_accu_train):
        ax[0].text(a, b, str(round(b, 1)))
    ax[0].plot(list_nb, list_accu_valid, label='Valid')
    for a, b in zip(list_nb, list_accu_valid):
        ax[0].text(a, b, str(round(b, 1)))
    ax[0].set_title('Accuracy given the epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(list_nb, list_loss_train, label='Train')
    for a, b in zip(list_nb, list_loss_train):
        ax[1].text(a, b, str(round(b, 2)))
    ax[1].plot(list_nb, list_loss_valid, label='Valid')
    for a, b in zip(list_nb, list_loss_valid):
        ax[1].text(a, b, str(round(b, 2)))
    ax[1].set_title('Loss given the epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid()

    fig.suptitle(name_figure)  # Global title
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(figure_path, name_figure+".png"), dpi=200, format='png')


def main_loop(use_path, weight_loss, learning_rate, epochs=20, batch_size=12):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param use_path: Path where the folder train, valid and test are located
    :param weight_loss: the weight to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    """
    # Detection number of bands
    use_path_train = os.path.join(use_path, "train", "")
    path_band = os.path.join(use_path_train, os.listdir(use_path_train)[0])
    file_band = [x for x in os.listdir(path_band) if "hdr" in x][0]
    image_band = sp.open_image(os.path.join(path_band, file_band))
    dim_in = image_band.shape[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(dim_in).to(device)

    weight = torch.tensor(weight_loss).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    name_file = input("Enter the name of the figure to save (it should contain which images are "
                      "selected to do the training, validation and testing): ")

    train_path = os.path.join(use_path, "train", "")
    valid_path = os.path.join(use_path, "valid", "")
    test_path = os.path.join(use_path, "test", "")

    df_path_train = load(train_path)
    print(df_path_train.head())
    train_set = CustomDataset(df_path_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_path_valid = load(valid_path)
    print(df_path_valid.tail())
    val_set = CustomDataset(df_path_valid)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_path_test = load(test_path)
    print(df_path_test.tail())
    test_set = CustomDataset(df_path_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    print('\nTraining model')
    list_accu_train = []
    list_accu_valid = []
    list_loss_train = []
    list_loss_valid = []
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        model, correct, correct_valid, train_loss, valid_loss = train_model(train_loader, val_loader, device, model=model,
                                                                            loss_fn=loss_fn, optimizer=optimizer)
        list_accu_train.append(correct*100)
        list_accu_valid.append(correct_valid*100)
        list_loss_train.append(train_loss.item())  # Apparently tensor
        list_loss_valid.append(valid_loss)

    use_path_model = os.path.join(use_path, "model_" + name_file + ".pth")
    print("\nSaving model at ", use_path_model)
    save_model(model, use_path_model)

    print("\nDisplay graphs of accuracy and loss and save figure at ", os.path.join(use_path, name_file+'.png'))
    display_save_figure(use_path, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid, name_file)

    print("\nTesting model")
    test_accu, test_loss = test_model(test_loader, device, model=model, loss_fn=loss_fn)

    # Saving values
    print("\nSaving values of train, validation and test loops")
    save_array = np.asarray([list_accu_train, list_accu_valid, list_loss_train, list_loss_valid])
    np.savetxt(os.path.join(use_path, name_file+"_values_train_valid.csv"), save_array,
               delimiter=",", fmt='%.5e')  # Train
    np.savetxt(os.path.join(use_path, name_file+"_values_test.csv"), np.asarray([[test_accu], [test_loss]]),
               delimiter=",", fmt='%.5e')  # Test

    print("\nDone!")


