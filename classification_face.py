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
import csv
from torchinfo import summary
from cnns import *

class CustomDataset(Dataset):
    """
    Create dataset for the network
    """

    def __init__(self, df_annot, annot_dir, labels_type):
        """
        Initialisation of the dataset
        :param df_annot: dataframe containing at least the columns "Name_hdr" and "Face"/"Species"
        :annot_dir: path of the directory containing the annotations
        :labels_type: name of the column containing the labels, here "Face" or "Species"
        """
        super().__init__()
        self.names_hdr = df_annot["Name_hdr"]
        self.annot_dir = annot_dir
        self.labels = df_annot[labels_type]
        
    def __len__(self):
        """
        Function used by the Dataset to retrieve the size
        :return: len of the dataframe
        """
        return len(self.names_hdr)

    def __getitem__(self, idx):
        """
        Creation of the tensor and the label
        :param idx: indice to select the path in the list
        :return: tensor, label
        """
        img = sp.open_image(self.annot_dir+self.names_hdr[idx])
        img_tensor = torch.tensor(img[:, :, :])
        return img_tensor, torch.tensor(self.labels[idx])


def train_model(train_loader, val_loader, device, model, loss_f, optimizer, verbose=False):
    """
    Loop to train the deep learning model.

    :param train_loader: Dataloader of the training dataset
    :param val_loader: Dataloader of the validation dataset
    :param device: cpu or cuda
    :param model: the CNN model
    :param loss_f: the loss to consider
    :param optimizer: optimizer (Adam)
    :param verbose: Set to True to display the model parameters and information during training
    :return: The trained model, accuracy training, accuracy validation, train_loss, valid_loss
    """
    
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
        image, labels = image.to(device), labels.to(device)

        image = image.permute(0, 3, 1, 2).float()
        # Compute prediction error
        output = model(image)
        list_y_pred += output.argmax(1).tolist()
        list_y += labels.tolist()
        loss = loss_f(output, labels)
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
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y)))

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
        loss = loss_f(target, label_val)
        # Calculate Loss
        valid_loss += loss.item()
        correct_valid += (target.argmax(1) == label_val).type(torch.float).sum().item()
    print(classification_report(list_y_val, list_y_pred_val, labels=np.unique(list_y_val)))

    train_loss /= num_batches_train
    valid_loss /= num_batches_valid
    correct /= size_train
    correct_valid /= size_valid
    print(f"Results : \t Accuracy Train: {(100 * correct):>0.1f}% \t Avg loss Train: {train_loss:>8f} \t Accuracy "
          f"Validation: {(100 * correct_valid):>0.1f}% \t Avg loss Validation: {valid_loss:>8f}")

    return model, correct, correct_valid, train_loss, valid_loss


def test_model(test_loader, device, model, loss_f, test_dir, model_fn, test_name = 'test_set', other_class = False):

    """
    Apply the trained model to test dataset

    :param test_loader: test dataloader
    :param device: cpu or cuda
    :param model: model to use (CNN)
    :param loss_f: the loss to consider
    :return: Accuracy and loss of the test
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    list_y_probas = []
    list_y_pred = []
    list_y = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 3, 1, 2).float()
            output = model(images)
            tmp = output.tolist()
            list_y_probas += [tmp[i] for i in range(images.size(dim=0))]
            list_y_pred += output.argmax(1).tolist()
            list_y += labels.tolist()
            test_loss += loss_f(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    df_test = pd.read_csv(test_dir + test_name + '.csv')
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
    df_test['Probas'] = list_y_probas
    df_test['Face_pred'] = list_y_pred
    df_test.to_csv('models/' + model_fn + '/test_preds.csv', index = False)
    print(f"Test : \n Accuracy: {(100 * correct):>0.1f}% \t Avg loss: {test_loss:>8f} \n")
    # Determination of other metrics
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y)))
    return 100*correct, test_loss


def summary_training(model,annot_dir,labels_type,weights_loss, learning_rate, epochs, batch_size, other_class, bands):
    n_classes=0
    if labels_type=='Face':
        if other_class:
            n_classes=3
        else: 
            n_classes=2
    else:
        n_classes=8
    struct = summary(model,(batch_size,len(bands),200,200))

    res ='''TASK
Directory used for annotations : {}
Classification type : {}
Number of classes : {}
Bands used : {}
Number of bands : {}


MODEL'S HYPERPARAMETERS
Structure : 
{}
Number of epochs : {}
Learning rate : {}
Batch size : {}
Classes' weights in the loss : {}'''.format(
        annot_dir,
        labels_type,
        n_classes,
        bands,
        len(bands),
        struct,
        epochs,
        learning_rate,
        batch_size, 
        weights_loss
    )
    return res

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


def display_save_figure(fig_dir,fig_fn, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid):
    """
    After the training is done, the results are displayed and saved

    :param fig_dir: path to save the figure and the data
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

    fig.suptitle(fig_fn)  # Global title
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(fig_dir, fig_fn+".png"), dpi=200, format='png')

def model_testing(model_fn, annot_dir, annot_path = 'test_set', other_face = False):
    df_test = pd.read_csv(annot_dir + annot_path + '.csv')
    if not other_face :
        df_test = df_test.loc[df_test['Face']!=2]
        df_test.index = [i for i in range(len(df_test))] 
    test_set = CustomDataset(df_test, annot_dir, labels_type = 'Face')
    test_loader = DataLoader(test_set, batch_size=12, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join('models', model_fn, model_fn + '.pth'))
    weight_loss = [2., 2.]
    weight = torch.tensor(weight_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    test_model(test_loader, device, model=model, loss_f=loss_f, test_dir = annot_dir, model_fn = model_fn)
    
    
    

def main_loop(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=20, batch_size=12, other_class = False):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param cnn: class of the CNN to be used.
    :param annot_dir: path to the directory of the annotations files
    :labels_type: name of the column containing the labels, here "Face" or "Species"
    :param weights_loss: the weights to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    """
    bands = []
    with open(annot_dir + 'bands.txt', "r") as f:
        first_line = f.readline()
        second_line = f.readline().split(': ')[1]
        if second_line[:3] == 'RGB' :
            bands = [22, 53, 89]
        elif second_line[:3] == 'All':
            bands = [i for i in range(216)]
    dim_in=len(bands)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn(dim_in).to(device)

    weight = torch.tensor(weights_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    df_train = pd.read_csv(annot_dir + 'train_set.csv')
    if not other_class :
        df_train = df_train.loc[df_train['Face']!=2]
        df_train.index = [i for i in range(len(df_train))]

    train_set = CustomDataset(df_train, annot_dir, labels_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_valid = pd.read_csv(annot_dir + 'validation_set.csv')
    if not other_class :
        df_valid = df_valid.loc[df_valid['Face']!=2]
        df_valid.index = [i for i in range(len(df_valid))]

    val_set = CustomDataset(df_valid, annot_dir, labels_type)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_test = pd.read_csv(annot_dir + 'test_set.csv')
    
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
        df_test.index = [i for i in range(len(df_test))]
    
    test_set = CustomDataset(df_test, annot_dir, labels_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Nubmer of trainables parameters :" +str(pytorch_total_params))
    print('\nTraining model')
    list_accu_train = []
    list_accu_valid = []
    list_loss_train = []
    list_loss_valid = []
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        model, correct, correct_valid, train_loss, valid_loss = train_model(train_loader, val_loader, device,
                                                                            model=model, loss_f=loss_f, optimizer=optimizer)
        list_accu_train.append(correct*100)
        list_accu_valid.append(correct_valid*100)
        list_loss_train.append(train_loss.item())  # Apparently tensor
        list_loss_valid.append(valid_loss)
        
    model_dir = os.path.join("models", model_fn)
    if not(os.path.exists(model_dir)):
        os.mkdir(model_dir)
    model_path = os.path.join("models",model_fn,model_fn + ".pth")
    print("\nSaving model at ", model_path)
    save_model(model, model_path)
    
    recap_path = os.path.join("models",model_fn,model_fn+"_summary.txt")
    with open(recap_path,'w') as recap_file:
        recap_file.write(summary_training(
            model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands = bands))
    fig_fn = model_fn+"_training_evolution"
    display_save_figure(model_dir, fig_fn, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid)

    print("\nTesting model")
    test_accu, test_loss = test_model(test_loader, device, model=model, loss_f=loss_f, test_dir = annot_dir, model_fn = model_fn)

    # Saving values
    print("\nSaving values of train, validation and test loops")
    df_res_train_val = pd.DataFrame({"Train accuracy":list_accu_train,"Validation accuracy":list_accu_valid,"Train loss":list_loss_train,"Validation loss":list_loss_valid})
    df_res_train_val.to_csv(os.path.join(model_dir, model_fn +"_values_train_valid.csv"),index=False)
    df_res_test = pd.DataFrame({"Test accuracy":test_accu,"Test loss":test_loss},index=[0])
    df_res_test.to_csv(os.path.join(model_dir, model_fn +"_values_test.csv"),index=False)


    print("\nDone!")
    
    
def main_loop2(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=20, batch_size=12, other_class = False):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param cnn: class of the CNN to be used.
    :param annot_dir: path to the directory of the annotations files
    :labels_type: name of the column containing the labels, here "Face" or "Species"
    :param weights_loss: the weights to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    """
    bands = []
    with open(annot_dir + 'bands.txt', "r") as f:
        first_line = f.readline()
        second_line = f.readline().split(': ')[1]
        if second_line[:3] == 'RGB' :
            bands = [22, 53, 89]
        elif second_line[:3] == 'All':
            bands = [i for i in range(216)]
    dim_in=len(bands)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn(dim_in).to(device)

    weight = torch.tensor(weights_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    df_train = pd.read_csv(annot_dir + 'train_set.csv')
    if not other_class :
        df_train = df_train.loc[df_train['Face']!=2]
        df_train.index = [i for i in range(len(df_train))]

    train_set = CustomDataset(df_train, annot_dir, labels_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_valid = pd.read_csv(annot_dir + 'validation_set.csv')
    if not other_class :
        df_valid = df_valid.loc[df_valid['Face']!=2]
        df_valid.index = [i for i in range(len(df_valid))]

    val_set = CustomDataset(df_valid, annot_dir, labels_type)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_test = pd.read_csv(annot_dir + 'test_set.csv')
    
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
        df_test.index = [i for i in range(len(df_test))]
    
    test_set = CustomDataset(df_test, annot_dir, labels_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Nubmer of trainables parameters :" +str(pytorch_total_params))
    print('\nTraining model')
    list_accu_train = []
    list_accu_valid = []
    list_loss_train = []
    list_loss_valid = []
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        model, correct, correct_valid, train_loss, valid_loss = train_model(train_loader, val_loader, device,
                                                                            model=model, loss_f=loss_f, optimizer=optimizer)
        list_accu_train.append(correct*100)
        list_accu_valid.append(correct_valid*100)
        list_loss_train.append(train_loss.item())  # Apparently tensor
        list_loss_valid.append(valid_loss)
        
    model_dir = os.path.join("models", model_fn)
    if not(os.path.exists(model_dir)):
        os.mkdir(model_dir)
    model_path = os.path.join("models",model_fn,model_fn + ".pth")
    print("\nSaving model at ", model_path)
    save_model(model, model_path)
    
    recap_path = os.path.join("models",model_fn,model_fn+"_summary.txt")
    with open(recap_path,'w') as recap_file:
        recap_file.write(summary_training(
            model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands = bands))
    fig_fn = model_fn+"_training_evolution"

    print("\nTesting model")
    test_accu, test_loss = test_model(test_loader, device, model=model, loss_f=loss_f, test_dir = annot_dir, model_fn = model_fn)

    # Saving values
    print("\nSaving values of train, validation and test loops")
    df_res_train_val = pd.DataFrame({"Train accuracy":list_accu_train,"Validation accuracy":list_accu_valid,"Train loss":list_loss_train,"Validation loss":list_loss_valid})
    df_res_train_val.to_csv(os.path.join(model_dir, model_fn +"_values_train_valid.csv"),index=False)
    df_res_test = pd.DataFrame({"Test accuracy":test_accu,"Test loss":test_loss},index=[0])
    df_res_test.to_csv(os.path.join(model_dir, model_fn +"_values_test.csv"),index=False)


    print("\nDone!")

def cross_validation(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=20, batch_size=12, other_class = False):
    for k in range(n_folds):
        main_loop(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=20, batch_size=12, other_class = False)
