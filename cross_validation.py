import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from cnns import *
from customdataset import CustomDataset
from utils import *


def train_model(train_loader, device, model, loss_f, optimizer, verbose=False):
    """
    Loop to train the deep learning model.

    :param train_loader: Dataloader of the training dataset
    :param device: cpu or cuda
    :param model: the CNN model
    :param loss_f: the loss to consider
    :param optimizer: optimizer (Adam)
    :param verbose: Set to True to display the model parameters and information during training
    :return: The trained model, accuracy training
    """

    train_loss, correct = 0, 0
    size_train = len(train_loader.dataset)
    num_batches_train = len(train_loader)

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
    correct /= size_train
    train_loss /= num_batches_train
    # Determination of other metrics (they are just print but can be retrieve as a dictionnary if necessary)
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y)))
    return model, correct, train_loss


def test_model(test_loader, device, model, loss_f, model_dir, model_fn, test_name='test_set', other_class=False):
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
            correct += (output.argmax(1) ==
                        labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(
        f"Test : \n Accuracy: {(100 * correct):>0.1f}% \t Avg loss: {test_loss:>8f} \n")
    # Determination of other metrics

    return correct, test_loss, list_y_pred, list_y


def summary_training(model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands):
    n_classes = 0
    if labels_type == 'Face':
        if other_class:
            n_classes = 3
        else:
            n_classes = 2
    else:
        n_classes = 8
    struct = summary(model, (batch_size, len(bands), 200, 200))

    res = '''TASK
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


def fold_loop(model, model_dir, model_fn, weights_loss, train_loader, test_loader, learning_rate=1e-4, epochs=80):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param cnn: class of the CNN to be used.
    :labels_type: name of the column containing the labels, here "Face" or "Species"
    :param weights_loss: the weights to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    weight = torch.tensor(weights_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Nubmer of trainables parameters :" + str(pytorch_total_params))
    print('\nTraining model')
    list_accu_train = []
    list_loss_train = []
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        model, correct, train_loss = train_model(
            train_loader, device, model=model, loss_f=loss_f, optimizer=optimizer)
        list_accu_train.append(correct)
        list_loss_train.append(train_loss.item())

    print("\nTesting model")
    test_accu, test_loss, list_y_pred, list_y = test_model(
        test_loader, device, model=model, loss_f=loss_f, model_dir=model_dir, model_fn=model_fn)

    report = classification_report(
        list_y, list_y_pred, labels=np.unique(list_y))
    df_res_train = pd.DataFrame(
        {"Train accuracy": list_accu_train, "Train loss": list_loss_train})
    df_res_test = pd.DataFrame(
        {"Test accuracy": test_accu, "Test loss": test_loss}, index=[0])
    return df_res_train, df_res_test, report


def display_save_average_metrics(fig_dir, fig_fn):
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


def k_folds(df, K):
    N = len(df)
    K = 5
    q = N//K
    r = N % K
    ens = []
    deb = 0
    for j in range(K):
        fin = deb + q
        if j < r:
            fin += 1
        ens.append([deb, fin])
        deb = fin
    return ens


def cross_validation(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate=1e-4, epochs=80,
                     batch_size=12, other_class=False, K=5, chosen_species = [], chosen_face = []):
    model_dir = os.path.join("models", model_fn)
    if not(os.path.exists(model_dir)):
        os.mkdir(model_dir)
    shuffle_full_set("full_set", model_dir)
    df_full = pd.read_csv(os.path.join(model_dir, "full_set"+".csv"))
    
    n_classes = 3
    bands = []
    with open(annot_dir + 'bands.txt', "r") as f:
        first_line = f.readline()
        second_line = f.readline().split(': ')[1]
        if second_line[:3] == 'RGB':
            bands = [22, 53, 89]
        elif second_line[:3] == 'All':
            bands = [i for i in range(216)]
    dim_in = len(bands)

    if not other_class:
        df_full = df_full.loc[df_full['Face'] != 2]
        df_full.index = [i for i in range(len(df_full))]
        df_full.to_csv(os.path.join(model_dir, "full_set.csv"), index=False)
        n_classes = 2
        
    if labels_type == 'Species' :
        if len(chosen_species)==0:
            chosen_species = [i for i in range(8)]
        print("Variétés que l'on va chercher à différencier : " + str(chosen_species)[1:-1])
        df_full = df_full.loc[df_full['Species'].isin(chosen_species)]
        if chosen_face == 'Dos' :
            df_full = df_full.loc[df_full['Face']==0]
        elif chosen_face == 'Sillon' :
            df_full = df_full.loc[df_train['Face']==1]
        n_classes = len(chosen_species)
    df_full.index = [i for i in range(len(df_full))]
    
    model = cnn(dim_in, n_classes)

    recap_path = os.path.join("models", model_fn, model_fn+"_summary.txt")
    with open(recap_path, 'w', encoding='utf-8') as recap_file:
        recap_file.write(summary_training(
            model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands=bands))
        
    partition_df = k_folds(df_full, K)
    k = 0
    df_res_trains = pd.DataFrame()
    df_res_tests = pd.DataFrame()
    for e in partition_df:
        df_train = df_full.iloc[[i < e[0] or i > e[1] for i in df_full.index]]
        df_train.index = [k for k in range(len(df_train))]
        df_test = df_full.iloc[e[0]:e[1]]
        df_test.index = [k for k in range(len(df_test))]

        train_set = CustomDataset(df_train, annot_dir, labels_type)
        test_set = CustomDataset(df_test, annot_dir, labels_type)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True, num_workers=0)
        df_res_train, df_res_test, report = fold_loop(
            model, model_dir, model_fn, weights_loss, train_loader, test_loader, learning_rate, epochs)
        print("\nSaving train and test performances for the {}-th fold".format(k))
        df_res_train.to_csv(os.path.join(
            model_dir, model_fn + "_values_train"+str(k)+".csv"), index=False)
        df_res_test.to_csv(os.path.join(
            model_dir, model_fn + "_values_test"+str(k)+".csv"), index=False)
        with open(os.path.join(model_dir, model_fn + "metrics_report"+str(k)+".txt"), 'w') as report_file:
            report_file.write(report)

        df_res_trains += df_res_train
        df_res_tests += df_res_test
        model = cnn(dim_in, n_classes)
        k += 1
    df_res_trains /= K
    df_res_tests /= K
    print("\nSaving train and test performances averaged over the {} folds".format(K))
    df_res_trains.to_csv(os.path.join(
        model_dir, model_fn + "_values_trains_averaged"+".csv"), index=False)
    df_res_tests.to_csv(os.path.join(model_dir, model_fn +
                        "_values_tests_averaged"+".csv"), index=False)
    print("\Done")
