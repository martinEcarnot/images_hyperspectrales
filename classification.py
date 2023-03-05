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


def summary_training(model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands):
    """Generates a summary of a training, and stores it in a .txt file.

    :param model: model used for the training
    :type model: torch.nn.Module
    :param annot_dir: directory of the annotations files
    :type annot_dir: str
    :param labels_type: 'Face' for the face classification, 'Species' for species classification
    :type labels_type: str
    :param weights_loss: loss' weights associated to each class
    :type weights_loss: list(float)
    :param learning_rate: learning rate used for the training
    :type learning_rate: float
    :param epochs: number of epochs used for the training
    :type epochs: int
    :param batch_size: batch size used for the training
    :type batch_size: int
    :param other_class: in the case of a face classification, True implies the usage of the 3rd class 'Autre'
    :type other_class: bool
    :param bands: bands of the spectrum used for the training
    :type bands: list(int)
    """
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


def display_save_figure(fig_dir, fig_fn, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid):
    """Display and save the accuracy and loss curves for train and validation data.

    :param fig_dir: directory which will contain the figure to be saved
    :type fig_dir: str
    :param fig_fn: filename of the figure to be saved
    :type fig_fn: str
    :param list_accu_train: list containing the train accuracy values over the epochs
    :type list_accu_train: list(float)
    :param list_accu_valid: list containing the validation accuracy values over the epochs
    :type list_accu_valid: list(float)
    :param list_loss_train: list containing the train loss values over the epochs
    :type list_loss_train: list(float)
    :param list_loss_valid: list containing the validation loss values over the epochs
    :type list_loss_valid: list(float)
    """
    
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7))
    ax = axes.ravel()
    list_nb = range(len(list_accu_train))
    ax[0].plot(list_nb, list_accu_train, label='Train')
    ax[0].plot(list_nb, list_accu_valid, label='Valid')
    ax[0].set_title('Accuracy given the epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(list_nb, list_loss_train, label='Train')
    ax[1].plot(list_nb, list_loss_valid, label='Valid')
    ax[1].set_title('Loss given the epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid()

    fig.suptitle(fig_fn)  # Global title
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(fig_dir, fig_fn+".png"), dpi=200, format='png')

def model_testing(model_fn, annot_dir, labels_type, test_annot_fn = 'test_set', other_class = False, chosen_var = [], chosen_face = None):
    """_summary_

    :param model_fn: filename of the model to use
    :type model_fn: str
    :param annot_dir: directory of the annotations .csv files
    :type annot_dir: str
    :param labels_type: 'Face' if face classification, 'Species' if species classification
    :type labels_type: str
    :param test_annot_fn: filename of the test annotations csv file , defaults to 'test_set'
    :type test_annot_fn: str, optional
    :param other_class: _description_, defaults to False
    :type other_class: bool, optional
    :param chosen_var: _description_, defaults to []
    :type chosen_var: list, optional
    :param chosen_face: _description_, defaults to None
    :type chosen_face: _type_, optional
    """
    df_test = pd.read_csv(annot_dir + test_annot_fn + '.csv')
    weight_loss = [2., 2., 2.]
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
        weight_loss = [2., 2.]
    if labels_type == 'Species' :
        if len(chosen_var)==0:
            chosen_var = [i for i in range(8)]
        df_test = df_test.loc[df_test['Species'].isin(chosen_var)]
        
        if chosen_face == 'Dos' :
            df_test = df_test.loc[df_test['Face']==0]
        elif chosen_face == 'Sillon' :
            df_test = df_test.loc[df_test['Face']==1]
    df_test.index = [i for i in range(len(df_test))]
    test_set = CustomDataset(df_test, annot_dir, labels_type = 'Face')
    test_loader = DataLoader(test_set, batch_size=12, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join('models', model_fn, model_fn + '.pth'))
    weight = torch.tensor(weight_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    test_model(test_loader, device, model=model, loss_f=loss_f, test_dir = annot_dir, model_fn = model_fn, labels_type = labels_type, other_class = other_class, chosen_var = chosen_var)
       
        
        
def train_model(train_loader, val_loader, device, model, loss_f, optimizer, verbose=False):
    """Performs one epoch of training and validation of a given model.
    :param train_loader: DataLoader of the training dataset
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: DataLoader of the validation dataset
    :type val_loader: torch.utils.data.DataLoader
    :param device: device on which the training will be performed ('cuda' or 'cpu')
    :type device: str
    :param model: model to be trained
    :type model: torch.nn.Module
    :param loss_f: loss function used for the training
    :type loss_f: torch.nn.functional
    :param optimizer: optimizer used for the training (Adam)
    :type optimizer: torch.optim
    :param verbose: Set to True to display the model parameters and information during training, defaults to False
    :type verbose: bool, optional
    :return: The trained model, train accuracy, validation accuracy, train loss, validation loss
    :rtype: tuple(torch.nn.Module, list(float), list(float), list(float), list(float))
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



def test_model(test_loader, device, model, loss_f, test_dir, model_fn, labels_type, test_name = 'test_set', other_class = False, chosen_var = [], chosen_face = None):
    """Performs a model inference on a test dataset and computes the metrics of performance, and saves them in a .csv file.
    :param test_loader: test DataLoader used for testing
    :type test_loader: torch.utils.data.DataLoader
    :param device: device on which the test will be performed ('cuda' or 'cpu')
    :type device: str
    :param model: model to be trained
    :type model: torch.nn.Module
    :param loss_f: loss function used for the training
    :type loss_f: torch.nn.functional
    :param test_dir: _description_
    :type test_dir: _type_
    :param model_fn: filename of the testing's output
    :type model_fn: str
    :param labels_type: 'Face' for the face classification, 'Species' for species classification
    :type labels_type: str
    :param test_name: _description_, defaults to 'test_set'
    :type test_name: str, optional
    :param other_class: True if the class 'Autre' is considered, defaults to False
    :type other_class: bool, optional
    :param chosen_var: _description_, defaults to []
    :type chosen_var: list, optional
    :param chosen_face: _description_, defaults to None
    :type chosen_face: _type_, optional
    :return: accuracy and loss of the test
    :rtype: tuple(float,float)
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
    if labels_type == 'Species' :
        if len(chosen_var)==0:
            chosen_var = [i for i in range(8)]
        df_test = df_test.loc[df_test['Species'].isin(chosen_var)]
        df_test.index = [i for i in range(len(df_test))]
        if chosen_face == 'Dos' :
            df_test = df_test.loc[df_test['Face']==0]
        elif chosen_face == 'Sillon' :
            df_test = df_test.loc[df_test['Face']==1]
    df_test.index = [i for i in range(len(df_test))]
    df_test['Probas'] = list_y_probas
    if labels_type == 'Face' :
        df_test['Face_pred'] = list_y_pred
    elif labels_type == 'Species':
        df_test['Species_pred'] = [chosen_var[i] for i in list_y_pred]
    
    #saving of the results
    df_test.to_csv('models/' + model_fn + '/test_preds.csv', index = False)
    print(f"Test : \n Accuracy: {(100 * correct):>0.1f}% \t Avg loss: {test_loss:>8f} \n")

    # Determination of other metrics
    print(classification_report(list_y, list_y_pred, labels=np.unique(list_y)))
    return 100*correct, test_loss



def compute_metrics(preds_dir, preds_fn):
    """Computes the metrics of performances (loss, accuracy, recall, precision, f-score) of a training,
    from a dataset containing both the labels and the predicted values, and stores them in a new .csv file

    :param preds_dir: directory of the dataset
    :type preds_dir: str
    :param preds_fn: filename of the dataset
    :type preds_fn: str
    """
    df = pd.read_csv(preds_dir + preds_fn + '.csv')
    if df.columns[-1] == 'Face_pred':
        expected = df['Face']
        preds = df['Face_pred']
    elif df.columns[-1] == 'Species_pred':
        expected = df['Species']
        preds = df['Species_pred']
    labels = np.unique(expected)
    liste_precision = []
    liste_recall = []
    for i in labels:
        prec = np.floor(len(df.loc[(preds==i) & (expected==i)]) / len(df.loc[preds==i])*100)/100
        rec =  np.floor(len(df.loc[(preds==i) & (expected==i)]) / len(df.loc[expected==i])*100)/100
        liste_precision.append(prec)
        liste_recall.append(rec)  
    df_metrics = pd.DataFrame({'Precision' : liste_precision, 'Recall' : liste_recall})
    df_metrics.index = labels
    df_metrics.to_csv(preds_dir + 'metrics.csv')   

    
def main_loop(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=50, batch_size=24, other_class = False, chosen_var = [], chosen_face = None):
    """For a given model, performs many epochs of training and validation, and tests the model.

    :param annot_dir: directory of the annotations .csv files
    :type annot_dir: str
    :param cnn: class of the used cnn
    :type cnn: CNN_1,CNN_2 or CNN_3
    :param model_fn: filename associated with the training
    :type model_fn: str
    :param labels_type: 'Face' for the face classification, 'Species' for species classification
    :type labels_type: str
    :param weights_loss: classes' weights used in the loss (one weight for each class)
    :type weights_loss: list(float)
    :param learning_rate: learning rate used for the training
    :type learning_rate: float
    :param epochs: number of epochs used for the training, defaults to 50
    :type epochs: int, optional
    :param batch_size: batch size used for the training, defaults to 24
    :type batch_size: int, optional
    :param other_class: True if the class 'Autre' is considered, defaults to False
    :type other_class: bool, optional
    :param chosen_var: _description_, defaults to []
    :type chosen_var: list, optional
    :param chosen_face: _description_, defaults to None
    :type chosen_face: _type_, optional
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param cnn: class of the CNN to be used.
    :param annot_dir: path to the directory of the annotations files
    :labels_type: name of the column containing the labels, here "Face" or "Species"
    :param weights_loss: the weights to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    
    n_classes = 3
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
    print(torch.cuda.is_available())
    
    df_train = pd.read_csv(annot_dir + 'train_set.csv')
    if not other_class :
        df_train = df_train.loc[df_train['Face']!=2]
        n_classes = 2
    if labels_type == 'Species' :
        if len(chosen_var)==0:
            chosen_var = [i for i in range(8)]
        print("Variétés que l'on va chercher à différencier : " + str(chosen_var)[1:-1])
        df_train = df_train.loc[df_train['Species'].isin(chosen_var)]
        if chosen_face == 'Dos' :
            df_train = df_train.loc[df_train['Face']==0]
        elif chosen_face == 'Sillon' :
            df_train = df_train.loc[df_train['Face']==1]
        n_classes = len(chosen_var)
    df_train.index = [i for i in range(len(df_train))]
    train_set = CustomDataset(df_train, annot_dir, labels_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)

    df_valid = pd.read_csv(annot_dir + 'validation_set.csv')
    if not other_class :
        df_valid = df_valid.loc[df_valid['Face']!=2]
        df_valid.index = [i for i in range(len(df_valid))]
    if labels_type == 'Species' :
        df_valid = df_valid.loc[df_valid['Species'].isin(chosen_var)]
        if chosen_face == 'Dos' :
            df_valid = df_valid.loc[df_valid['Face']==0]
        elif chosen_face == 'Sillon' :
            df_valid = df_valid.loc[df_valid['Face']==1]
    df_valid.index = [i for i in range(len(df_valid))]
    val_set = CustomDataset(df_valid, annot_dir, labels_type)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)

    df_test = pd.read_csv(annot_dir + 'test_set.csv')
    
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
    if labels_type == 'Species' :
        df_test = df_test.loc[df_test['Species'].isin(chosen_var)]
        if chosen_face == 'Dos' :
            df_test = df_test.loc[df_test['Face']==0]
        elif chosen_face == 'Sillon' :
            df_test = df_test.loc[df_test['Face']==1]
    df_test.index = [i for i in range(len(df_test))]
    
    model = cnn(dim_in, n_classes).to(device)
    print(next(model.parameters()).device)

    weight = torch.tensor(weights_loss).to(device)
    loss_f = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    test_set = CustomDataset(df_test, annot_dir, labels_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)
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
    torch.save(model, model_path)
    
    recap_path = os.path.join("models",model_fn,model_fn+"_summary.txt")
    with open(recap_path,'w') as recap_file:
        recap_file.write(summary_training(
            model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands = bands))
    fig_fn = model_fn+"_training_evolution"
    display_save_figure(model_dir, fig_fn, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid)

    print("\nTesting model")
    test_accu, test_loss = test_model(test_loader, device, model=model, loss_f=loss_f, test_dir = annot_dir, model_fn = model_fn, labels_type = labels_type, other_class = other_class, chosen_var = chosen_var, chosen_face = chosen_face)

    # Saving values
    print("\nSaving values of train, validation and test loops")
    df_res_train_val = pd.DataFrame({"Train accuracy":list_accu_train,"Validation accuracy":list_accu_valid,"Train loss":list_loss_train,"Validation loss":list_loss_valid})
    df_res_train_val.to_csv(os.path.join(model_dir, model_fn +"_values_train_valid.csv"),index=False)
    df_res_test = pd.DataFrame({"Test accuracy":test_accu,"Test loss":test_loss},index=[0])
    df_res_test.to_csv(os.path.join(model_dir, model_fn +"_values_test.csv"),index=False)


    print("\nDone!")