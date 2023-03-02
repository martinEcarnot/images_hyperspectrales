import spectral as sp
import numpy as np
import pandas as pd
import torch


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