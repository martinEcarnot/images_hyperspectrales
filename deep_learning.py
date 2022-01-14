# Global import
import pandas as pd
import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader, Dataset
# from sklearn.metrics import classification_report


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
