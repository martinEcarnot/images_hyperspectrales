import os

import numpy as np
import pandas as pd
import spectral as sp
from tqdm import tqdm


def brightest_band(img):
    """Isolates a small part of the spectralon for each band and calculates its mean brightness.

    :param img: the hyperspectral image
    :type img: SpyFile object
    :return: the index of the brightest band and the associated value
    :rtype: list
    """
    

    rows = [i for i in range(100)]
    cols = [i for i in range(150, img.shape[1])]

    means = []
    for k in tqdm(range(216), desc='Calculating the brightest band'):
        bands = [k]

        band = np.array(img.read_subimage(rows, cols, bands))
        mean = np.mean(band)
        means.append(mean)

    return means.index(max(means)), max(means)


def retrieve_all_brightest_bands_to_csv(img_dir):
    """Create a CSV with the value and index of the brightest band for each image.

    :param img_dir: path of the images directory
    :type img_dir: str
    """
    
    list_fn = os.listdir(img_dir)
    list_hdr_fn = [x for x in list_fn if "hdr" in x]
    list_hdr_paths = [os.path.join(img_dir, x) for x in list_hdr_fn]

    df = pd.DataFrame(columns=["band", "max_ref"])
    for idx, path in enumerate(list_hdr_paths):
        img = sp.open_image(path)
        band, max_ref = brightest_band(img)
        df_tmp = pd.DataFrame([[band, max_ref]], columns=[
                              "band", "max_ref"], index=[list_hdr_fn[idx][:-4]])
        df = df.append(df_tmp)

    csv_dir = os.path.join(img_dir, "csv")
    if not os.path.exists(csv_dir):  # Creation directory if it doesn't exit
        os.makedirs(csv_dir)
    df.to_csv(os.path.join(csv_dir, "brightest_bands.csv"))
