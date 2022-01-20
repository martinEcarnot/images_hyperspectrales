import spectral as sp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def brightest_band(img):
    """
    isolates a small part of the spectralon for each band
    and calculates its mean brightness

    :param img: the spectral image (only hdr)
    :return: the index of the brightest band
    """
    rows = [i for i in range(100)]
    cols = [i for i in range(150, img.shape[1])]

    means = []
    for k in tqdm(range(216), desc='calculating the brightest band'):
        bands = [k]

        band = np.array(img.read_subimage(rows, cols, bands))
        mean = np.mean(band)
        means.append(mean)

    return means.index(max(means))

# PATH = 'E:\\Etude Technique\\raw\\'
# # PATH = 'D:\\Etude Technique\\'
# file = 'var1-x73y14_7000_us_2x_2021-10-23T151946_corr'
#
# print(brightest_band(PATH, file))

