import numpy as np
from tqdm import tqdm


def brightest_band(img):
    """
    Isolates a small part of the spectralon for each band and calculates its mean brightness

    :param img: the spectral image (only hdr)
    :return: the index of the brightest band and the associated value
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


