import cv2
import matplotlib.pyplot as plt
import spectral as sp
import numpy as np
import torch
import time

def band_brigthness(img, k):
    """
    isolates a small part of the spectralon for band k
    and calculates its mean brightness

    :param img: the spectral image (only hdr)
    :param k: the band index
    :return: the index of the brightest band
    """
    rows = [i for i in range(100)]
    cols = [i for i in range(150, img.shape[1])]
    bands = [k]
    band = np.array(img.read_subimage(rows, cols, bands))
    mean = np.mean(band)

    return mean

def display_img(path_in, filename):
    '''
    Display a hyperspectral image as an RGB image
    R : 460 nm -> band n°22
    G : 540 nm -> band n°53
    B : 640 nm -> band n°89
    :param path_in: path containing the image file
    :param filename: name of the image file
    '''

    img = sp.open_image(path_in + filename + '.hdr')

    # Normalizing each band regarding the mean brigthness of the spectralon
    img_r = img[:, :, 22] / band_brigthness(img, 22)
    img_g = img[:, :, 53] / band_brigthness(img, 53)
    img_b = img[:, :, 89] / band_brigthness(img, 89)

    # Stack 3 channels
    img = np.dstack((img_b, img_g, img_r))
    # print(img.shape)

    # Rotate the image by 90 degree
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Display the image
    plt.axis('off')
    plt.imshow(img)
    plt.show()

PATH = "E:\\Etude technique\\raw\\"
#PATH = "D:/Etude technique/"
file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
file = 'var1-x73y14_7000_us_2x_2021-10-23T151946_corr'

a = time.time()
display_img(PATH, file)
print(int(time.time() - a))
