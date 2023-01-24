import cv2
import matplotlib.pyplot as plt
import spectral as sp
import numpy as np
import time


def band_brigthness(img, k):
    """
    isolates a small part of the spectralon for band k
    and calculates its mean brightness

    :param img: the spectral image (only hdr)
    :param k: the band index
    :return: the mean brightness of the band k
    """
    rows = [i for i in range(100)]
    cols = [i for i in range(150, img.shape[1])]
    bands = [k]
    band = np.array(img.read_subimage(rows, cols, bands))
    mean = np.mean(band)

    return mean


def display_img(path_in, filename):
    """
    Display a hyperspectral image as an RGB image
    R : 460 nm -> band n°22         15
    G : 540 nm -> band n°53         52
    B : 640 nm -> band n°89         80
    :param path_in: path containing the image file
    :param filename: name of the image file
    """
    a = time.time()

    img = sp.open_image(path_in + filename + '.hdr')
    #img = sp.open_image(path_in + '/' +  filename)

    # Normalizing each band regarding the mean brigthness of the spectralon
    img_r = img[:, :, 22] / band_brigthness(img, 22)
    img_g = img[:, :, 53] / band_brigthness(img, 53)
    img_b = img[:, :, 89] / band_brigthness(img, 89)

    # Stack 3 channels
    img = np.dstack((img_b, img_g, img_r))
    print(img.shape)

    # Rotate the image by 90 degree
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    s = int(time.time() - a)
    t = str(s // 60) + ' min ' + str(s % 60) + ' sec'
    print(t)
    # Display the image
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def display_single_band(path_in, filename, selected_bands):
    a = time.time()

    img = sp.open_image(path_in + filename + '.hdr')
    
    # Normalizing each band regarding the mean brigthness of the spectralon
    """
    for i in selected_bands:
        img_band = img[:, :, i] / band_brigthness(img, i)
        comp_images.append(img_band)
    """
    img = img[:, :, selected_bands] / band_brigthness(img, selected_bands)
   
    # Rotate the image by 90 degree
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    s = int(time.time() - a)
    t = str(s // 60) + ' min ' + str(s % 60) + ' sec'
    print(t)
    # Display the image
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()

PATH = 'img/'
#file = "var1_2020_x75y20_8000_us_2x_2022-04-26T122543_corr"
#file = "var1_2020_x75y20_8000_us_2x_2022-04-26T130045_corr"
file = "var4_2020_x82y12_8000_us_2x_2022-04-27T093216_corr"
#file = "var4_2020_x82y12_8000_us_2x_2022-04-27T092007_corr"
#file = "cropped/var4_2020_x82y12_8000_us_2x_2022-04-27T093216_corr_grain25"


display_single_band(PATH, file, 103)
#display_img(PATH, file)
