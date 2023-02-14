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



def display_single_band(path_in, filename, selected_bands):
    """
    Display a hyperspectral image as an RGB image
    R : 460 nm -> band n°22
    G : 540 nm -> band n°53
    B : 640 nm -> band n°89
    :param path_in: path containing the image file
    :param filename: name of the image file
    """
    a = time.time()

    img = sp.open_image(path_in + filename + '.hdr')
    #img = sp.open_image(path_in + '/' +  filename)
    
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


# PATH = "E:\\Etude technique\\raw\\"
PATH = 'img/'
#file = "var1_2020_x75y20_8000_us_2x_2022-04-26T122543_corr"
file = "var1_2020_x75y20_8000_us_2x_2022-04-26T130045_corr"
#file = 'vaches'
# file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
# file = 'var1-x73y14_7000_us_2x_2021-10-23T151946_corr'
# file = 'x30y21-var1_11000_us_2x_2020-12-02T095609_corr'
# file = 'x32y23-var8_8000_us_2x_2020-12-02T155853_corr'

display_img(PATH, file, 106)

