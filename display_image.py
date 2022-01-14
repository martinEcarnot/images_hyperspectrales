import cv2
import matplotlib.pyplot as plt
import spectral as sp
import numpy as np
import torch

def display_img(path_in, filename, ext):
    img = sp.open_image(path_in + filename + ext)
    img_r = img[:, :, 70]
    img_g = img[:, :, 40]
    img_b = img[:, :, 10]

    img_r = cv2.resize(img_r, (200, 900))
    img_g = cv2.resize(img_g, (200, 900))
    img_b = cv2.resize(img_b, (200, 900))

    img = np.dstack((img_r,img_g,img_b))
    print(img.shape)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    #plt.axis('off')
    #plt.imshow(img)
    #plt.show()

PATH = 'D:/EMA/Image/ET/'
#PATH = "D:/Etude technique/"
file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
PATH_OUT = PATH + file + '/'
ext = '.hdr'  # '.hyspex'
display_img(PATH, file, ext)