import cv2
import matplotlib.pyplot as plt
import spectral as sp
import spectral.io.envi as envi
import spectral.io.bipfile as bf
import numpy as np
from preprocessing import preprocessing

PATH = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/'
PATH_OUT = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/img/'
file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
ext = '.hdr' #'.hyspex'

array_bbox_ = [[2000, 200, 2500, 310],
               [2000, 200, 2250, 550],
               [2000, 1000, 2250, 1250]]

array_bbox_ = preprocessing(PATH, file)
print(array_bbox_[0])
print(array_bbox_[1])
print(array_bbox_[2])
#exit()

all_heights = []
all_widths = []
for k in range(len(array_bbox_)):
    width = array_bbox_[k][3] - array_bbox_[k][1]
    height = array_bbox_[k][2] - array_bbox_[k][0]
    all_widths.append(width)
    all_heights.append(height)


max_height = max(all_heights)
max_width = max(all_widths)

print(max_height)
print(max_width)


img = sp.open_image(PATH + file + ext)
print(img.shape)
c = 0
for box in array_bbox_:
    grain_img = img[box[1]:box[3], box[0]:box[2]]
    print(grain_img.shape)
    w, h = grain_img.shape[0], grain_img.shape[1]
    print('w', w)
    print('h', h)

    x1, y1 = (max_width - w) // 2, (max_height - h) // 2
    print('x1', x1)
    print('y1', y1)

    x2, y2 = x1 + w, y1 + h
    print('x2', x2)
    print('y2', y2)

    new_img = np.zeros((max_width, max_height, 216))
    new_img[x1:x2, y1:y2, :] = grain_img

    file_name = PATH_OUT + 'grain' + str(c) + '.hdr'
    c += 1
    envi.save_image(file_name, new_img, force = True)

file = 'grain0'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 100]
plt.imshow(img0)
plt.show()

file = 'grain1'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 100]
plt.imshow(img0)
plt.show()

file = 'grain2'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 100]
plt.imshow(img0)
plt.show()
