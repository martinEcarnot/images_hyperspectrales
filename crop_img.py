import cv2
import matplotlib.pyplot as plt
import spectral as sp
import spectral.io.envi as envi
import spectral.io.bipfile as bf
import numpy as np
from preprocessing import preprocessing
from tqdm import tqdm

PATH = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/'
PATH_OUT = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/img/'
file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
ext = '.hdr' #'.hyspex'


array_bbox_ = preprocessing(PATH, file)

def crop_image(path, filename, arr_box, band_step = 1, apply_mask = False):
    all_heights = []
    all_widths = []
    for k in range(0, len(arr_box), band_step):
        width = array_bbox_[k][3] - array_bbox_[k][1]
        height = array_bbox_[k][2] - array_bbox_[k][0]
        all_widths.append(width)
        all_heights.append(height)

    max_height = max(all_heights)
    max_width = max(all_widths)

    img = sp.open_image(PATH + file + ext)
    for k in tqdm(range(0, len(arr_box), band_step)):

        box = arr_box[k]
        grain_img = img[box[1]:box[3], box[0]:box[2]]
        w, h = grain_img.shape[0], grain_img.shape[1]

        x1, y1 = (max_width - w) // 2, (max_height - h) // 2
        x2, y2 = x1 + w, y1 + h

        new_img = np.zeros((max_width, max_height, 216))
        new_img[x1:x2, y1:y2, :] = grain_img

        file_name = PATH_OUT + 'grain' + str(k) + '.hdr'
        envi.save_image(file_name, new_img, force = True)

'''
file = 'grain2'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 100]
plt.imshow(img0)
plt.show()
'''
