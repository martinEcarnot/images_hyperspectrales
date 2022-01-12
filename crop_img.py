import cv2
import matplotlib.pyplot as plt
import spectral as sp
import spectral.io.envi as envi
import spectral.io.bipfile as bf
import numpy as np
from preprocessing import preprocessing
from tqdm import tqdm
import os

PATH = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/'
file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
PATH_OUT = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/' + file + '/'
ext = '.hdr'  # '.hyspex'


def crop_image(path_in, path_out, filename, ext, band_step=1, apply_mask=False):
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        arr_bbox, masks = preprocessing(path_in, filename)
        all_heights = []
        all_widths = []
        for k in range(0, len(arr_bbox), band_step):
            width = arr_bbox[k][3] - arr_bbox[k][1]
            height = arr_bbox[k][2] - arr_bbox[k][0]
            all_widths.append(width)
            all_heights.append(height)

        max_height = max(all_heights)
        max_width = max(all_widths)

        img = sp.open_image(path_in + filename + ext)
        for k in tqdm(range(0, len(arr_bbox), band_step)):

            box = arr_bbox[k]
            grain_img = img[box[1]:box[3], box[0]:box[2]]
            w, h = grain_img.shape[0], grain_img.shape[1]

            x1, y1 = (max_width - w) // 2, (max_height - h) // 2
            x2, y2 = x1 + w, y1 + h

            n_bands = 216 // band_step
            new_img = np.zeros((max_width, max_height, n_bands))

            for j in range(n_bands):
                if apply_mask:
                    new_img[x1:x2, y1:y2, j] = cv2.bitwise_and(grain_img, grain_img, mask=masks[j * band_step])
                else:
                    new_img[x1:x2, y1:y2, j] = grain_img[:, :, j * band_step]

            file_name = path_out + 'grain' + str(k) + '.hdr'
            envi.save_image(file_name, new_img, force=True)


crop_image(PATH, PATH_OUT, file, ext, band_step=20)

file = 'grain0'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 5]
plt.imshow(img0)
plt.show()
