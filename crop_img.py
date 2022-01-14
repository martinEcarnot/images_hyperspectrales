import cv2
import spectral as sp
import spectral.io.envi as envi
import numpy as np
from preprocessing import preprocessing
from tqdm import tqdm
import os


def crop_image(path_in, path_out, filename, ext, band_step=1, apply_mask=True, force_creation=False):
    bool_file = 0
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        bool_file = 1
    if bool_file or force_creation:
        arr_bbox, masks = preprocessing(path_in, filename)
        all_heights = []
        all_widths = []
        for k in range(len(arr_bbox)):
            width = arr_bbox[k][3] - arr_bbox[k][1]
            height = arr_bbox[k][2] - arr_bbox[k][0]
            all_widths.append(width)
            all_heights.append(height)

        max_height = max(all_heights)
        max_width = max(all_widths)

        img = sp.open_image(path_in + filename + ext)
        for k in tqdm(range(len(arr_bbox))):

            box = arr_bbox[k]
            grain_img = img[box[1]:box[3], box[0]:box[2]]
            w, h = grain_img.shape[0], grain_img.shape[1]

            x1, y1 = (max_width - w) // 2, (max_height - h) // 2
            x2, y2 = x1 + w, y1 + h

            n_bands = 216 // band_step
            new_img = np.zeros((max_width, max_height, n_bands))

            for j in range(n_bands):
                new_grain = grain_img[:, :, j * band_step]
                if apply_mask:
                    dst = cv2.bitwise_and(new_grain, new_grain, mask = np.array(masks[k]).transpose())
                    new_img[x1:x2, y1:y2, j] = dst
                else:
                    new_img[x1:x2, y1:y2, j] = new_grain

            file_name = path_out + 'grain' + str(k) + '.hdr'
            envi.save_image(file_name, new_img, force=True)
