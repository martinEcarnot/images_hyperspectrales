import cv2
import spectral as sp
import spectral.io.envi as envi
import numpy as np
from preprocessing import preprocessing
from tqdm import tqdm
import os
from os import walk


def crop_image(path_in, path_out, filename, ext, thresh_lum_spectralon=22000, crop_idx_dim1=1000,
               band_step=1, apply_mask=False, force_creation=False):
    """
    Given an hyperspectral image, use the function preprocessing from preprocessing.py to retrieve bbox coordinates,
    extract the hyperspectral image for each grain and save it in a particular folder.

    :param path_in: path of the folder of the hyperspectral image
    :param path_out: path of the folder to save grain images (ex: path_in + filename + '/')
    :param filename: name of the image to work with (ex: 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr')
    :param ext: extension ('hdr' here)
    :param thresh_lum_spectralon: threshold of light intensity to remove background + milli
    :param crop_idx_dim1: index of the edge of the spectralon
    :param band_step: step between two wave bands ( if set to 2, takes one out of two bands)
    :param apply_mask: bool to apply convex mask to the grain in order to keep only the grain, no background
    :param force_creation: bool, if the file already exist, set to True to force the rewriting
    """
    bool_file = 0
    # Creation of the folder if it doesn't exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        bool_file = 1
    # By default, if the folder already exists, nothing is done
    if bool_file or force_creation:
        arr_bbox, masks = preprocessing(path_in, filename, thresh_lum_spectralon=thresh_lum_spectralon,
                                        crop_idx_dim1=crop_idx_dim1)
        # all_heights = []
        # all_widths = []
        # for k in range(len(arr_bbox)):
        #     width = arr_bbox[k][3] - arr_bbox[k][1]
        #     height = arr_bbox[k][2] - arr_bbox[k][0]
        #     all_widths.append(width)
        #     all_heights.append(height)
        # max_height = max(all_heights)
        # max_width = max(all_widths)

        # Static max for the neural network to work
        max_height = 180
        max_width = 180

        img = sp.open_image(path_in + filename + ext)
        # Loop over all bbox detected with a smart progress meter (tqdm)
        for k in tqdm(range(len(arr_bbox))):

            box = arr_bbox[k]
            grain_img = img[box[1]:box[3], box[0]:box[2]]
            w, h = grain_img.shape[0], grain_img.shape[1]

            # Set the grain in the middle of the image
            x1, y1 = (max_width - w) // 2, (max_height - h) // 2
            x2, y2 = x1 + w, y1 + h

            # The number of band is considered as a static number, 216
            n_bands = 216 // band_step
            new_img = np.zeros((max_width, max_height, n_bands))

            for j in range(n_bands):
                new_grain = grain_img[:, :, j * band_step]
                if apply_mask:
                    dst = cv2.bitwise_and(new_grain, new_grain, mask=np.array(masks[k]).transpose())
                    new_img[x1:x2, y1:y2, j] = dst
                else:
                    new_img[x1:x2, y1:y2, j] = new_grain

            file_name = path_out + 'grain' + str(k) + '.hdr'
            envi.save_image(file_name, new_img, force=True)


def crop_all_images(use_path, band_step_=20, thresh_lum_spectralon_=8000, crop_idx_dim1_=1200,
                    apply_mask=True, force_creation=True):
    """
    Use of the crop_image function to extract all hyperspectral image at once into a train and test sub-folders

    :param use_path: path where all hyperspectral images are stored
    :param band_step_: step between two wave bands ( if set to 2, takes one out of two bands)
    :param thresh_lum_spectralon_: threshold of light intensity to remove background, set to 8000 due to darker images
    :param crop_idx_dim1_: index of the edge of the spectralon
    :param apply_mask: bool to apply convex mask to the grain in order to keep only the grain, no background
    :param force_creation: bool, if the file already exist, set to True to force the rewriting
    """
    all_files = next(walk(use_path), (None, None, []))[2]  # Detect only the files, not the folders
    hdr_files = [x for x in all_files if "hdr" in x]  # Extract hdr files
    train_path = os.path.join(use_path, "train")
    test_path = os.path.join(use_path, "test")
    if not os.path.exists(train_path):  # Creation train folder
        os.makedirs(train_path)
    if not os.path.exists(test_path):  # Creation test folder
        os.makedirs(test_path)

    ext = '.hdr'
    # More values can be ask for each hyperspectral such as thresh_refl or area_range.
    path = os.path.join(use_path, "")
    for file in hdr_files:
        filename = file[:-4]  # Remove extension
        path_out = os.path.join(test_path, filename, "") if "2021" in file else os.path.join(train_path, filename, "")
        crop_image(path, path_out, filename, ext, thresh_lum_spectralon=thresh_lum_spectralon_,
                   crop_idx_dim1=crop_idx_dim1_, band_step=band_step_, apply_mask=apply_mask,
                   force_creation=force_creation)
