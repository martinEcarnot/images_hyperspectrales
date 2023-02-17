import cv2
import matplotlib.pyplot as plt
import spectral as sp
import spectral.io.envi as envi
import numpy as np
from preprocessing import preprocessing, show_image
from tqdm import tqdm
import os
from os import walk
import statistics
import pandas as pd
import csv
from utils import *

def crop_image(img_dir, img_fn, out_dir, crop_idx_dim1=1300,
               bands = [i for i in range(216)], apply_mask=False, force_creation=False, verbose=True, sillon = True, defauts = [], autre_cat = []):
    """
    Given an hyperspectral image, use the function preprocessing from preprocessing.py to retrieve 
    bbox coordinates,extract the hyperspectral image for each grain and save it in a particular folder.

    :param img_dir: path of the folder of the hyperspectral image
    :param out_dir: path of the folder to save grain images (ex: img_dir + img_fn + '/')
    :param img_fn: name of the image to work with (ex: 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr')
    :param crop_idx_dim1: index of the edge of the graph paper
    :param band_step: step between two wave bands ( if set to 2, takes one out of two bands)
    :param apply_mask: bool to apply convex mask to the grain in order to keep only the grain, no background
    :param force_creation: bool, if the file already exist, set to True to force the rewriting
    :param verbose: Display the comparison between original image and the reflectance one and the 
                    original image with bbox if set to True
    """
    bool_file = 0
    # Creation of the folder if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        bool_file = 1

    # By default, if the folder already exists, nothing is done
    if bool_file or force_creation:
        coord_centroids, arr_bbox, masks = preprocessing(img_dir, img_fn, crop_idx_dim1=crop_idx_dim1, verbose=verbose)
        
        labels = []
        for i in range(len(coord_centroids)):
            labels.append(2 if i in defauts else int(sillon)^(i in autre_cat))
        arr_bbox_copy = [x.tolist() if type(x)==np.ndarray else x for x in arr_bbox]
        df_centroids = pd.DataFrame({'Coord_centroid':coord_centroids, 'Bbox': list(arr_bbox_copy), 'Label' : labels})  #, 'Class' : liste_labels
        df_centroids.to_csv(os.path.join(out_dir, "annotations_" + img_fn + '.csv'), index = False)
        # Static max for the neural network to work
        max_height = 200
        max_width = 200
        
        
        img = sp.open_image(img_dir + img_fn + '.hdr')

        # The number of band is considered as a static number, 216
        n_bands = len(bands)

        # Retrieve values to convert grain image to reflectance
        array_ref = np.genfromtxt(os.path.join(img_dir, "csv", "lum_spectralon_" + img_fn + ".csv"), delimiter=',')

        # Loop over all bbox detected with a smart progress meter (tqdm)
        for k in tqdm(range(len(arr_bbox)), desc="Creating images"):

            box = arr_bbox[k]
            grain_img = img[box[1]:box[3], box[0]:box[2]].astype('float64')
            # Function built in the spectral library but as fast as the previous method

            w, h = grain_img.shape[0], grain_img.shape[1]

            # Set the grain in the middle of the image
            x1, y1 = (max_width - w) // 2, (max_height - h) // 2
            x2, y2 = x1 + w, y1 + h

            new_img = np.zeros((max_width, max_height, n_bands))

            edge = box[0]  # Coordinate of the column in the original image

            for j in range(n_bands):
                new_grain = grain_img[:, :, bands[j]]
                # Reflectance
                for x in range(0, h):
                    lum = array_ref[x+edge, bands[j]]
                    new_grain[:, x] = new_grain[:, x] / lum if lum != 0 else [0] * w
                if apply_mask:
                    dst = cv2.bitwise_and(new_grain, new_grain, mask=np.array(masks[k]).transpose())
                    new_img[x1:x2, y1:y2, j] = dst
                else:
                    new_img[x1:x2, y1:y2, j] = new_grain

            file_name = out_dir + img_fn + '_grain' + str(k) + '.hdr'
            envi.save_image(file_name, new_img, force=True)


def crop_all_images(path_img, path_out, path_defauts, bands = [i for i in range(216)], crop_idx_dim1=1300, apply_mask=True, force_creation=True, verbose=False):
    """
    Use of the path_img function to extract all hyperspectral image at once into a train, valid and test sub-folders
    of a folder entitled with the number of bands

    :param path_img: path where all hyperspectral images are stored
    :param band_step: step between two wave bands ( if set to 2, takes one out of two bands)
    :param crop_idx_dim1: index of the edge of the graph paper
    :param apply_mask: bool to apply convex mask to the grain in order to keep only the grain, no background
    :param force_creation: bool, if the file already exist, seft to True to force the rewriting
    :param verbose: Display the comparison between original image and the reflectance one and the original
                    image with bbox if set to True
    """

    all_files = next(walk(path_img), (None, None, []))[2]  # Detect only the files, not the folders
    hdr_files = [x for x in all_files if "hdr" in x]  # Extract hdr files
    
    if not os.path.exists(path_out):  # Creation folder entitled with the number of band
        os.makedirs(path_out)
    
    df_annotations = pd.read_csv(path_defauts)
    
    df_annotations.set_index('Image', inplace=True)
    df_annotations.fillna('', inplace=True)
    for file in hdr_files:
        filename = file[:-4]  # Remove extension
        sillon = df_annotations.loc[filename,"Type attendu"]
        if sillon=='sillon':
            sillon=True
        else :
            sillon=False
        defauts = df_annotations.loc[filename,"Liste grains defauts"].split(',')
        defauts = [int(elem) for elem in defauts if elem!='']
        autre_cat = df_annotations.loc[filename,"Liste grains autre type"].split(',')
        autre_cat = [int(elem) for elem in autre_cat if elem!='']
        crop_image(path_img, filename, path_out, crop_idx_dim1=crop_idx_dim1, bands=bands,
                   apply_mask=apply_mask, force_creation=force_creation, verbose=verbose,sillon = sillon, defauts = defauts, autre_cat = autre_cat)


def get_grain_size(path_in, crop_idx_dim1=1300, verbose=False):
    """
    Display the comparison between grain size of all images

    :param path_in: path of the hyperspectral images
    :param crop_idx_dim1: index of the edge of the graph paper
    :param verbose: To display preprocessing figure
    """
    use_path = os.path.join(path_in, "")
    all_files = next(walk(use_path), (None, None, []))[2]  # Detect only the files, not the folders
    name_files = [x[:-4] for x in all_files if x[-3:] == "hdr"]  # Extract name files

    names, size_boxplot = [""], []

    for name in name_files:
        print(name)
        _, arr_bbox, masks = preprocessing(use_path, name, crop_idx_dim1=crop_idx_dim1, verbose=verbose)

        all_sizes = []
        for mask in masks:
            size = sum(sum(mask))
            all_sizes.append(size)
        size_boxplot.append(all_sizes)

        print('len', len(all_sizes))
        print('min', min(all_sizes))
        print('max', max(all_sizes))
        print('mean', statistics.mean(all_sizes))
        print('stdev', np.std(all_sizes))
        print('quantiles', statistics.quantiles(all_sizes))
        print('\n')

        # Determination of the year and the variety
        variety = name.split("_")[0].split("-")
        var = variety[0] if "var" in variety[0] else variety[1]
        year = name.split("_")[-2].split("-")[0]
        position = name.split("-")
        pos_tmp = position[0] if "x" in position[0] else position[1]
        pos_tmp = pos_tmp.split("_")
        pos = pos_tmp[0] if "x" in pos_tmp[0] else pos_tmp[2]
        names.append(f'{pos}_{var}_{year}')

    plt.boxplot(size_boxplot)

    list_nb = range(len(name_files)+1)

    plt.xticks(list_nb, names)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Name image')
    plt.ylabel('Area (pixel)')
    plt.title("Comparison sizes of grains per image")
    plt.grid()
    plt.tight_layout()
    plt.show()