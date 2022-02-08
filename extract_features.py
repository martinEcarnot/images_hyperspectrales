from preprocessing import *
import cv2
import statistics
from os import walk


def extract_features(path_in, filename, ext, crop_idx_dim1=1300, verbose=False):
    """
    Given an hyperspectral image, use the function preprocessing from preprocessing.py to retrieve bbox coordinates,
    extract the hyperspectral image for each grain and save it in a particular folder.

    :param path_in: path of the folder of the hyperspectral image
    :param filename: name of the image to work with (ex: 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr')
    :param ext: extension ('hdr' here)
    :param crop_idx_dim1: index of the edge of the graph paper
    :param verbose: Display the comparison between original image and the reflectance one and the original
                    image with bbox if set to True
    """
    path_in = os.path.join(path_in, "")
    arr_bbox, masks = preprocessing(path_in, filename, crop_idx_dim1=crop_idx_dim1, verbose=verbose)

    img = sp.open_image(path_in + filename + ext)

    n_bands = 216
    band_step = n_bands // 216  # To make sure to select the right band in the array_ref of the spectralon

    # Retrieve values to convert grain image to reflectance
    array_ref = np.genfromtxt(os.path.join(path_in, "csv", "lum_spectralon_" + filename + ".csv"), delimiter=',')

    # Array to store features (order: min, max, median, mean, std)
    array_features = np.zeros([len(arr_bbox), n_bands, 5])

    # Loop over all bbox detected with a smart progress meter (tqdm)
    for k in tqdm(range(len(arr_bbox)), desc="Extracting features"):

        box = arr_bbox[k]
        grain_img = img[box[1]:box[3], box[0]:box[2]].astype('float64')
        h, w = grain_img.shape[0], grain_img.shape[1]

        edge = box[0]  # Coordinate of the column in the original image

        for j in range(n_bands):
            new_grain = grain_img[:, :, j*band_step]
            # Reflectance
            for x in range(0, w):
                lum = array_ref[x+edge, j*band_step]
                new_grain[:, x] = new_grain[:, x] / lum if lum != 0 else [0] * h

            new_grain_mask = cv2.bitwise_and(new_grain, new_grain, mask=np.array(masks[k]).transpose())

            # Computing the average reflectance of the grain for one band
            x_size, y_size = new_grain_mask.shape
            total_value = []
            for i in range(x_size):
                for t in range(y_size):
                    value = new_grain_mask[i, t]
                    if value != 0:
                        total_value.append(value)

            # In some rare cases, lum = 0 -> new_grain is a zeros array. 0 are left in the features array
            if total_value:
                array_features[k, j] = [max(total_value), min(total_value), statistics.median(total_value),
                                        statistics.mean(total_value), statistics.stdev(total_value)]

    # Save
    use_path = os.path.join(path_in, "csv", "features_grains_" + filename + ".npy")
    np.save(use_path, array_features)  # Order: max, min, median, mean, std

    # Can t save 3D array with savetxt, so no CSV
    # array_load = np.load(use_path)  # To load back


def extract_all_features(path_in, crop_idx_dim1=1300, verbose=False):
    """
    Use of the function extract_features on all hyperspectral images and save them in the csv file.

    :param path_in: path of the folder of the hyperspectral image
    :param crop_idx_dim1: index of the edge of the graph paper
    :param verbose: Display the comparison between original image and the reflectance one and the original
                    image with bbox if set to True
    """
    use_path = os.path.join(path_in, "")
    all_files = next(walk(use_path), (None, None, []))[2]  # Detect only the files, not the folders
    hdr_files = [x for x in all_files if "hdr" in x]  # Extract hdr files
    number_files = len(hdr_files)
    ext = '.hdr'

    for idx, file in enumerate(hdr_files):
        print(f"\nNew file, {idx+1}/{number_files}\n")
        filename = file[:-4]  # Remove extension
        extract_features(path_in, filename, ext, crop_idx_dim1, verbose)


def save_reflectance_spectralon(use_path, crop_idx_dim1=1300):
    """
    Save the luminance value for all images for all bands in CSV files in a subfolder "csv"
    :param use_path: Global path where all hyperspectral images are
    :param crop_idx_dim1: index of the edge of the graph paper
    """
    path = os.path.join(use_path, "")
    all_files = next(walk(path), (None, None, []))[2]  # Detect only the files, not the folders
    hdr_files = [x for x in all_files if "hdr" in x]  # Extract hdr files

    for num, file in enumerate(hdr_files):
        print(f"\nImage progression: {num+1}/8\n")
        img = sp.open_image(os.path.join(path + file))
        # Retrieve values to convert grain image to reflectance
        array_ref = reflectance_grain(img, crop_idx_dim1-350, 1)  # -350 to remove graph paper, 1 for all bands

        # Save
        np.savetxt(os.path.join(path, "csv", "lum_spectralon_" + file[:-4] + ".csv"), array_ref, delimiter=",",
                   fmt='%1.1f')

