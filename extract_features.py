from preprocessing import *
import cv2
import statistics


def extract_features(path_in, filename, ext, crop_idx_dim1=1300, verbose=True):
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

    # The number of band is considered as a static number, 216
    n_bands = 216

    # Retrieve values to convert grain image to reflectance
    array_ref = reflectance_grain(img, crop_idx_dim1-350, 1)  # -350 to remove graph paper

    # Array to store features (order: min, max, median, mean, std)
    array_features = np.empty([len(arr_bbox), n_bands, 5])

    # Loop over all bbox detected with a smart progress meter (tqdm)
    for k in tqdm(range(len(arr_bbox)), desc="Extracting features"):

        box = arr_bbox[k]
        grain_img = img[box[1]:box[3], box[0]:box[2]].astype('float16')
        h, w = grain_img.shape[0], grain_img.shape[1]

        edge = box[0]  # Coordinate of the column in the original image

        for j in range(n_bands):
            new_grain = grain_img[:, :, j]
            # Reflectance
            for x in range(0, w):
                lum = array_ref[x+edge, j]
                new_grain[:, x] = new_grain[:, x] / lum if lum != 0 else [0] * h

            new_grain_mask = cv2.bitwise_and(new_grain, new_grain, mask=np.array(masks[k]).transpose())

            # Computing the average reflectance of the grain for one band
            x_size, y_size = new_grain_mask.size
            total_value = []
            for i in range(x_size):
                for t in range(y_size):
                    value = new_grain_mask[i, t]
                    if value != 0:
                        total_value.append(value)

            array_features[k, j] = [max(total_value), min(total_value), statistics.median(total_value),
                                    statistics.mean(total_value), statistics.stdev(total_value)]

    # Save
    use_path = os.path.join(path_in, "csv", "")
    np.savetxt(os.path.join(use_path, "features_grains_" + filename + ".csv"), array_features, delimiter=",",
               header="min, max, median, mean, std")






