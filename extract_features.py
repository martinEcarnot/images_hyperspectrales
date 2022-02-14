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
        print(f"\nImage progression: {num+1}/{len(hdr_files)}\n")
        img = sp.open_image(os.path.join(path + file))
        # Retrieve values to convert grain image to reflectance
        array_ref = reflectance_grain(img, crop_idx_dim1-350, 1)  # -350 to remove graph paper, 1 for all bands

        # Save
        np.savetxt(os.path.join(path, "csv", "lum_spectralon_" + file[:-4] + ".csv"), array_ref, delimiter=",",
                   fmt='%1.1f')


def display_features(path_in, feature, nb_grain_figure=0):
    """
    Display the features for each image
    Order for all file: max, min, median, mean, std
    Shape: ( number of grains detected, 216, 5 )

    :param path_in: path of the folder of the hyperspectral image ( ex: "D:\\Etude technique" )
    :param feature: Feature to choose from max, min, median, mean, std
    :param nb_grain_figure: Number of grain to display. Set to 0 for all of them
    """
    use_path = os.path.join(path_in, "csv")
    all_files = next(walk(use_path), (None, None, []))[2]  # Detect only the files, not the folders
    npy_files = [x for x in all_files if x[-3:] == "npy"]  # Extract npy files
    # number_files = len(npy_files)
    order = ["max", "min", "median", "mean", "std"]
    idx_feature = order.index(feature)

    fig, axs = plt.subplots(2, 4, figsize=(17, 9))

    for idx, file in enumerate(npy_files):
        path = os.path.join(use_path, file)
        data = np.load(path)

        list_grain = []  # Will store list of a feature for each grain
        number_grain = data.shape[0]
        number_bands = data.shape[1]
        nb_grain = nb_grain_figure if nb_grain_figure != 0 else number_grain
        grain_step = number_grain // nb_grain
        for grain in range(nb_grain):
            list_tmp = []
            for i in range(number_bands):
                list_tmp.append(data[grain * grain_step, i, idx_feature])
            list_grain.append(list_tmp)

        x_axis = range(number_bands)
        row = 0 if idx < 4 else 1
        for k in list_grain:
            axs[row, idx % 4].plot(x_axis, k)
        # Determination of the year and the variety
        variety = file.split("_")[2].split("-")
        var = variety[0] if "var" in variety[0] else variety[1]
        year = file.split("_")[6].split("-")[0]

        axs[row, idx % 4].set_title(f'{var}_{year}')
        axs[row, idx % 4].set_xlabel('Bands')
        axs[row, idx % 4].set_ylabel('Reflectance')
        axs[row, idx % 4].grid()
        # axs[row, idx % 4].legend()

    fig.suptitle(f"{feature} of each grain of each image per band")  # Global title
    fig.tight_layout()
    plt.show()
    # fig.savefig(os.path.join(figure_path, name_figure+".png"), dpi=200, format='png')


def display_boxplot(path_in, number_of_the_band=150):
    """
    Display the comparison of boxplot for a given band for all images

    :param path_in: path of the folder of the hyperspectral image ( ex: "D:\\Etude technique" )
    :param number_of_the_band: The band to realize the boxplot
    """
    use_path = os.path.join(path_in, "csv")
    all_files = next(walk(use_path), (None, None, []))[2]  # Detect only the files, not the folders
    npy_files = [x for x in all_files if x[-3:] == "npy"]  # Extract npy files

    mini, maxi, mu, sigma, name = [], [], [], [], []
    for idx, file in enumerate(npy_files):
        path = os.path.join(use_path, file)
        data = np.load(path)
        number_grain = data.shape[0]
        mini_tmp, maxi_tmp, mu_tmp, sigma_tmp = 0, 0, 0, 0
        for i in range(number_grain):
            mini_tmp += data[i, number_of_the_band, 1]
            maxi_tmp += data[i, number_of_the_band, 0]
            mu_tmp += data[i, number_of_the_band, 3]
            sigma_tmp += data[i, number_of_the_band, 4]
        mini.append(mini_tmp/number_grain)
        maxi.append(maxi_tmp/number_grain)
        mu.append(mu_tmp/number_grain)
        sigma.append(sigma_tmp/number_grain)

        # Determination of the year and the variety
        variety = file.split("_")[2].split("-")
        var = variety[0] if "var" in variety[0] else variety[1]
        year = file.split("_")[6].split("-")[0]
        position = file.split("-")
        pos_tmp = position[0] if "x" in position[0] else position[1]
        pos_tmp = pos_tmp.split("_")
        pos = pos_tmp[0] if "x" in pos_tmp[0] else pos_tmp[2]
        name.append(f'{pos}_{var}_{year}')

    # create stacked errorbars:
    plt.errorbar(np.arange(8), mu, sigma, fmt='ok', lw=3)
    plt.errorbar(np.arange(8), mu, [[mu_i - mini_i for mu_i, mini_i in zip(mu, mini)],
                                    [maxi_i - mu_i for maxi_i, mu_i in zip(maxi, mu)]],
                 fmt='.k', ecolor='gray', lw=1)
    list_nb = range(len(sigma))
    for a, b in zip(list_nb, mu):
        plt.text(a+0.07, b-0.005, str(round(b, 2)), ha='left')
    for a, b in zip(list_nb, mini):
        plt.text(a+0.07, b-0.005, str(round(b, 2)), ha='left')
    for a, b in zip(list_nb, maxi):
        plt.text(a+0.07, b-0.005, str(round(b, 2)), ha='left')

    plt.xticks(list_nb, name)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Name image')
    plt.ylabel('Reflectance')
    plt.title(f"Boxplot of reflectance per image for the band {number_of_the_band}")
    plt.tight_layout()
    plt.show()
