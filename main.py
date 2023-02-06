
"""
Retrieve the brightest band for each image to do the detection
"""
# from brightest_band import *
# use_path = "D:\\Etude technique"
# retrieve_all_brightest_bands_to_csv(use_path)


"""
Function to save values of the spectralon for each band for the reflectance and 
save into a 'csv' folder
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# save_reflectance_spectralon(path_in, crop_idx_dim1=1300)


"""
Display and retrieve the image after the detection of bounding boxes and masks
"""
# PATH = "D:\\Etude technique\\"
# sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"
# # sImg = "x30y21-var1_11000_us_2x_2020-12-02T095609_corr"
# centroid_coord, array_bbox_, masks = preprocessing(PATH, sImg, thresh_lum_spectralon=20000)


"""
Example to crop a single image
"""
# from crop_img import *
# PATH = "D:\\Etude technique\\"
# file = 'var1-x75y20_7000_us_2x_2021-10-19T160916_corr'
# PATH_OUT = PATH + file + '\\'
# ext = '.hdr'  # '.hyspex'
# crop_image(PATH, PATH_OUT, file, ext, band_step=108, force_creation=True, apply_mask=True)


"""
Automatically detect and extract grain into a subfolder called 'csv' given a band step
"""
# from crop_img import *
# use_path_ = "D:\\Etude technique"
# crop_all_images(use_path_, band_step_=72, verbose=False)


"""
Extraction of features of all images. Retrieve and save max, min, mean, median, std into a 'csv' file
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# extract_all_features(path_in)


"""
Display of an image with 8 graphs given a parameter (max, min, mean, median, std) to compare reflectances
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# display_features(path_in, "mean", nb_grain_figure=0)

"""
Same as previously but with a boxplot
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# display_boxplot(path_in)

"""
Get the boxplot to compare size of grain for each image
"""
from crop_img import *
# use_path_ = "D:\\Etude technique"
# get_grain_size(use_path_)


"""
Deep learning training, testing, saving model (optimizer and loss directly 
                                               defined inside the function)
"""
# from deep_learning import *
#
# use_path = "D:\\Etude technique\\21_bands\\"
# learning_rate = 1e-4
# epochs = 20
# weight_loss = [2., 2.]
#
# main_loop(use_path, weight_loss, learning_rate, epochs=epochs, batch_size=12)

from preprocessing import *

Path = 'img/'
#file = "var1_2020_x75y20_8000_us_2x_2022-04-26T122543_corr"
#file = "var1_2020_x75y20_8000_us_2x_2022-04-26T130045_corr"
file = "var4_2020_x82y12_8000_us_2x_2022-04-27T092007_corr"
#file = 'var4_2020_x82y12_8000_us_2x_2022-04-27T093216_corr'
preprocessing(Path, file)

#crop_image(Path, Path + 'cropped/', file, ext = '.hdr', force_creation=True, 
           #sillon = True, liste_grains_defauts = [0, 1, 15, 31, 52, 53, 55, 59, 67])
