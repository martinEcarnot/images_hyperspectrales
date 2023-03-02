"""
Retrieve the brightest band for each image to do the detection
"""
from os import getcwd
from os.path import join
from brightest_band import *
from display_image import display_img

path_img = "img/"
path_defauts = "img/csv/liste_defauts_grains.csv"
# retrieve_all_brightest_bands_to_csv(use_path)

from crop_img import *
crop_all_images(path_img, path_defauts)

"""
from classifcation_face import *
annotations_folder = "img/cropped/"
learning_rate = 1e-4
epochs = 20
weight_loss = [2., 2.]
main_loop(annotations_folder, weight_loss, learning_rate, epochs=20, batch_size=12)
"""
"""
Function to save values of the spectralon for each band for the reflectance and 
save into a 'csv' folder
"""

# from extract_features import *

# extract_all_features(use_path)
# save_reflectance_spectralon(use_path)


"""
Display and retrieve the image after the detection of bounding boxes and masks
"""

#sImg = "var3_2020_x77y15_8000_us_2x_2022-04-27T085658_corr"
#array_bbox_, masks = preprocessing(use_path, sImg)

"""
Example to crop a single image
"""
# from crop_img import *
# file = 'var1-x75y20_7000_us_2x_2021-10-19T160916_corr'
# PATH_OUT = PATH + file + '\\'
# ext = '.hdr'  # '.hyspex'
# crop_image(PATH, PATH_OUT, file, ext, band_step=108, force_creation=True, apply_mask=True)


"""
Automatically detect and extract grain into a subfolder called 'csv' given a band step
"""




"""
Extraction of features of all images. Retrieve and save max, min, mean, median, std into a 'csv' file
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# extract_all_features(use_path)


"""
Display of an image with 8 graphs given a parameter (max, min, mean, median, std) to compare reflectances
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# display_features(use_path, "max", nb_grain_figure=0)

"""
Same as previously but with a boxplot
"""
# from extract_features import *
# path_in = "D:\\Etude technique"
# display_boxplot(use_path)

"""
Get the boxplot to compare size of grain for each image
"""
# use_path_ = "D:\\Etude technique"
# get_grain_size(use_path)


"""
Deep learning training, testing, saving model (optimizer and loss directly 
                                               defined inside the function)
"""
"""
from deep_learning import *
#
training_path = join(use_path,"3_bands")
learning_rate = 1e-4
epochs = 20
weight_loss = [2., 2.]
#
main_loop(training_path, weight_loss, learning_rate, epochs=epochs, batch_size=12)
"""