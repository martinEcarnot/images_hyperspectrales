# from preprocessing import preprocessing
# from crop_img import crop_image
# import matplotlib.pyplot as plt
# import spectral as sp
from deep_learning import *
from torch.utils.data import DataLoader

# Set path
PATH = "D:/Etude technique/"
# PATH = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/'

# Select file name
# file = 'var8-x75y12_7000_us_2x_2021-10-20T113607_corr'
file = "x30y21-var1_11000_us_2x_2020-12-02T095609_corr"

"""
# Preprocessing only

# sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"
# array_bbox_, masks = preprocessing(PATH, sImg)
# print(array_bbox_)
# print(array_bbox_.shape)
# print(len(masks))

# Crop_image with plot

PATH_OUT = PATH + file + '/'
ext = '.hdr'  # '.hyspex'

thresh_lum_spectralon_ = 8000
crop_idx_dim1_ = 1200
crop_image(PATH, PATH_OUT, file, ext, thresh_lum_spectralon=thresh_lum_spectralon_, crop_idx_dim1=crop_idx_dim1_,
           band_step=20, apply_mask=True, force_creation=True)

file = 'grain1'
img = sp.open_image(PATH_OUT + file + ext)
print(img.shape)
img0 = img[:, :, 5]
plt.imshow(img0, cmap='gray')
plt.show()

"""

use_path_ = "D:\\Etude technique"

df_path = load(use_path_)
# print(df_path.columns)
# print(df_path)


# Create datasets
train_set = CustomDataset(df_path)
# valid_set = CustomDataset(df_path_valid)
# test_set = CustomDataset(df_path_test)

# Create data loaders
batch_size = 32
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
for files, labels in trainloader:
    print(files)
    print("LABELS :", labels)
# validloader = DataLoader(valid_set, batch_size = batch_size, shuffle = True)
# testloader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
