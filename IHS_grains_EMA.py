#!/usr/bin/env python3

# Author(s): T. Flutre - M. Ecarnot
# to be shared only with members of the PerfoMix project

# References:
# http://www.plantphysiol.org/lookup/doi/10.1104/pp.112.205120
# https://codegolf.stackexchange.com/questions/40831/counting-grains-of-rice

import sys

# sys.path.insert(0, "/home/ecarnot/Documents/INRA/Projets/perfomix/perfomixspectro/src/")
sys.path.append("C:/Users/seedmeister/Documents/Martin/perfomixspectro/src")

# sys.path.append("D:\PycharmProjects\Tasks")

# dependencies
# import os
# os.chdir("C:/Users/seedmeister/PycharmProjects/perfomix")
# print(sys.path)
# print("Current working directory: {0}".format(os.getcwd()))
import numpy as np
from numpy import matlib as mb
# import scipy as sci
import matplotlib.pyplot as plt
import cv2 as cv
# import spectral as sp
import spectral.io.envi as envi
from skimage.measure import label, regionprops  # , regionprops_table
import gzip
import time

# from gala import iterprogress
# from gala import morpho

t = time.time()
# PATH of hyperspectral images
# PATH =  'D:/2021/IRD-dates-2021/'  # 'F:/Q2/CHS/'  # 'D:\\2021\\perfomix\\'   #
PATH = 'D:/2021/perfomix/Recolte_2021/CHS/'
sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"
# # input parameters
cropIdxDim1 = 1000
thresh_refl = 0.15  # threshold of reflectance to remove background
thresh_lum_spectralon = 24500  # threshold of light intensity to remove background + milli
areaRange = 1000  # range of grain area in number of pixels
band = 100  # spectral band to extract (#100 : 681 nm)

img = envi.open(PATH + sImg + '.hdr', PATH + sImg + '.hyspex')

img = np.array(img.load(), dtype=np.int16)
img = np.transpose(img, (1, 0, 2))
imr = np.empty(img.shape, np.float32)

# Detect and extract spectralon
im0 = img[:, 1:cropIdxDim1, :]
#        ret0, binaryImage0 = cv.threshold(im0[:,:,band[0]]/im0[:,:,band[1]], thresh_lum_spectralon,1, cv.THRESH_BINARY)
ret0, binaryImage0 = cv.threshold(im0[:, :, band], thresh_lum_spectralon, 1, cv.THRESH_BINARY)
binaryImage0 = cv.erode(binaryImage0, np.ones((10, 10), np.uint8))
binaryImage0 = cv.morphologyEx(binaryImage0, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))

# Conversion to reflectance  : Essential for shape detection
ref = np.zeros((img.shape[0], img.shape[2]), img.dtype)
for x in range(0, img.shape[0]):
    nz = binaryImage0[x, :] != 0
    if sum(nz) > 50:
        ref[x, :] = np.mean(im0[x, nz, :], 0)
        imr[x, :, :] = img[x, :, :] / np.tile(ref[x, :], (img.shape[1], 1))

plt.imshow(imr[:, :, (80, 52, 15)])

# Reduce image to 1 section: To modify for each series of seeds
name_out = "_V-Mazafati_A-2016_R-Iran"  # "_V-DegletNour_A-2019_R-Tunisie"#"_V-Kentichi_A-2016_R-Tunisie"#"_V-Bournow_A-2019_R-Tchad"#"_V-PhoenixRoeb_A-2019_R-Espagne"#"_V-KhametFtimi_A-2016_R-Tunisie"#"_V-Khadrawy_A-2016_R-Emirats" #"_V-Mazafati_A-2016_R-Iran" #"_V-Khalass_A-2019_R-Emirats" #"_V-PhoenixCan_A-2020_R-MPL" #"_V-Zamili_A-2016_R-Emirats"  # "_V-Medjool_A-2019_R-Emirats" #"_V-Lulu_A-2016_R-Emirats"
colmin = 1400
colmax = 3600

# Grain detection and split close grains
im1 = imr[:, colmin:colmax, band]
ret, binaryImage = cv.threshold(im1, thresh_refl, 1, cv.THRESH_BINARY)
plt.imshow(binaryImage)
# opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN,  np.ones((10,10),np.uint8))
binaryImage = cv.morphologyEx(binaryImage, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))

labeled_array = label(binaryImage)
regions = regionprops(labeled_array)
plt.imshow(labeled_array)

o = [x for x in regions if x.area >= areaRange and x.solidity > 0.9]

# Save spectra and morph into file
attrok = ('area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent',
          'feret_diameter_max', 'filled_area', 'label', 'major_axis_length', 'minor_axis_length', 'orientation',
          'perimeter', 'perimeter_crofton', 'solidity')
sp = np.empty((0, img.shape[2] + 3)).astype(np.int16)  # np.empty((len(o),img.shape[2]))
morph = np.empty((len(o), len(attrok)))
imredu = img[:, colmin:colmax, :]
dep = np.reshape(imredu, (imredu.shape[0] * imredu.shape[1], imredu.shape[2]))

for i in range(0, len(o)):
    id = np.ravel_multi_index(np.transpose(o[i].coords),
                              (imredu.shape[0], imredu.shape[1]))  # coord of grains pixels in unfolded image
    sp1 = np.array([dep[j, :] for j in id]).astype(np.int16)
    # sp1 = np.array([dep[j,:] for j in id])
    spcoord = np.concatenate((mb.repmat(i + 1, len(id), 1), o[i].coords, sp1), axis=1).astype(np.int16)
    sp = np.concatenate((sp, spcoord))

    for j in range(0, len(attrok)):
        morph[i, j] = getattr(o[i], attrok[j])

# Save Spectra
f = gzip.GzipFile(PATH + sImg + name_out + "_sp.gz", "wb")
np.save(f, sp)
f.close()

# Save ref
f = gzip.GzipFile(PATH + sImg + name_out + "_ref.gz", "wb")
np.save(f, ref)
f.close()

# Save morpho values
morph = morph.astype(np.float32)
fmorph = gzip.GzipFile(PATH + sImg + name_out + "_morph.gz", "wb")
np.save(fmorph, morph)
fmorph.close()

elapsed = time.time() - t
