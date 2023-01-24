# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:21:49 2023

@author: hadri
"""

from skimage import data, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

img = util.img_as_ubyte(data.coins()) >120
plt.imshow(img)
label_img = label(img, connectivity=img.ndim)
props = regionprops(label_img)
# centroid of first labeled object
#♥print(props[72].centroid)
# centroid of first labeled object
centr_coord = [prop.centroid for prop in props]
print(centr_coord)
