import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral.io.envi as envi
from skimage.measure import label, regionprops


def show_image(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def preprocessing(folder_path, s_img, crop_idx_dim1=1000, thresh_refl=0.15, thresh_lum_spectralon=22000, band=100,
                  area_range=1000):
    """
    Take an image and create sub-images for each grain

    :param folder_path: PATH of hyperspectral images
    :param s_img: name of a hyperspectral image
    :param crop_idx_dim1:
    :param thresh_refl: threshold of reflectance to remove background
    :param thresh_lum_spectralon: threshold of light intensity to remove background + milli
    :param band: spectral band to extract (#100 : 681 nm)
    :param area_range: range of grain area in number of pixels
    :return: array of bbox of all grains
    """
    img = envi.open(folder_path + s_img + '.hdr', folder_path + s_img + '.hyspex')
    img = np.array(img.read_band(band), dtype=np.int16)
    img = np.transpose(img)
    imr = np.empty(img.shape, np.float32)
    # show_image(img)

    # Detect and extract spectralon
    im0 = img[:, 1:crop_idx_dim1]
    # show_image(im0)

    ret0, binary_image0 = cv.threshold(im0, thresh_lum_spectralon, 1, cv.THRESH_BINARY)
    # show_image(binary_image0)

    binary_image0 = cv.erode(binary_image0, np.ones((10, 10), np.uint8))
    binary_image0 = cv.morphologyEx(binary_image0, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    # show_image(binary_image0)

    # Conversion to reflectance: Essential for shape detection
    ref = np.zeros((img.shape[0]), img.dtype)
    for x in range(0, img.shape[0]):
        nz = binary_image0[x, :] != 0
        if sum(nz) > 50:
            ref[x] = np.mean(im0[x, nz], 0)
            imr[x, :] = img[x, :] / np.tile(ref[x], (img.shape[1]))
    # show_image(imr)

    colmin = 1400  # Reduce image to remove spectralon

    # Grain detection and split close grains
    im1 = imr[:, colmin:]
    ret, binary_image = cv.threshold(im1, thresh_refl, 1, cv.THRESH_BINARY)
    # show_image(binary_image)

    binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    labeled_array = label(binary_image)
    # show_image(labeled_array)

    regions = regionprops(labeled_array)
    list_bbox = [x.bbox for x in regions if x.area >= area_range and x.solidity > 0.9]
    # list_bbox = list_bbox + colmin
    result = img.copy()
    # result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    for bbox in list_bbox:
        x1, y1, x2, y2 = bbox
        cv.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(result)
    plt.show()

    return np.array(list_bbox)


PATH = "D:/Etude technique/"
sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"

array_bbox = preprocessing(PATH, sImg)
# print(array_bbox)
print(array_bbox.shape)
