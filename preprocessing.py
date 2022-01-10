import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral.io.envi as envi
from skimage.measure import label, regionprops
import matplotlib.patches as patches


def show_image(img):
    """
    Display the image in gray

    :param img: the image to display
    """
    plt.imshow(img, cmap="gray")
    plt.show()


def preprocessing(folder_path, s_img, crop_idx_dim1=1000, thresh_refl=0.15, thresh_lum_spectralon=22000, band=100,
                  area_range=1000, verbose=1):
    """
    Take an image and create sub-images for each grain

    :param folder_path: PATH of hyperspectral images
    :param s_img: name of a hyperspectral image
    :param crop_idx_dim1: indexe of the edge of the spectralon
    :param thresh_refl: threshold of reflectance to remove background
    :param thresh_lum_spectralon: threshold of light intensity to remove background + milli
    :param band: spectral band to extract (#100 : 681 nm)
    :param area_range: range of grain area in number of pixels
    :param verbose: display the image with bbox
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

    colmin = 1300  # Reduce image to remove spectralon

    # Grain detection and split close grains
    im1 = imr[:, colmin:]
    ret, binary_image = cv.threshold(im1, thresh_refl, 1, cv.THRESH_BINARY)
    # show_image(binary_image)

    # Better result without morph_close
    # binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # thresh_image = binary_image.astype(np.uint8)
    # contours, hierarchy = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # plt.imshow(cv.drawContours(im1, contours, -1, (255, 0, 0), 3))
    # plt.show()

    labeled_array = label(binary_image)
    # show_image(labeled_array)
    regions = regionprops(labeled_array)

    list_bbox = [x.bbox for x in regions if x.area >= area_range and x.solidity > 0.9]

    # Case if solidity condition is not met
    list_bbox_bar = [x.bbox for x in regions if x.solidity < 0.9 and x.area >= area_range]
    ind = 0
    print(list_bbox_bar[ind])
    im_bar = imr[list_bbox_bar[ind][0]:list_bbox_bar[ind][2], list_bbox_bar[ind][1]+colmin:list_bbox_bar[ind][3]+colmin]
    ret_bar, binary_image_bar = cv.threshold(im_bar, thresh_refl, 1, cv.THRESH_BINARY)
    binary_image_bar = cv.erode(binary_image_bar, np.ones((20, 20), np.uint8))
    labeled_array_bar = label(binary_image_bar)
    plt.imshow(labeled_array_bar)
    plt.show()
    exit()

    list_bbox = list_bbox + list_bbox_bar
    list_bbox = np.array(list_bbox)
    # Add colmin to y values
    for i in range(len(list_bbox)):
        list_bbox[i][1] = list_bbox[i][1] + colmin
        list_bbox[i][3] = list_bbox[i][3] + colmin

    if verbose:
        fig, ax = plt.subplots()
        ax.xaxis.tick_top()
        result = img
        ax.imshow(result)

        for bbox in list_bbox:
            x1, y1, x2, y2 = bbox
            ax.add_patch(patches.Rectangle((y1, x1), y2-y1, x2-x1, fill=False, edgecolor='red', lw=2))
        plt.imshow(result, cmap="gray")
        plt.show()

    return list_bbox


PATH = "D:/Etude technique/"
sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"

array_bbox = preprocessing(PATH, sImg)
# print(array_bbox)
print(array_bbox.shape)
