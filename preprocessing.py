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
    # binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    labeled_array = label(binary_image)
    # show_image(labeled_array)
    regions = regionprops(labeled_array)

    list_bbox = [x.bbox for x in regions if 12000 >= x.area >= area_range and x.solidity > 0.9]

    # Check if two grains have been considered as one (the area is considered for comparison)
    # list_area = []
    # list_size = []
    # for x in regions:
    #     if x.area >= area_range and x.solidity > 0.9:
    #         list_area.append(x.area)
    #         if x.area > 11000:
    #             list_size.append(x.bbox)
    # print(sorted(list_area, reverse=True))

    # Case if solidity and size (upper bound) conditions are not met
    list_bbox_bar = [x.bbox for x in regions if (x.solidity < 0.9 and x.area >= area_range)
                     or (x.solidity > 0.9 and x.area > 12000)]
    list_bar = []
    # With the bbox of groups of grains, the image is retrieved for each one, then labeled and region properties are
    # determined. The aim is to separate grain but to retrieve only useful pixels, not overlapping areas.
    for ind in range(len(list_bbox_bar)):
        im_bar = imr[list_bbox_bar[ind][0]:list_bbox_bar[ind][2],
                     list_bbox_bar[ind][1] + colmin:list_bbox_bar[ind][3] + colmin]  # Select the image inside the bbox
        ret_bar, binary_image_bar = cv.threshold(im_bar, thresh_refl + 0.1, 1, cv.THRESH_BINARY)  # Higher threshold
        # binary_image_bar = cv.morphologyEx(binary_image_bar, cv.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        labeled_array_bar = label(binary_image_bar)
        regions_bar = regionprops(labeled_array_bar)
        list_regions_bar = np.array([x for x in regions_bar if x.area >= area_range and x.solidity > 0.9])  # + 3500
        array_bbox_bar_res = np.array([x.bbox for x in list_regions_bar])
        for idx, x in enumerate(list_regions_bar):
            # show_image(x.convex_image)  # filled_image, convex_image
            # print(x.coords)
            # show_image(im_bar)
            img_tmp = im_bar[array_bbox_bar_res[idx][0]:array_bbox_bar_res[idx][2],
                             array_bbox_bar_res[idx][1]:array_bbox_bar_res[idx][3]]
            # show_image(img_tmp)
            # print(x.convex_image.astype(np.uint8))
            mask = x.convex_image.astype(np.uint8)
            # mask = np.zeros(img_tmp.shape, dtype=np.uint8)
            # for coord in x.coords:
            #     x, y = coord
            #     x -= array_bbox_bar_res[idx][0]
            #     y -= array_bbox_bar_res[idx][1]
            #     mask[x][y] = 1
            result = cv.bitwise_and(img_tmp, img_tmp, mask=mask)
            #show_image(result)
        #exit()

        # The bbox coordinates must be set for the global image, not locally
        for i in range(len(array_bbox_bar_res)):
            array_bbox_bar_res[i][0] = array_bbox_bar_res[i][0] + list_bbox_bar[ind][0]
            array_bbox_bar_res[i][1] = array_bbox_bar_res[i][1] + list_bbox_bar[ind][1]
            array_bbox_bar_res[i][2] = array_bbox_bar_res[i][2] + list_bbox_bar[ind][0]
            array_bbox_bar_res[i][3] = array_bbox_bar_res[i][3] + list_bbox_bar[ind][1]
        list_bar = [*list_bar, *array_bbox_bar_res]

    # print(array_bbox_bar_res)

    # list_bbox = list_size
    # list_bbox = list_bar
    list_bbox = [*list_bbox, *list_bar]
    array_bbox = np.array(list_bbox)

    # Add colmin to y values
    for i in range(len(array_bbox)):
        array_bbox[i][1] = array_bbox[i][1] + colmin
        array_bbox[i][3] = array_bbox[i][3] + colmin

    if verbose:
        fig, ax = plt.subplots()
        ax.xaxis.tick_top()
        result = img
        ax.imshow(result)

        for bbox in array_bbox:
            x1, y1, x2, y2 = bbox
            ax.add_patch(patches.Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, edgecolor='red', lw=2))
        plt.imshow(result, cmap="gray")
        plt.show()

    return array_bbox


def remove_white_area(array_bbox):
    return 0


PATH = "D:/Etude technique/"
PATH = 'C:/Users/kiera/Documents/EMA/3A/2IA/Image/ET/'
sImg = "var8-x75y12_7000_us_2x_2021-10-20T113607_corr"

print('hello')
array_bbox_ = preprocessing(PATH, sImg)
print(array_bbox_)
print(array_bbox_.shape)

