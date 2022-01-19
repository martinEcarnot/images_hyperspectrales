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


def reflectance(image, crop_idx_dim1, thresh_lum_spectralon, verbose=False):
    """
    Conversion to reflectance
    :param image: original image
    :param crop_idx_dim1: index of the edge of the spectralon
    :param thresh_lum_spectralon: threshold of light intensity to remove background + milli
    :param verbose: Display figure if set to True
    :return: reflectance image
    """
    imr = np.empty(image.shape, np.float32)

    # Detect and extract spectralon
    im0 = image[:, 1:crop_idx_dim1]
    ret0, binary_image0 = cv.threshold(im0, thresh_lum_spectralon, 1, cv.THRESH_BINARY)
    binary_image0 = cv.erode(binary_image0, np.ones((10, 10), np.uint8))
    binary_image0 = cv.morphologyEx(binary_image0, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Conversion to reflectance: Essential for shape detection
    ref = np.zeros((image.shape[0]), image.dtype)
    for x in range(0, image.shape[0]):
        nz = binary_image0[x, :] != 0
        if sum(nz) > 50:
            ref[x] = np.mean(im0[x, nz], 0)
            imr[x, :] = image[x, :] / np.tile(ref[x], (image.shape[1]))

    if verbose:
        fig, axes = plt.subplots(ncols=2, figsize=(9, 3))
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Original image')
        ax[1].imshow(imr, cmap="gray")
        ax[1].set_title('Reflectance image')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()
    return imr


def preprocessing(folder_path, s_img, crop_idx_dim1=1000, thresh_refl=0.15, thresh_lum_spectralon=22000, band=100,
                  area_range=1000, verbose=1):
    """
    Take an image and create sub-images for each grain

    :param folder_path: PATH of hyperspectral images
    :param s_img: name of a hyperspectral image
    :param crop_idx_dim1: index of the edge of the spectralon
    :param thresh_refl: threshold of reflectance to remove background
    :param thresh_lum_spectralon: threshold of light intensity to remove background + milli
    :param band: spectral band to extract (#100 : 681 nm)
    :param area_range: minimum area to be considered, in pixels
    :param verbose: display the image with bbox
    :return: array of bbox of all grains, list of masks for all grains
    """
    img = envi.open(folder_path + s_img + '.hdr', folder_path + s_img + '.hyspex')
    img = np.array(img.read_band(band), dtype=np.int16)
    img = np.transpose(img)
    # Conversion to reflectance
    imr = reflectance(img, crop_idx_dim1, thresh_lum_spectralon, verbose=True)
    exit()
    colmin = 1300  # Reduce image to remove spectralon

    # Grain detection and split close grains
    im1 = imr[:, colmin:]
    ret, binary_image = cv.threshold(im1, thresh_refl, 1, cv.THRESH_BINARY)
    # show_image(binary_image)

    # Better result without morph_close
    # binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    labeled_array = label(binary_image)
    regions = regionprops(labeled_array)

    list_bbox = [x.bbox for x in regions if 12000 >= x.area >= area_range and x.solidity > 0.9]
    # the max area find was around 11000. Two grains close can have a solidity > 0.9 so a upper bound must be set.

    # In addition to bbox, the mask is retrieved
    list_mask = [x.convex_image.astype(np.uint8) for x in regions if 12000 >= x.area >= area_range and x.solidity > 0.9]

    # create_lists_areas(regions, area_range)  # Display the list of area sizes in pixels

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
        labeled_array_bar = label(binary_image_bar)
        regions_bar = regionprops(labeled_array_bar)
        # + 3000 to remove partial grains (bbox can include sliced grains)
        list_regions_bar = np.array([x for x in regions_bar if x.area >= area_range + 3000 and x.solidity > 0.9])
        array_bbox_bar_res = np.array([x.bbox for x in list_regions_bar])

        # Creation of the mask for each grain by using the convex_image
        # list_mask.append(create_mask(im_bar, array_bbox_bar_res, list_regions_bar))  # First solution
        for x in list_regions_bar:
            mask = x.convex_image.astype(np.uint8)
            list_mask.append(mask)

        # The bbox coordinates must be set for the global image, not locally
        for i in range(len(array_bbox_bar_res)):
            array_bbox_bar_res[i][0] = array_bbox_bar_res[i][0] + list_bbox_bar[ind][0]
            array_bbox_bar_res[i][1] = array_bbox_bar_res[i][1] + list_bbox_bar[ind][1]
            array_bbox_bar_res[i][2] = array_bbox_bar_res[i][2] + list_bbox_bar[ind][0]
            array_bbox_bar_res[i][3] = array_bbox_bar_res[i][3] + list_bbox_bar[ind][1]
        list_bar = [*list_bar, *array_bbox_bar_res]  # Full list with all bbox

    # list_bbox = list_bar
    list_bbox = [*list_bbox, *list_bar]
    array_bbox = np.array(list_bbox)

    # Add colmin to y values
    for i in range(len(array_bbox)):
        array_bbox[i][1] = array_bbox[i][1] + colmin
        array_bbox[i][3] = array_bbox[i][3] + colmin

    # Display
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

    return array_bbox, list_mask


def create_lists_areas(regions, area_range):
    """
    Display the sorted list of all areas given the condition x.area >= area_range and x.solidity > 0.9.
    Moreover, check if two grains have been considered as one by printing areas > 11000 pixels

    :param regions: Measured properties of labeled image regions
    :param area_range: minimum area to be considered, in pixels
    """
    list_area = []
    list_size = []
    for x in regions:
        if x.area >= area_range and x.solidity > 0.9:
            list_area.append(x.area)
            if x.area > 11000:
                list_size.append(x.bbox)
    print(sorted(list_area, reverse=True))
    print("List of area > 11000 pixels: ", list_size)


def create_mask(im_bar, array_bbox_bar_res, list_regions_bar):
    """
    Compute a mask to retrieve only the grain of an image

    :param im_bar: image delimited by the bbox
    :param array_bbox_bar_res: array of bbox
    :param list_regions_bar: List of measured properties of labeled image regions
    :return: a list of binary masks
    """
    list_mask = []
    for idx, x in enumerate(list_regions_bar):
        img_tmp = im_bar[array_bbox_bar_res[idx][0]:array_bbox_bar_res[idx][2],
                         array_bbox_bar_res[idx][1]:array_bbox_bar_res[idx][3]]
        show_image(img_tmp)
        mask = np.zeros(img_tmp.shape, dtype=np.uint8)
        for coord in x.coords:
            x, y = coord
            x -= array_bbox_bar_res[idx][0]
            y -= array_bbox_bar_res[idx][1]
            mask[x][y] = 1
        result = cv.bitwise_and(img_tmp, img_tmp, mask=mask)
        show_image(result)
        list_mask.append(mask)
    return list_mask


def watershed():
    """
    Function to try watershed algorithm
    """
    img = envi.open("D:\\Etude technique\\var8-x75y12_7000_us_2x_2021-10-20T113607_corr" + '.hdr',
                    "D:\\Etude technique\\var8-x75y12_7000_us_2x_2021-10-20T113607_corr" + '.hyspex')
    img = np.array(img.read_band(100), dtype=np.int16)
    img = np.transpose(img)
    imr = reflectance(img, 1300, 8000)

    colmin = 1300  # Reduce image to remove spectralon

    # Grain detection and split close grains
    im1 = imr[:, colmin:]
    ret, binary_image = cv.threshold(im1, 0.15, 1, cv.THRESH_BINARY)
    # show_image(binary_image)

    # Watershed part

    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    # https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    from gala.morpho import watershed

    new_img = watershed(binary_image)
    plt.imshow(new_img)
    plt.show()
