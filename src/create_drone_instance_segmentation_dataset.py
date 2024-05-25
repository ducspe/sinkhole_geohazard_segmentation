import cv2
import numpy as np
import os
from patchify import patchify
import tifffile as tiff
import splitfolders
from sklearn.utils import class_weight

#####################################################
# Script parameters:
INTERACTIVE_MODE = False
TILE_HEIGHT = 256
TILE_WIDTH = 256
PIXEL_SKIP_STEP = 128


#####################################################
input_img = cv2.imread("../cloudcontainer/drone/original_data/OF2016Res10cmAcc60cm_2.tif")
print("Original drone image shape: ", input_img.shape)

label_img = cv2.imread("../cloudcontainer/drone/original_data/Sinkhole_labels_Poly_4.tif")
print("Original drone label shape: ", label_img.shape)

##########################################
# My custom grayscale:

img_red_channel = label_img[:, :, 0]
img_green_channel = label_img[:, :, 1]
img_blue_channel = label_img[:, :, 2]

gamma = 1.6
red_const = 0.2126
green_const = 0.7152
blue_const = 0.0722


custom_grayscale = red_const * img_red_channel ** gamma + green_const * img_green_channel ** gamma + blue_const * img_blue_channel ** gamma

##########################################
# My custom edge detection procedure based on the Sobel filtering idea:
filter_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
filter_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

rows, cols = custom_grayscale.shape
print("rows, cols = ", rows, cols)


if os.path.exists("../cloudcontainer/drone/edges_between_sinkholes.npy"):
    with open("../cloudcontainer/drone/edges_between_sinkholes.npy", "rb") as fin:
        filtered_img = np.load(fin)
else:
    filtered_img = np.zeros_like(custom_grayscale)
    for i in range(3, rows-2):
        for j in range(3, cols-2):
            print(f"row {i}, col {j}")
            # First do the convolution/cross-correlation with both filters and compute a gradient value for both x and y direction
            if np.any(custom_grayscale[i:i+3, j:j+3] == 0):  # == 0 meaning I detected black background
                # Do not compute edge between black background and sinkhole.
                # The idea is to compute an edge between two touching sinkholes only (in the else branch)
                # Later on I will dilate this edge and mark it as a class 3 segment signifying the transition region between two sinkholes.
                print("Background detected in the patch => No edge will be computed")
                filtered_img[i+1, j+1] = 0
            else:
                gx = np.sum(np.multiply(filter_x, custom_grayscale[i:i+3, j:j+3]))
                gy = np.sum(np.multiply(filter_y, custom_grayscale[i:i+3, j:j+3]))

                # Now use pythagorean formula to come up with the magnitude of the gradient:
                filtered_img[i+1, j+1] = np.sqrt(gx ** 2 + gy ** 2)

    with open("../cloudcontainer/drone/edges_between_sinkholes.npy", "wb") as fout:
        np.save(fout, filtered_img)

# End of my custom edge detection procedure.
##########################################

# Dilation code:
kernel = np.ones((3, 3), np.uint8)
filtered_img_int = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # very important, to be able to remove close to zero floats, that are not actually part of the edge.

filtered_image_mask_greater_than_zero = filtered_img_int > 0
filtered_image_mask_equal_to_zero = filtered_img_int == 0
print("filtered_img_int > 0: ", filtered_img_int[filtered_image_mask_greater_than_zero].shape)
print("filtered_img_int == 0: ", filtered_img_int[filtered_image_mask_equal_to_zero].shape)
filtered_img_int[filtered_image_mask_greater_than_zero] = 255

dilated_edges = cv2.dilate(filtered_img_int, kernel, iterations=2)
dilated_edge_mask = dilated_edges > 0
dilated_edges_binary = np.where(dilated_edge_mask, 128.0, 0)

flat_dilated_edges_binary = dilated_edges_binary.reshape(-1)
print("dilated edges unique values: ", np.unique(flat_dilated_edges_binary))

# End of dilation code.

# Generate the binary map of the label image, where foreground consists of sinkhole pixels and the rest of the pixels are considered black background:
cv2_gray_label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
binary_label_threshold, binary_label_img = cv2.threshold(cv2_gray_label_img, 0, 255, cv2.THRESH_BINARY)  # Note: Otsu had to be removed because it was ignoring labels with dark color.
print("binary_label_threshold=", binary_label_threshold)
print("binary label unique values: ", np.unique(binary_label_img))

# End binary label map generation.

# Superimpose dilated edges on label and binary label maps:
dilated_edges_superimposed_on_original_label = label_img.copy()
dilated_edges_superimposed_on_original_label[dilated_edge_mask] = 255

threeclass_label_img = binary_label_img.copy()
threeclass_label_img[dilated_edge_mask] = 128
print("final map unique labels: ", np.unique(threeclass_label_img, return_counts=True))
print("reshaped final map: ", threeclass_label_img.reshape(-1))

# Next compute class weights to be used during training
# This can be useful since our dataset is unbalanced. For this next line to work make sure RAM and Swap file is big enough.
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(threeclass_label_img, return_counts=False), y=threeclass_label_img.reshape(-1))
print("class_weights=", class_weights)
# End of class weight computation.

# End of superimpose code.
cv2.imwrite("../cloudcontainer/drone/saved_full_images/threeclass_img_from_originallabel.jpg", threeclass_label_img)
cv2.imwrite("../cloudcontainer/drone/saved_full_images/original_img.jpg", input_img)
cv2.imwrite("../cloudcontainer/drone/saved_full_images/label_img.jpg", label_img)
if INTERACTIVE_MODE:
    cv2.namedWindow("Drone RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone RGB", 600, 600)
    cv2.imshow("Drone RGB", input_img)

    cv2.namedWindow("Drone label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone label", 600, 600)
    cv2.imshow("Drone label", label_img)

    cv2.namedWindow("Binary label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary label", 600, 600)
    cv2.imshow("Binary label", binary_label_img)

    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale", 600, 600)
    cv2.imshow("Grayscale", cv2_gray_label_img)

    cv2.namedWindow("Edges between sinkholes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edges between sinkholes", 600, 600)
    cv2.imshow("Edges between sinkholes", filtered_img_int)

    cv2.namedWindow("Dilated edges binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dilated edges binary", 600, 600)
    cv2.imshow("Dilated edges binary", dilated_edges_binary)

    cv2.namedWindow("Dilated edges superimposed on original label image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dilated edges superimposed on original label image", 600, 600)
    cv2.imshow("Dilated edges superimposed on original label image", dilated_edges_superimposed_on_original_label)

    cv2.namedWindow("Dilated edges superimposed on binary label image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dilated edges superimposed on binary label image", 600, 600)
    cv2.imshow("Dilated edges superimposed on binary label image", threeclass_label_img)

    cv2.waitKey(0)


resize_height = int(np.floor(input_img.shape[0] / TILE_HEIGHT) * TILE_HEIGHT)
print("Resizing height to fit the specified TILE_HEIGHT is: ", resize_height)
resize_width = int(np.floor(input_img.shape[1] / TILE_WIDTH) * TILE_WIDTH)
print("Resizing height to fit the specified TILE_WIDTH is: ", resize_width)

resized_input_img = input_img[0:resize_height, 0:resize_width]
print("resized_input_img shape: ", resized_input_img.shape)
img_patches = patchify(resized_input_img, (TILE_HEIGHT, TILE_WIDTH, 3), step=PIXEL_SKIP_STEP)
print("img_patches shape: ", img_patches.shape)

resized_threeclass_label_img = threeclass_label_img[0:resize_height, 0:resize_width]
print("resized_threeclass_label_img: ", resized_threeclass_label_img.shape)
threeclass_label_patches = patchify(resized_threeclass_label_img, (TILE_HEIGHT, TILE_WIDTH), step=PIXEL_SKIP_STEP)
print("threeclass_label_patches: ", threeclass_label_patches.shape)

non_black_image_counter = 0
for i in range(img_patches.shape[0]):
    for j in range(img_patches.shape[1]):
        single_img_patch = img_patches[i, j, 0, :, :]
        single_threeclass_label_patch = threeclass_label_patches[i, j, :, :]

        if INTERACTIVE_MODE:
            cv2.namedWindow("Patch", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Patch", 600, 600)
            print("one_img_patch shape:", single_img_patch.shape)
            cv2.imshow("Patch", single_img_patch)

            cv2.namedWindow("Three class label patch", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Three class label patch", 600, 600)
            print("threeclass_label_patch shape:", single_threeclass_label_patch.shape)
            cv2.imshow("Three class label patch", single_threeclass_label_patch)

            cv2.waitKey(0)

        if single_threeclass_label_patch.flatten().sum() != 0:
            tiff.imwrite("../cloudcontainer/drone/sinkhole_patches/images/" + 'image_' + str(i) + '_' + str(j) + '.tif', single_img_patch)
            tiff.imwrite("../cloudcontainer/drone/sinkhole_patches/masks/" + 'mask_' + str(i) + '_' + str(j) + '.tif', single_threeclass_label_patch)
            non_black_image_counter += 1
        else:
            tiff.imwrite("../cloudcontainer/drone/nonsinkhole_patches/images/" + 'image_' + str(i) + '_' + str(j) + '.tif', single_img_patch)
            tiff.imwrite("../cloudcontainer/drone/nonsinkhole_patches/masks/" + 'mask_' + str(i) + '_' + str(j) + '.tif', single_threeclass_label_patch)

splitfolders.ratio('../cloudcontainer/drone/sinkhole_patches', output="../cloudcontainer/drone/full_dataset", seed=70, ratio=(0.8, 0.1, 0.1))
splitfolders.ratio('../cloudcontainer/drone/nonsinkhole_patches', output="../cloudcontainer/drone/full_dataset", seed=70, ratio=(0.8, 0.1, 0.1))