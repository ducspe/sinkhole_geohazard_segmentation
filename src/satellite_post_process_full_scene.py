import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

# Script parameters:
INTERACTIVE_MODE = False
NUM_EROSION_ITERATIONS = 0
NUM_DILATION_ITERATIONS = 1
# End of script parameters section

threeclass_label_img = cv2.imread("../cloudcontainer/satellite/saved_full_images/sat_full_semantic_segmentation_prediction_image.jpg", cv2.IMREAD_GRAYSCALE)
print("Input shape: ", threeclass_label_img.shape)

# Section to connect the edge tails from both sides, separated by a gap:

processing_kernel = np.ones((5, 5), np.uint8)
processed_threeclass_label_img = cv2.erode(threeclass_label_img, processing_kernel, iterations=NUM_EROSION_ITERATIONS)
processed_threeclass_label_img = cv2.dilate(processed_threeclass_label_img, processing_kernel, iterations=NUM_DILATION_ITERATIONS)
# Note that erosion followed by dilation is actually the opening morphological operation:
# processed_threeclass_label_img = cv2.morphologyEx(threeclass_label_img, cv2.MORPH_OPEN, processing_kernel, iterations=10)

# End of section that connects edge tails from both sides, separated by a gap.

sure_fg = (processed_threeclass_label_img == 255).astype(np.uint8) * 255
print("unique foreground pixels: ", np.unique(sure_fg))

ret, label_instances = cv2.connectedComponents(sure_fg)
print("labels: ", label_instances)
unique_label_indices = np.unique(label_instances)
print("unique labels: ", unique_label_indices)


# Potential optional ToDO: Find a better color scheme formula such that the nearby sinkholes don't have similar colors:
# my_colors = [(round(color_intensity / np.max(unique_label_indices), 5) ** 3, 1 - round(color_intensity / np.max(unique_label_indices), 5) ** 0.5, round(color_intensity / np.max(unique_label_indices), 5) ** 2) for color_intensity in unique_label_indices]
# print("my_colors", my_colors)
# base_colors = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

# Section to create more random looking color maps:
rng = np.random.default_rng(seed=43529)
unique_colors = unique_label_indices.copy()
rng.shuffle(unique_colors)  # pseudo-random shuffling since I fixed the seed above.
unique_colors = unique_colors / np.max(unique_colors)
print("random color indices: ", unique_colors)

my_cmap = matplotlib.cm.get_cmap('gist_rainbow')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html#miscellaneous
my_random_colors = [my_cmap(color_intensity)[:3] for color_intensity in unique_colors]
print("random colors: ", my_random_colors)

# End of section to create more random looking color maps.

labels_colored = color.label2rgb(label_instances, bg_label=0, colors=my_random_colors)

cv2.imwrite("../cloudcontainer/satellite/saved_full_images/sat_full_instance_segmentation_prediction_image.jpg", labels_colored * 255.0)

if INTERACTIVE_MODE:
    plt.imshow(label_instances, cmap='gray')
    plt.show()

    cv2.namedWindow("Three class label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Three class label", 600, 600)
    cv2.imshow("Three class label", threeclass_label_img)

    cv2.namedWindow("Processed three class label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Processed three class label", 600, 600)
    cv2.imshow("Processed three class label", processed_threeclass_label_img)

    cv2.namedWindow("Sure foreground", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sure foreground", 600, 600)
    cv2.imshow("Sure foreground", sure_fg)

    cv2.namedWindow("Connected components", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Connected components", 600, 600)
    cv2.imshow("Connected components", label_instances / np.max(label_instances) * 255)

    cv2.namedWindow("Labels colored", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labels colored", 600, 600)
    cv2.imshow("Labels colored", labels_colored)

    cv2.waitKey(0)