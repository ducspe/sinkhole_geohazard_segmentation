import numpy as np
import os
from patchify import patchify
import tifffile as tiff
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import splitfolders

tile_size = 256
step= 128
interractive_mode = False
year = 2022
month = "_8_" 

image_path = "cloudcontainer/satellite/original_data/2022_8_image-002.tif"
mask_path = "cloudcontainer/satellite/original_data/2022_8_mask_py.tif"
output_path = f"cloudcontainer/satellite/sat_output_patches/{year}/"

folder_path = os.path.join(output_path, f"patches{month}")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
output_path = f"cloudcontainer/satellite/sat_output_patches/{year}/patches{month}/"

for folder in ["nonsinkhole_patches/images", "nonsinkhole_patches/masks", "sinkhole_patches/images", "sinkhole_patches/masks"]:
    folder_path = os.path.join(output_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_image(image_path):
    image =  tiff.imread(image_path)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return image

def drop_extra_bands(image, mask):
    num_mask_bands = 0
    if mask.ndim == 3:
        num_mask_bands = mask.shape[-1]
    num_image_bands = image.shape[-1]
    if num_mask_bands == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    if mask.ndim == 2:
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    if num_image_bands > 3:
        image = image[:, :, :3]

    return image, mask

def image_correction(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    b, g, r = cv2.split(image)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    image = cv2.merge((b_eq, g_eq, r_eq))
    del b, g, r,b_eq, g_eq, r_eq

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_mean, b_mean = cv2.mean(a)[0], cv2.mean(b)[0]
    a_adj = 128 - a_mean
    b_adj = 128 - b_mean
    a_new = cv2.addWeighted(a, 1, np.zeros(a.shape, a.dtype), 0, a_adj)
    b_new = cv2.addWeighted(b, 1, np.zeros(b.shape, b.dtype), 0, b_adj)
    lab_new = cv2.merge((l, a_new, b_new))
    del lab, l, a, b, a_mean, b_mean, a_adj, b_adj, a_new, b_new
    image = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
    del lab_new

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value = 255 - v.max()
    v_new = np.clip(v + value, 0, 255)
    hsv_new = cv2.merge((h, s, v_new))
    image = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    
    return image

def convert_mask_to_binary(mask):
    img = np.array(mask)
    unique_vals, counts = np.unique(img, return_counts=True)
    none_sinkhole_value = unique_vals[np.argmax(counts)]
    for i in range(len(unique_vals)):
        if counts[i] > 0.2 * img.size:
            background_value = unique_vals[i]
            mask[mask == background_value] = 0
    mask[mask == none_sinkhole_value] = 0
    mask[mask != 0] = 255
    return mask

def is_single_color(img, threshold=0.95, span=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    max_index = np.argmax(hist)
    close_colors_percent = (hist[max_index] + np.sum(hist[max_index-span:max_index+span])) / np.sum(hist)

    if close_colors_percent > threshold:
        return True
    else:
        return False

def save_tiles(image, mask, tile_size=256):
    resize_height = int(np.floor(image.shape[0] / tile_size) * tile_size)
    resize_width = int(np.floor(image.shape[1] / tile_size) * tile_size)
    resized_input_img = image[0:resize_height, 0:resize_width, :]
    resized_label_img = mask[0:resize_height, 0:resize_width, :]
    
    img_patches = patchify(resized_input_img, (tile_size, tile_size, 3), step=step)
    label_patches = patchify(resized_label_img, (tile_size, tile_size, 1), step=step)
    
    for i in tqdm(range(img_patches.shape[0])):
        for j in range(img_patches.shape[1]):
            tile_image = img_patches[i, j, 0, :, :, :]
            tile_mask = label_patches[i, j, 0, :, :, :]
            if np.sum(tile_mask) == 0:
                single_color = is_single_color(tile_image, .80, 8)
            else:
                single_color = is_single_color(tile_image, .95, 2)
            
            if not single_color:
                if interractive_mode == True and j%10 == 0 :
                    plt.imshow(superimpose(tile_image, tile_mask))
                    plt.show
                if np.sum(tile_mask) == 0:
                    tiff.imsave(output_path + f"nonsinkhole_patches/images/image_{i}_{j}.tif", tile_image)
                    tiff.imsave(output_path + f"nonsinkhole_patches/masks/mask_{i}_{j}.tif", tile_mask)
                else:
                    if interractive_mode and j % 10 == 0 :
                        plt.imshow(superimpose(tile_image, tile_mask))
                        plt.show()
                    tiff.imsave(output_path + f"sinkhole_patches/images/image_{i}_{j}.tif", tile_image)
                    tiff.imsave(output_path + f"sinkhole_patches/masks/mask_{i}_{j}.tif", tile_mask)

def superimpose(image, mask):
    background = image
    overlay = cv2.merge([mask , mask, mask])
    alpha = 0.35
    result = cv2.addWeighted(background, 1-alpha, overlay, alpha, 0)
    return result

def cropimg(img, x=20, y=30, b = 2):
    img_crop = img [255 * (x-b):255 * (x+b), 255 * (y-b):255 * (y+b), :]
    return img_crop

image = read_image(image_path)
mask = read_image(mask_path)

image, mask = drop_extra_bands(image, mask)
image = image_correction(image)
cv2.imwrite("cloudcontainer/satellite/original_data/correctedsatimage.tif", image)

mask = convert_mask_to_binary(mask)
cv2.imwrite("cloudcontainer/satellite/original_data/correctedmaskimage.tif", mask)

save_tiles(image, mask, tile_size=256)

print("\n","\n","\n","\033[1m" + f"Tiles saved successfully {year}{month} !" + "\033[0m","\n","\n","\n")

splitfolders.ratio('cloudcontainer/satellite/sat_output_patches/2022/patches_8_/sinkhole_patches', output="cloudcontainer/satellite/full_dataset", seed=70, ratio=(0.8, 0.1, 0.1))
splitfolders.ratio('cloudcontainer/satellite/sat_output_patches/2022/patches_8_/nonsinkhole_patches', output="cloudcontainer/satellite/full_dataset", seed=70, ratio=(0.8, 0.1, 0.1))