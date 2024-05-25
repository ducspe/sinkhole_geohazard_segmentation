import pytorch_lightning as pl
import numpy as np
import cv2
from patchify import patchify
import torch
from pytorch_lightning import loggers as pl_loggers
from deadsea_unet_train import DeadSeaUNet

# Parameters section:
RUN_ON_LAPTOP = False  # when resources are a problem, resize the original image so that it doesn't take too long to predict all the patches.
INTERACTIVE_MODE = False  # on the server make this False, because there is no GUI
TILE_HEIGHT = 256
TILE_WIDTH = 256
PIXEL_SKIP_STEP = 256  # for train dataset this was 128, which meant overlapping images. Here I must not overlap, so 256 is correct.
TEST_NUM_DEVICES = 1
NUM_NODES = 1

args = {
    'in_channels': 3,
    'out_classes': 3,
    'batch_size': 64,
    'logs_dir': '../cloudcontainer/logsfrom_predict_full_picture_dot_py'
}
test_checkpoint_path = "../cloudcontainer/satellite/transfer_learning_models/tr_best_fine_tuned_model.ckpt"
best_model_path = "../cloudcontainer/satellite/transfer_learning_models/tr_best_fine_tuned_model.ckpt"
mylogger = pl_loggers.TensorBoardLogger(save_dir=args['logs_dir'])
input_img = cv2.imread("../cloudcontainer/satellite/original_data/correctedsatimage.tif")

#####################################################################################################

if RUN_ON_LAPTOP:
    input_img = cv2.resize(input_img, (1600, 1600))

print("Original image shape: ", input_img.shape)

reconstructed_img = np.zeros_like(input_img)

resize_height = int(np.floor(input_img.shape[0] / TILE_HEIGHT) * TILE_HEIGHT)
print("Resizing height to fit the specified TILE_HEIGHT is: ", resize_height)
resize_width = int(np.floor(input_img.shape[1] / TILE_WIDTH) * TILE_WIDTH)
print("Resizing height to fit the specified TILE_WIDTH is: ", resize_width)

resized_input_img = input_img[0:resize_height, 0:resize_width]
print("resized_input_img shape: ", resized_input_img.shape)
img_patches = patchify(resized_input_img, (TILE_HEIGHT, TILE_WIDTH, 3), step=PIXEL_SKIP_STEP)
print("img_patches shape: ", img_patches.shape)

trainer = pl.Trainer(devices=TEST_NUM_DEVICES, num_nodes=NUM_NODES, logger=mylogger)
inference_model = DeadSeaUNet.load_from_checkpoint(checkpoint_path=best_model_path, n_channels=args['in_channels'], n_classes=args['out_classes'], batch_size=args['batch_size'])
inference_model.eval()

for i in range(img_patches.shape[0]):
    for j in range(img_patches.shape[1]):
        print(f"i={i}, j={j}")
        img_hwc = img_patches[i, j, 0, :, :, :]
        print("img_hwc patch shape: ", img_hwc.shape)
        img_chw = np.transpose(img_hwc, axes=[2, 0, 1])
        print("img_chw", img_chw.shape)
        img_chw_normalized = img_chw / 255.0

        input_tensor = torch.from_numpy(img_chw_normalized).float().unsqueeze(0)
        img_out = inference_model(input_tensor)
        print("img_out: ", img_out.shape)
        prediction = torch.argmax(img_out, dim=1)
        print("prediction: ", prediction.shape)

        prediction_hwc = np.transpose(prediction.numpy(), axes=[1, 2, 0])
        print("prediction_hwc: ", prediction_hwc.shape)
        prediction_final = prediction_hwc.squeeze(2)

        threechannel_gray_patch_prediction = np.zeros((256, 256, 3))

        threechannel_gray_patch_prediction[:, :, 0] = prediction_final
        threechannel_gray_patch_prediction[:, :, 1] = prediction_final
        threechannel_gray_patch_prediction[:, :, 2] = prediction_final

        reconstructed_img[TILE_WIDTH*i:TILE_WIDTH*i + TILE_WIDTH, TILE_HEIGHT*j:TILE_HEIGHT*j + TILE_HEIGHT, :] = threechannel_gray_patch_prediction


reconstructed_img[reconstructed_img == 1] = 128
reconstructed_img[reconstructed_img == 2] = 255
print("reconstructed_img: ", reconstructed_img.shape)

cv2.imwrite("../cloudcontainer/satellite/saved_full_images/sat_full_semantic_segmentation_prediction_image.jpg", reconstructed_img)

if INTERACTIVE_MODE:
    cv2.namedWindow("reconstructed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("reconstructed", 600, 600)
    cv2.imshow("reconstructed", reconstructed_img)
    cv2.waitKey(0)