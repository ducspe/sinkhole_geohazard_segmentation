import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class DeadSeaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", "mask"))
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # L because we have grayscale. L stands for Luminance.

        # Normalization rules:
        image = image / 255.0
        mask[mask == 128] = 1
        mask[mask == 255] = 2
        # if 2 in mask:
        #     print("normalized mask: ", np.unique(mask))
        # End of normalization rules

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        #print("mask shape: ", mask.shape)
        return image, mask.long()