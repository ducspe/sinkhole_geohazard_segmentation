import torch
import torchvision
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A  # used for data augmentation during training
from albumentations.pytorch import ToTensorV2

from deadsea_dataset import DeadSeaDataset
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers

import nni
import argparse
import os

from utils import calculate_metrics_v1, calculate_metrics_v2


# Server parameters section:
RUN_WITH_NNI = True 
device = 'cuda'
NUM_DEVICES = 1
NUM_NODES = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "../cloudcontainer/drone/full_dataset/train/images/"
TRAIN_MASK_DIR = "../cloudcontainer/drone/full_dataset/train/masks/"
VAL_IMG_DIR = "../cloudcontainer/drone/full_dataset/val/images/"
VAL_MASK_DIR = "../cloudcontainer/drone/full_dataset/val/masks/"
TEST_IMG_DIR = "../cloudcontainer/drone/full_dataset/test/images/"
TEST_MASK_DIR = "../cloudcontainer/drone/full_dataset/test/masks/"
TEST_DEMO_DIR = "../cloudcontainer/drone/saved_test_demos/"
VAL_DEMO_DIR = "../cloudcontainer/drone/saved_val_demos/"

"""
# Local parameters section for development purposes:
RUN_WITH_NNI = False
device = 'cpu'
NUM_DEVICES = 1
NUM_NODES = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "../cloudcontainer/drone/local_dataset_subset/train/images/"
TRAIN_MASK_DIR = "../cloudcontainer/drone/local_dataset_subset/train/masks/"
VAL_IMG_DIR = "../cloudcontainer/drone/local_dataset_subset/val/images/"
VAL_MASK_DIR = "../cloudcontainer/drone/local_dataset_subset/val/masks/"
TEST_IMG_DIR = "../cloudcontainer/drone/local_dataset_subset/test/images/"
TEST_MASK_DIR = "../cloudcontainer/drone/local_dataset_subset/test/masks/"
TEST_DEMO_DIR = "../cloudcontainer/drone/saved_test_demos/"
VAL_DEMO_DIR = "../cloudcontainer/drone/saved_val_demos/"
"""

class DoubleConv(pl.LightningModule):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(pl.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DeadSeaUNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, batch_size, instantiated_only_to_test_checkpoint=False):
        super(DeadSeaUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.instantiated_only_to_test_checkpoint = instantiated_only_to_test_checkpoint

        self.validation_loss_list = []
        self.validation_iou_list = []
        self.validation_dicescore_list = []
        
        self.test_loss_list = []
        self.test_iou_list = []
        self.test_dicescore_list = []
        self.test_acc_list = []

        self.indconv = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x1 = self.indconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits

    def setup(self, stage):
        # setup() runs on every GPU
        # Here you can do train/val splitting, and transformations, and local_dataset loading, etc.
        # As an exception from the general rule of assigning things in "init", it is acceptable to use "self" and assign things in this function as well.
        self.train_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                ToTensorV2(transpose_mask=False),  # For BCEWithLogitsLoss, the labels were one-hot encoded, so transpose_mask was True to transform them
                # from HWC to CHW format. For CrossEntropyLoss, the labels are not one-hot encoded, so the channel component does not exist. Hence,
                # no need to transpose it.
            ],
        )

        self.val_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                ToTensorV2(transpose_mask=False),
            ],
        )

        self.test_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                ToTensorV2(transpose_mask=False),
            ],
        )

        self.train_ds = DeadSeaDataset(
            image_dir=TRAIN_IMG_DIR,
            mask_dir=TRAIN_MASK_DIR,
            transform=self.train_transforms,
        )

        self.val_ds = DeadSeaDataset(
            image_dir=VAL_IMG_DIR,
            mask_dir=VAL_MASK_DIR,
            transform=self.val_transforms
        )

        self.test_ds = DeadSeaDataset(
            image_dir=TEST_IMG_DIR,
            mask_dir=TEST_MASK_DIR,
            transform=self.test_transforms
        )

        self.best_val_dice_score = torch.tensor([0])

    # PyTorch Lightning has lazy loading. The train_dataloader is called when need-be, not when I simply call init.
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=False
        )

        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=False
        )

        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        train_data, train_targets = batch

        train_logit_outputs = self(train_data)

        train_softmax_outputs = nn.functional.softmax(train_logit_outputs, dim=1)
        train_preds_argmax = torch.argmax(train_softmax_outputs, dim=1)

        train_J = self.loss(train_logit_outputs, train_targets)

        # Train metrics:
        train_iou, train_dice_score = calculate_metrics_v1(train_preds_argmax, train_targets, n_classes=self.n_classes, device=device)
        
        self.log('train_iou', train_iou)
        self.log('train_dice_score', train_dice_score)

        return {'loss': train_J}

    def validation_step(self, batch, batch_idx):
        val_data, val_targets = batch

        val_logit_outputs = self(val_data)
        #print("val_logit_outputs: ", val_logit_outputs[:, :, 0, 0])
        val_softmax_outputs = nn.functional.softmax(input=val_logit_outputs, dim=1)
        #print("val_softmax_outputs: ", val_softmax_outputs[:, :, 0, 0])
        val_preds_argmax = torch.argmax(val_softmax_outputs, dim=1)

        val_J = self.loss(val_logit_outputs, val_targets)
        self.validation_loss_list.append(val_J)

        # Validation metrics:
        val_iou, val_dice_score = calculate_metrics_v1(val_preds_argmax, val_targets, n_classes=self.n_classes, device=device)
        self.validation_iou_list.append(val_iou)
        self.validation_dicescore_list.append(val_dice_score)
        
        return {'loss': val_J, 'val_iou': val_iou, 'val_dice_score': val_dice_score}

    def test_step(self, batch, batch_idx):
        test_data, test_targets = batch

        test_logit_outputs = self(test_data)
        test_softmax_outputs = nn.functional.softmax(test_logit_outputs, dim=1)
        test_preds_argmax = torch.argmax(test_softmax_outputs, dim=1)

        test_J = self.loss(test_logit_outputs, test_targets)
        self.test_loss_list.append(test_J)

        # Test metrics:
        test_iou, test_dice_score = calculate_metrics_v1(test_preds_argmax, test_targets, n_classes=self.n_classes, device=device)
        test_acc = calculate_metrics_v2(test_preds_argmax, test_targets, n_classes=self.n_classes, device=device)
        self.test_iou_list.append(test_iou)
        self.test_dicescore_list.append(test_dice_score)
        self.test_acc_list.append(test_acc)

        # Save results as images for qualitative evaluation (tensors have to be float type to avoid errors when saving the images):
        if self.instantiated_only_to_test_checkpoint:
            print("Saving in the checkpoint_test_folder")
            os.makedirs(f"{TEST_DEMO_DIR}/checkpoint_test_folder", exist_ok=True)
            torchvision.utils.save_image(test_preds_argmax.float().unsqueeze(1), f"{TEST_DEMO_DIR}/checkpoint_test_folder/pred_test_{batch_idx}.png")
            torchvision.utils.save_image(test_targets.float().unsqueeze(1), f"{TEST_DEMO_DIR}/checkpoint_test_folder/target_test_{batch_idx}.png")
            torchvision.utils.save_image(test_data, f"{TEST_DEMO_DIR}/checkpoint_test_folder/testimg_{batch_idx}.png")
        elif RUN_WITH_NNI:
            print("Saving in the nni test folder")
            os.makedirs(f"{TEST_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}", exist_ok=True)
            torchvision.utils.save_image(test_preds_argmax.float().unsqueeze(1), f"{TEST_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}/pred_test_{batch_idx}.png")
            torchvision.utils.save_image(test_targets.float().unsqueeze(1), f"{TEST_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}/target_test_{batch_idx}.png")
        else:
            print("Saving in the non_nni_test_folder")
            os.makedirs(f"{TEST_DEMO_DIR}/non_nni_test_folder", exist_ok=True)
            torchvision.utils.save_image(test_preds_argmax.float().unsqueeze(1), f"{TEST_DEMO_DIR}/non_nni_test_folder/pred_test_{batch_idx}.png")
            torchvision.utils.save_image(test_targets.float().unsqueeze(1), f"{TEST_DEMO_DIR}/non_nni_test_folder/target_test_{batch_idx}.png")
            torchvision.utils.save_image(test_data, f"{TEST_DEMO_DIR}/non_nni_test_folder/testimg_{batch_idx}.png")

        return {'test_loss': test_J}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.tensor([x for x in self.validation_loss_list]).mean()
        avg_val_iou = torch.tensor([x for x in self.validation_iou_list]).mean()
        avg_val_dice_score = torch.tensor([x for x in self.validation_dicescore_list]).mean()
        
        self.log('avg_val_loss_per_epoch', avg_val_loss)
        self.log('avg_val_iou_per_epoch', avg_val_iou)
        self.log('avg_val_dice_score_per_epoch', avg_val_dice_score)

        nni.report_intermediate_result({"default": avg_val_dice_score.item(), 'avg_val_iou': avg_val_iou.item(), 'avg_val_loss': avg_val_loss.item()})

        if avg_val_dice_score.item() > self.best_val_dice_score.item():
            self.best_val_dice_score = avg_val_dice_score
            print("Found new best validation dice score equal to ", avg_val_dice_score.item())
            print("Saving new best checkpoint")

            self.trainer.save_checkpoint(f"../cloudcontainer/drone/saved_models/checkpoint_model_bs{self.batch_size}_ep{self.current_epoch}_valloss{avg_val_loss:.2f}_valdice{avg_val_dice_score:.2f}.ckpt")
            self.trainer.save_checkpoint(f"../cloudcontainer/drone/saved_models/best_model.ckpt")

        return {'val_loss': avg_val_loss, 'avg_val_dice_score': avg_val_dice_score, 'avg_val_iou': avg_val_iou}

    def on_train_end(self):
        print("Training is finished, the best validation dice score was: ", self.best_val_dice_score.item())
        print(self.trainer.callback_metrics)
        nni.report_final_result({"default": self.trainer.callback_metrics['avg_val_dice_score_per_epoch'].item()})

        # To see the qualitative results on the validation dataset so that we are able to compare with the analogous output for the test dataset (created by test_step())
        best_val_model = DeadSeaUNet.load_from_checkpoint("../cloudcontainer/drone/saved_models/best_model.ckpt", n_channels=args['in_channels'], n_classes=args['out_classes'], batch_size=args['batch_size'])

        for batch_idx, batch in enumerate(self.val_dataloader()):
            val_data, val_targets = batch
            val_targets = val_targets.unsqueeze(dim=1)
            val_logit_outputs = best_val_model(val_data)
            val_preds_int = torch.argmax(val_logit_outputs, dim=1).unsqueeze(1)

            if RUN_WITH_NNI:
                os.makedirs(f"{VAL_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}", exist_ok=True)
                torchvision.utils.save_image(val_preds_int.float(), f"{VAL_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}/pred_val_{batch_idx}.png")
                torchvision.utils.save_image(val_targets.float(), f"{VAL_DEMO_DIR}/{nni.get_experiment_id()}/{nni.get_trial_id()}/target_val_{batch_idx}.png")
            else:
                os.makedirs(f"{VAL_DEMO_DIR}/non_nni_val_folder", exist_ok=True)
                torchvision.utils.save_image(val_preds_int.float(), f"{VAL_DEMO_DIR}/non_nni_val_folder/pred_val_{batch_idx}.png")
                torchvision.utils.save_image(val_targets.float(), f"{VAL_DEMO_DIR}/non_nni_val_folder/target_val_{batch_idx}.png")

    def on_test_end(self):
        avg_test_loss = torch.tensor([x for x in self.test_loss_list]).mean()
        avg_test_iou = torch.tensor([x for x in self.test_iou_list]).mean()
        avg_test_dice_score = torch.tensor([x for x in self.test_dicescore_list]).mean()
        avg_test_acc_score = torch.vstack(self.test_acc_list).mean(dim=0)

        print('avg_test_loss', avg_test_loss.item())
        print('avg_test_dice_score', avg_test_dice_score.item())
        print('avg_test_iou', avg_test_iou.item())
        print("Per class test dataset accuracy: ", avg_test_acc_score)

def main(args):
    pl.seed_everything(34)  # for reproducibility

    model = DeadSeaUNet(n_channels=args['in_channels'], n_classes=args['out_classes'], batch_size=args['batch_size'])
    mylogger = pl_loggers.TensorBoardLogger(save_dir=args['logs_dir'])
    if NUM_DEVICES > 1:
        trainer = pl.Trainer(max_epochs=args['num_epochs'], devices=NUM_DEVICES, num_nodes=NUM_NODES, logger=mylogger, accelerator='ddp')
    else:
        trainer = pl.Trainer(max_epochs=args['num_epochs'], devices=NUM_DEVICES, num_nodes=NUM_NODES, logger=mylogger)

    trainer.fit(model)

    # Qualitative evaluation of the best model:
    test_model = DeadSeaUNet.load_from_checkpoint("../cloudcontainer/drone/saved_models/best_model.ckpt", n_channels=args['in_channels'], n_classes=args['out_classes'], batch_size=args['batch_size'])
    trainer.test(test_model)


if __name__ == '__main__':
    os.makedirs("../cloudcontainer/drone/saved_models", exist_ok=True)
    parser = argparse.ArgumentParser()
    if RUN_WITH_NNI:
        params = nni.get_next_parameter()  # this function is called once per nni trial. It gets the hyperparameters out of the pool specified in the
                                           # search space file
        print("nni section params: ", params)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--out_classes', type=int, default=3)
        parser.add_argument('--logs_dir', type=str, default='../cloudcontainer/drone')
        args_dict = vars(parser.parse_args())
        args_dict.update(params)
        args = vars(argparse.Namespace(**args_dict))
        main(args)
    else:
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--out_classes', type=int, default=3)
        # For out_classes in case of binary semantic segmentation I might set default=1 or default=2.
        # If I set default=1, I have the option of using the sigmoid(logit) > 0.5 approach.
        # If I set default=2, I will have to use the argmax approach.
        # To extend semantic segmentation to 3 classes I will have to use the argmax approach, not the sigmoid.

        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--num_epochs', type=int, default=3)
        parser.add_argument('--logs_dir', type=str, default='../cloudcontainer/drone')
        args = vars(parser.parse_args())
        print("non-nni section args: ", args)
        main(args)