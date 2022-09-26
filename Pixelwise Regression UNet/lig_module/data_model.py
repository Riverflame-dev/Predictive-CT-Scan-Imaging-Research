import random
from pathlib import Path
from typing import List, Optional
import glob
import os
import numpy as np
import pytorch_lightning as pl
import torch
from monai.transforms import Compose, Randomizable, apply_transform
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from monai.data import Dataset

from utils.const import DATA_ROOT
from utils.transforms import get_label_transforms, get_train_img_transforms, get_val_img_transforms
from monai.utils import first

import numpy as np
import monai
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandAffined,
    Resized,
    SpatialPadd,
    ToTensord,
    AddChanneld,
    ScaleIntensityRanged,
    CropForeground,
    LoadImaged,
    ScaleIntensityd,
    CropForegroundd
)
from monai.transforms.compose import Transform

import matplotlib.pyplot as plt

# I/O Directories
INPUT_DIR = "D:\\EDDIE_D\\Multichannel-input-pixelwise-regression-u-nets\\data\\"
# Volume Names & Sample Names
VOLUME_NAMES = ['S2020', 'S3020', 'S4020', 'S5020']
#SAMPLE_NAMES = ['S13430', 'S13860', 'S13940', 'S14260', 'S14650', 'S14790', 'S14850', 'S14870', 'S14890', 'S13990'] 

SAMPLE_NAMES = ['S13430', 'S13860', 'S13940', 'S14260'] 

class Unsqueeze(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.expand_dims(img, axis=0)

class DataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int, X_image: str, y_image: str
    ):
        super().__init__()
        self.batch_size = batch_size
        self.X_image = X_image
        self.y_image = y_image
        self.train_dataset = []
        self.val_dataset = []
        self.orig_dataset = []

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:

        for VOLUME in VOLUME_NAMES:
            if VOLUME == 'S2020': channel1 = sorted(glob.glob(os.path.join(INPUT_DIR, VOLUME, "*.nii.gz")))
            if VOLUME == 'S3020': channel2 = sorted(glob.glob(os.path.join(INPUT_DIR, VOLUME, "*.nii.gz")))
            if VOLUME == 'S4020': channel3 = sorted(glob.glob(os.path.join(INPUT_DIR, VOLUME, "*.nii.gz")))
            if VOLUME == 'S5020': targets3d = sorted(glob.glob(os.path.join(INPUT_DIR, VOLUME, "*.nii.gz")))
            print("All " + VOLUME + " volumes imported!")

        train_files_3d = [{"channel1": c1, "channel2": c2, "channel3": c3, "targets3d": t3d} 
                            for c1, c2, c3, t3d in zip(channel1, channel2, channel3, targets3d)]

        print("Number of images imported: ", 4*len(train_files_3d))

        # Define transforms
        train_transforms = Compose(
        [
            LoadImaged(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
            AddChanneld(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
            ScaleIntensityRanged(keys=['channel1', 'channel2', 'channel3', 'targets3d'], 
                                 a_min=0, a_max=209, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=['channel1', 'channel2', 'channel3', 'targets3d'], 
                        spatial_size=[512, 512, 100], method="symmetric", mode="constant"),
            CropForegroundd(keys=['channel1', 'channel2', 'channel3', 'targets3d'], source_key='channel1'),
            Resized(keys=['channel1', 'channel2', 'channel3', 'targets3d'], spatial_size=[128, 128, 128]),
            ToTensord(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
        ]
        )
        orig_transforms = Compose(
        [
            LoadImaged(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
            AddChanneld(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
            Resized(keys=['channel1', 'channel2', 'channel3', 'targets3d'], spatial_size=[128, 128, 128]),
            ToTensord(keys=['channel1', 'channel2', 'channel3', 'targets3d']),
        ]
        )

        # Apply transform
        self.orig_dataset = Dataset(data=train_files_3d, transform = orig_transforms)
        self.train_dataset_3d = Dataset(data=train_files_3d, transform = train_transforms)
        #self.val_dataset = Dataset(data=train_files_3d, transform = train_transforms)

        print("transformed size: ", self.train_dataset_3d[0]['channel1'].shape)

        channel1_slices = []
        channel2_slices = []
        channel3_slices = []
        train_target = []

        for data in self.train_dataset_3d:
            for i in range(128):
                ch1, ch2, ch3 = data['channel1'][0, : ,: ,i], data['channel2'][0, : ,: ,i], data['channel3'][0, : ,: ,i]
                channel1_slices.append(ch1)
                channel2_slices.append(ch2)
                channel3_slices.append(ch3)
                trgt = data['targets3d'][0, : ,: ,i]
                train_target.append(trgt)
                # TEST
                print("Number of slices: ", len(channel1_slices)) 

        train_image = []
        
        for i in range(len(channel1_slices)):
            # Create a blank image that has three channels and the same number of pixels as your original input
            multi_channel_img = torch.zeros((3, 128, 128))
            # Add the channels to the needed image one by one
            multi_channel_img[0, :, :] = channel1_slices[i]
            multi_channel_img[1, :, :] = channel2_slices[i]
            multi_channel_img[2, :, :] = channel3_slices[i]
            train_image.append(multi_channel_img)

        n = (int)(len(channel1_slices)*0.9)
        m = len(channel1_slices) - n
        print("Training images: ", n)
        print("Validating images: ", m)

        train_files = [{"image": image, "target": target} for image, target in zip(train_image[:n], train_target[:n])]
        val_files = [{"image": image, "target": target} for image, target in zip(train_image[n+1:], train_target[n+1:])]
        
        self.train_dataset = Dataset(data=train_files)
        self.val_dataset = Dataset(data=val_files)

        # TEST
        train_loader = self.train_dataloader()
        orig_loader = self.orig_dataloader()
        transformed = first(train_loader)
        original = first(orig_loader)
        print("original: ")
        print(original['channel1'].shape)
        print("Channel1: ", original['channel1'].min(), original['channel1'].max())
        print("Channel2: ", original['channel2'].min(), original['channel2'].max())
        print("Channel3: ", original['channel3'].min(), original['channel3'].max())
        print("transformed: ")
        print(transformed['image'].shape)
        print(transformed['image'].min(), transformed['image'].max())

    def orig_dataloader(self):
        return DataLoader(
            self.orig_dataset,
            batch_size=1,  #self.batch_size
            #collate_fn=monai.data.utils.pad_list_data_collate
        )

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 2D image!")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 2D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)  #num_workers = 8

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 2D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0) #num_workers = 8
