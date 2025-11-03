import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

import os
import yaml

from loguru import logger
from rich.console import Console
from rich.progress import Progress

class GTSRBDataset(Dataset):
    '''
    Dataset for GTRSB to read images from folder 
    '''

    def __init__(self, root, transform):
        
        console = Console()
        logger.info("Save classes paths and labels")

        self.root = root
        self.transfomr = transform
        
        # get sorted classes names
        classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        self.paths = []
        self.labels = []
        
        with Progress() as progress:
            task = progress.add_task("Saving image paths and class labels", total=len(classes))
            for cls_name in classes:
                
                cls_idx = self.class_to_idx[cls_name]
                cls_folder = os.path.join(self.root, cls_name)

                for fname in sorted(os.listdir(cls_folder)):
                    self.paths.append(os.path.join(cls_folder, fname))
                    self.labels.append(cls_idx)
                progress.advance(task)

    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])

        img = cv2.imread(img_path)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]

        return img_tensor, label
    
def get_albumentations_transform(img_size: int = 64, is_train = True) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomCrop(img_size, img_size, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    
    # read parameters
    params = yaml.safe_load(open('params.yaml'))['preprocess']
    root = params['out_root']
    img_size = params['img_size']
    batch_size = params['batch_size']
    num_workers = params['num_workers']

    logger.info("Create train and validation datasets")
    # create custom train dataset
    train_ds = GTSRBDataset(root=os.path.join(root, 'train'), 
                            transform=get_albumentations_transform(img_size, is_train=True))
    
    # create custom validation dataset
    val_ds = GTSRBDataset(root=os.path.join(root, "val"),
                          transform=get_albumentations_transform(img_size, is_train=False))
    
    logger.info("Create train and validation dataloaders")
    # create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

if __name__ == '__main__':
    logger.info("Start gtsrb_dataset.py")
    tr, da = get_dataloaders()
    logger.success("Complete gtsrb_dataset.py")