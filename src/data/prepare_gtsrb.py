import yaml
import os, shutil

from sklearn.model_selection import train_test_split

from loguru import logger
from rich.console import Console
from rich.progress import Progress


def prepare_gtsrb():

    console = Console()

    console.rule("[bold blue] Step 1: Loading and splitting the dataset")
    
    # read parameters
    params = yaml.safe_load(open('params.yaml'))['preprocess']

    raw_root = params['raw_root']
    out_root = params['out_root']
    val_size = params['val_size']
    seed = params['seed']

    # start logging the preparing process
    logger.info(f"Train-Validation split, val_size={val_size}") 

    # create dirs
    os.makedirs(out_root, exist_ok=True)

    out_train_root = os.path.join(out_root, 'train')
    out_val_root = os.path.join(out_root, 'val')

    os.makedirs(out_train_root, exist_ok=True)
    os.makedirs(out_val_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'test'), exist_ok=True)

    train_path = os.path.join(raw_root, 'Train')
    
    # read Test floder
    all_data = []
    with Progress() as progress:

        class_folders = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        logger.info(f"{len(class_folders)} classes was found")

        # create progress bar
        task = progress.add_task("[cyan] Copying images...", total = len(class_folders))


        for class_folder in class_folders:
            # get path of current class
            class_path = os.path.join(train_path, class_folder)

            os.makedirs(os.path.join(out_train_root, class_folder), exist_ok=True)
            os.makedirs(os.path.join(out_val_root, class_folder), exist_ok=True)

            if not os.path.isdir(class_path):
                continue
                
            # get a list of images in current folder
            imgs = os.listdir(class_path)

            # split images in current class
            train_imgs, val_images = train_test_split(imgs, test_size=val_size, random_state=seed)

            # copy every image in data/processed
            for fname in train_imgs:
                shutil.copy(os.path.join(class_path, fname), f"{out_root}/train/{class_folder}/{fname}")

            for fname in val_images:
                shutil.copy(os.path.join(class_path, fname), f"{out_root}/val/{class_folder}/{fname}")

            progress.advance(task)
        
    logger.success(f"The division was complete")


if __name__ == "__main__":
    prepare_gtsrb()
