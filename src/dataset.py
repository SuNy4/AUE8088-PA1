# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages
#import src.config as cfg

class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(self.cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=self.cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        tf_train = transforms.Compose([
            transforms.RandomRotation(self.cfg.IMAGE_ROTATION),
            transforms.RandomHorizontalFlip(self.cfg.IMAGE_FLIP_PROB),
            transforms.RandomCrop(self.cfg.IMAGE_NUM_CROPS, padding=self.cfg.IMAGE_PAD_CROPS),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.IMAGE_MEAN, self.cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.IMAGE_MEAN, self.cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            #shuffle validation dataset for batch-wise metric calc.
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.IMAGE_MEAN, self.cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=self.cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )
