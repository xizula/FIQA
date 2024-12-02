import torch
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import random_split


class FaceRecoDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, folder_mapping=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = folder_mapping

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        image, label = original_tuple
        path, _ = self.samples[index]

        return image, label, path
    
    def get_folder_to_label_mapping(self):
        return {v: k for k, v in self.class_to_idx.items()}

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_dir = config['data']['path']

    subfolders = sorted(os.listdir(dataset_dir))
    folder_to_label = {folder: idx for idx, folder in enumerate(subfolders)}

    dataset = FaceRecoDataset(root=dataset_dir, transform=transform, folder_mapping=folder_to_label)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader


class QualityDataset(Dataset):
    '''
    Build dataset via data list file
    '''
    def __init__(self, img_list=None, batch_size=None):

        self.img_list = img_list
        self.batch_size = batch_size
        data = pd.read_csv(self.img_list)
        self.imgPath = data['path'].values
        self.target = data['label'].values

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        img = Image.open(imgPath)                                          
        target = self.target[index]
        return imgPath, img, target

    def __len__(self):
        return(len(self.imgPath))


def load_quality_data(img_list, batch_size, pin_memory, num_workers, train=False, split_ratio=0.85):
    torch.manual_seed(42)
    img_list = img_list # csv z pseudo-labelkami
    dataset = QualityDataset(img_list=img_list, batch_size=batch_size)
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=pin_memory, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader

