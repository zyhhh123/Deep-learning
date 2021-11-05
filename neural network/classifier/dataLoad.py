# dataload
#引用

import torch
from torch.utils.data import Dataset,DataLoader, dataloader
from torchvision import transforms
import os
from PIL import Image
import random
import math
import numpy as np
# 图像预处理，
mean,std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
train_transforme = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])
test_transforme = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])
eval_transforme = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])
target_transform = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),

])

#定义一个数据集
class OneDataset(Dataset):
    """数据集演示
        Args:
            img_dir: directory of dataset
            lable_dir: 
            transform: transform class
            file_select: need to partition or not
    """
    def __init__(self,img_dir,lable_dir,transform = None):
        
        self.img_dir = img_dir
        self.lable = lable_dir
        self.transform = transform
        
       
    def __len__(self):
        '''
        返回df的长度
        '''
        img_paths = os.listdir(self.img_dir)
        return len(img_paths)
    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        img_paths = os.listdir(self.img_dir)
        img_filename = os.path.join(self.img_dir,img_paths[idx])
        image = Image.open(img_filename)
        if self.transform:
            image = self.transform(image)
        
        label_paths = os.listdir(self.label_dir)
        label_path = os.path.join(self.label_dir,label_paths[idx])
        label = Image.open(label_path)
        return image,label

