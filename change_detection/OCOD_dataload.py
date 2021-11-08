import sys
from typing import Collection
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


from torchvision.transforms.transforms import ToTensor

sys.path.append('./')

import torchvision.transforms.functional as F
from read_and_write import *
import torch

from torchvision import transforms
import os
from PIL import Image
import numpy as np
import collections
import numpy as np


def OCOD(root):
    '''
    Args:
        image_dir: (string) root directory contains images
        label_dir: (string)root directory contains labels

    Return:
        train_list: (list) train_city
        test_lits: (list) test_city
    '''
    img_dir = os.path.join(root,'images\\Onera Satellite Change Detection dataset - Images')
    # find train city and test city
    file_name = os.listdir(img_dir)

    with open(os.path.join(img_dir,'train.txt')) as f:
        train_city_name = f.read().strip('\n').split(',')
    with open(os.path.join(img_dir,'test.txt')) as f:
        test_city_name = f.read().strip('\n').split(',')  
    
    return train_city_name,test_city_name

# 改写
def get_path(root,train):
    c1 = 'Onera Satellite Change Detection dataset - Images'
    c2 = 'Onera Satellite Change Detection dataset - Train Labels'
    img_dir = os.path.join(root,'images',c1)
    label_dir = os.path.join(root,'train_labels',c2)
    return img_dir,label_dir
    # 改写
def read_img(img_path,label_path,idx,train):
    # gdal2numpy
    T1_path = os.path.join(img_path,train[idx],'imgs_1_rect')
    T2_path =  os.path.join(img_path,train[idx],'imgs_2_rect')
    label_path = os.path.join(label_path,train[idx],'cm\\cm.png')
    list1 = []
    
    for i in os.listdir(T1_path):
        list1.append(gdal2np(os.path.join(T1_path,i)))
    T1 = np.dstack(list1)
    list1.clear()
    
    for i in os.listdir(T2_path):
        list1.append(gdal2np(os.path.join(T2_path,i)))
    T2 = np.dstack(list1)
    label = Image.open(label_path).convert('L')
    return T1,T2,np.array(label)

def encode_one_hot(img):
    img1 = img.copy()
    img[img==1] = 0
    img[img ==0] =1
    return np.dstack((img,img1))
# channel 0-1
def normlization(img):
    C = img.shape[2]
    for i in range(C):
        max1 = np.max(img[...,i])
        min1 = np.min(img[...,i])
        img[...,i] = (img[...,i]-min1)/(max1-min1)
    return img
class OCOD_Dataset(Dataset):
    '''
    Args:
        root: contains T1 image directory,T2 image directory,GT directory
        crop_size: random pick size of image 

    '''
    def __init__(self,root,train,crop_size) :

        self.img,self.label = get_path(root,train)
        self.size = crop_size
        self.train = train
        pass
    
    def __len__(self):
        num = len(self.train)
        return num

    def __getitem__(self,idx) :
       
        T1,T2,label = read_img(self.img,self.label,idx,self.train)
        T1 = T1.astype(np.float32)
        T2 = T2.astype(np.float32)
        # lable 0/1 化
        label[label != 0] = 1.0
        label = encode_one_hot(label)
        T1 = normlization(T1)[:self.size,:self.size]
        T2 = normlization(T2)[:self.size,:self.size]
        # crop_size
        T1 = ToTensor()(T1)
        T2 = ToTensor()(T2)
        label = ToTensor()(label[:self.size,:self.size])
        # F.crop(T1,0,0,self.size,self.size)
        # F.crop(T2,0,0,self.size,self.size)
        # F.crop(label,0,0,self.size,self.size)
        # data augmentation
        return T1,T2,label
if __name__ == "__main__":
    root ='D:\hello\OCOD'
    l1,l2 = OCOD(root)
    
    data1 = OCOD_Dataset(root,l1,320)
    # for x,y ,z in data1:
    #     print(x.shape)
    #     print(y.shape)
    #     print(z.shape)
    dataload_train = DataLoader(data1,batch_size=1,shuffle=True,num_workers=0,pin_memory=False)
    for x,y,z in dataload_train:
        
        

        pass

    pass
    
    
        
        
        
        
        
        
    
    
    

        