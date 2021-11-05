import sys

sys.path.append('./')

from torchvision.transforms.transforms import ToTensor
from params import params

from read_and_write import *
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# pre processing
mean,std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    # transforms.Resize([2,2]), 
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(90),
    # transforms.RandomRotation(180),
    # transforms.RandomRotation(270),
    transforms.ToTensor(),  
                                            
    # transforms.Normalize(mean=mean, std=std),  
    
  

])

target_transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),

])

test_transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),

])
val_transform = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),

])

transform_list = [train_transform,test_transform,val_transform,target_transform]

def OCOD(root):
    '''
    Args:
        image_dir: (string) root directory contains images
        label_dir: (string)root directory contains labels

    Return:
        train_list: (list) train_city
        test_lits: (list) test_city
    '''
    img_dir = os.path.join(root,'images//Onera Satellite Change Detection dataset - Images')
    # find train city and test city
    file_name = os.listdir(img_dir)

    with open(os.path.join(img_dir,'train.txt')) as f:
        train_city_name = f.read().strip('\n').split(',')
    with open(os.path.join(img_dir,'test.txt')) as f:
        test_city_name = f.read().strip('\n').split(',')  
    
    return train_city_name,test_city_name


class OCOD_Dataset(Dataset):
    '''
    Args:
        root: contains T1 image directory,T2 image directory,GT directory
        

    '''
    def __init__(self,img_root,label_root,city,transform,target_transform) :
        self.city = city
        self.img_dir = img_root 
        self.label_dir = label_root
        self.transform = transform
        self.target_transform = target_transform
        pass

    def __len__(self):
        return len(self.city)
    
    def __getitem__(self, idx) :
        city_name = self.city[idx]

        image_path = os.path.join(self.img_dir,city_name)
        label_path = os.path.join(self.label_dir,city_name)
        T1_path = os.path.join(image_path,'imgs_1_rect')
        T2_path = os.path.join(image_path,'imgs_2_rect')
        GT = np.asarray(Image.open(os.path.join(label_path,'cm\\'+'cm.png')).convert("RGB"))[:320,:360,:]
        # read images
        list1 =[]
        for i in os.listdir(T1_path):
            list1.append(gdal2np(os.path.join(T1_path,i)))
        T1 = np.dstack(list1).astype('float32')[:320,:360,:]
        
        list2 =[]
        for i in os.listdir(T2_path):
            list2.append(gdal2np(os.path.join(T1_path,i)))
        T2 = np.dstack(list2).astype('float32')[:320,:360,:]
        
        T1 = ToTensor()(T1)
        T2 = ToTensor()(T2)
        GT = ToTensor()(GT) 
        # if self.transform and target_transform:
        #     T1 = self.transform((T1.astype('float32')))
        #     T2 = self.transform((T2.astype('float32')))
        #     GT = self.target_transform(GT)
        return T1,T2,GT
def OCOD_DataLoader(img_dir,label_dir,city,train_transform,target_transform,params):
    return DataLoader(OCOD_Dataset(img_dir,label_dir,city,train_transform,target_transform),batch_size =params.batch_size
        ,num_workers = params.num_workers,pin_memory =params.pin_memory,shuffle=True)


if __name__ == "__main__":
    
    root  = 'D:\hello\OCOD'
    train_city,test_city = OCOD(root)
    image_dir = 'D:\hello\OCOD\images\Onera Satellite Change Detection dataset - Images'
    
    label_dir = 'D:\hello\OCOD\\train_labels\Onera Satellite Change Detection dataset - Train Labels'
    p = params(3,batch_size=1)
    dataload = OCOD_DataLoader(image_dir,label_dir,train_city,train_transform,target_transform,p)
    for m ,(x,y,z) in enumerate(dataload):
        print(m)
        print(x.shape)
        print(y.shape)
    
    
    

        