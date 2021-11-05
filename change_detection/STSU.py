

import torch
from torch.utils.data import Dataset,DataLoader
class STSU_Dataset(Dataset):
    def __init__(self,root,transform,target_transform) :
        
        self.transform = transform
        self.target_transform = target_transform
        pass

    def __len__(self):
        return 
    
    def __getitem__(self, idx) :
        pass