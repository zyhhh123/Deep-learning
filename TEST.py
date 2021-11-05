class foo:
    def __init__(self) :
        pass

    def h(self):
        print('h')
    def __b(self):
        print('b')

from PIL import Image
import numpy as np
import os

import torch
from change_detection.model.FC_EF import FC_EF
if __name__ =="__main__":
    a= np.ndarray(20).reshape(2,2,5)
    b= np.mean(a,axis=2)
    c = np.std(a,axis=2)
    print(b)
    print(c)
    # print((a-b)/c)

        

    
        
    
    