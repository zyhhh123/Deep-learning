import torch
import matplotlib.pyplot as plt
if __name__ =='__main__':
    x = torch.linspace(-2,2,100)
    y = x.pow(2)+torch.rand(x.size()*0.3)
    