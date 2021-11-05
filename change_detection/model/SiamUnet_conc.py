
import sys
sys.path.append('.\\change_detection')
from OCOD_dataload import *
from params import params
import torch
import torch.nn as nn
# netural network's parameters

class SiamUnet_conc(nn.Module):
    def __init__(self,params):
        super(SiamUnet_conc,self).__init__()
        self.chnnels = params.channels
        dropout_rate = params.dropout_rate
        self.conv11 = nn.Conv2d(self.chnnels,16,3,padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(self.dropout_rate) 
        self.conv12 = nn.Conv2d(16,16,3,padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(self.dropout_rate)

        self.conv21 = nn.Conv2d(16,32,3,padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(self.dropout_rate) 
        self.conv22 = nn.Conv2d(32,32,3,padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(self.dropout_rate)

        self.conv31 = nn.Conv2d(32,64,3,padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(self.dropout_rate) 
        self.conv32 = nn.Conv2d(64,64,3,padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(self.dropout_rate)

        self.conv41 = nn.Conv2d(64,128,3,padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(self.dropout_rate) 
        self.conv42 = nn.Conv2d(128,128,3,padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(self.dropout_rate)

        # upsample
        self.upconv4 = nn.ConvTranspose2d(128,128,3,stride=2,output_padding=1)

    def forward(x1):


        pass

if __name__ == "__main__":
    pass