import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
class FC_EF(nn.Module):
    def __init__(self,input_channels,label) :
        super(FC_EF,self).__init__()
        self.input_channels = input_channels
        # encode
        self.conv11 = nn.Conv2d(self.input_channels,16,3,padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d()
        self.conv12 = nn.Conv2d(16,16,3,1,padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d()

        self.conv21 = nn.Conv2d(16,32,3,padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d()
        self.conv22 = nn.Conv2d(32,32,3,padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d()

        self.conv31 = nn.Conv2d(32,64,3,padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d() 
        self.conv32 = nn.Conv2d(64,64,3,padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d()
        self.conv33 = self.conv32
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d()

        self.conv41 = nn.Conv2d(64,128,3,padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d() 
        self.conv42 = nn.Conv2d(128,128,3,padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d() 
        self.conv43 = nn.Conv2d(128,128,3,padding=1)
        self.bn43 = self.bn42
        self.do43 = self.do42


        # decode
        self.upconv4 = nn.ConvTranspose2d(128,128,3,stride=2,output_padding=1,padding=1)

        self.econv41 = nn.Conv2d(256,128,3,padding=1)
        self.ebn41 = self.bn41
        self.edo41 = self.do42
        self.econv42 = self.conv42
        self.ebn42 = self.ebn41
        self.edo42 = self.edo41
        self.econv43 = nn.Conv2d(128,64,3,padding=1)
        self.ebn43 = nn.BatchNorm2d(64)
        self.edo43 = nn.Dropout2d()
        
        self.upconv3 = nn.ConvTranspose2d(64,64,3,stride=2,output_padding=1,padding=1)

        self.econv31 = nn.Conv2d(128,64,3,padding=1)
        self.ebn31 = nn.BatchNorm2d(64)
        self.edo31 = nn.Dropout2d()
        self.econv32 = self.conv32
        self.ebn32 = nn.BatchNorm2d(64)
        self.edo32 = nn.Dropout2d()
        self.ebn33 = nn.BatchNorm2d(64)
        self.edo33 = nn.Dropout2d()
        self.econv33 = nn.Conv2d(64,32,3,padding=1)
        self.ebn33 = nn.BatchNorm2d(32)
        self.edo33 = nn.Dropout2d()

        self.upconv2 = nn.ConvTranspose2d(32,32,3,stride=2,output_padding=1,padding=1)

        self.econv21 = self.econv33
        self.ebn21 = nn.BatchNorm2d(32)
        self.edo21 = nn.Dropout2d()
        self.econv22 = nn.Conv2d(32,16,3,padding=1)
        self.ebn22 = nn.BatchNorm2d(16)
        self.edo22 = nn.Dropout2d()

        self.upconv1 = nn.ConvTranspose2d(16,16,3,stride=2,output_padding=1,padding=1)
        
        self.econv11 = nn.Conv2d(32,16,3,padding=1)
        self.ebn11 = nn.BatchNorm2d(16)
        self.edo11 = nn.Dropout2d()
        self.econv12 = nn.Conv2d(16,label,3,padding=1)
        self.out = nn.Softmax(dim=1)

    def forward(self,x):
        # encode
        # stage1
        
        x = self.do11(F.relu(self.bn11(self.conv11(x))))
        x1 = self.do12(F.relu(self.bn12(self.conv12(x))))
        pool = F.max_pool2d(x1,stride =2 ,kernel_size= 2)
      
        # stage2
        x = self.do21(F.relu(self.bn21(self.conv21(pool))))
        x2 = self.do22(F.relu(self.bn22(self.conv22(x))))
        pool = F.max_pool2d(x2,stride =2 ,kernel_size= 2)
       
        # stage3
        x = self.do31(F.relu(self.bn31(self.conv31(pool))))
        x = self.do32(F.relu(self.bn32(self.conv32(x))))
        x3 = self.do33(F.relu(self.bn33(self.conv33(x))))
        pool = F.max_pool2d(x3,stride =2 ,kernel_size= 2)
       
        # stage4
        x = self.do41(F.relu(self.bn41(self.conv41(pool))))
        x = self.do42(F.relu(self.bn42(self.conv42(x))))
        x4 = self.do43(F.relu(self.bn43(self.conv43(x))))
        pool = F.max_pool2d(x4,stride =2 ,kernel_size= 2)
       
        # decode
        # stage 4d
        x = self.upconv4(pool)
        
        
        x = torch.cat((x,x4),dim=1)
        
        x = self.edo41(F.relu(self.ebn41(self.econv41(x))))
        x = self.edo42(F.relu(self.ebn42(self.econv42(x))))
        x = self.edo43(F.relu(self.ebn43(self.econv43(x))))
        
        # stage 3d
        x = self.upconv3(x)
        x = torch.cat((x,x3),dim=1)
        x = self.edo31(F.relu(self.ebn31(self.econv31(x))))
        x = self.edo32(F.relu(self.ebn32(self.econv32(x))))
        x = self.edo33(F.relu(self.ebn33(self.econv33(x))))
        
        # stage 2d
        x = self.upconv2(x)
        x = torch.cat((x,x2),dim=1)
        x = self.edo21(F.relu(self.ebn21(self.econv21(x))))
        x = self.edo22(F.relu(self.ebn22(self.econv22(x))))
        
        # stage 1d
        x = self.upconv1(x)
        x = torch.cat((x,x1),dim=1)
        x = self.edo11(F.relu(self.ebn11(self.econv11(x))))
        x = self.econv12(x)
        x = self.out(x)
        
        return x
from torchsummary import summary
import numpy as np
if __name__ == "__main__":
    x= torch.randn(1,3,256,256)
    net = FC_EF(3,2)
    device = torch.device('cuda')
    x= torch.randn(1,3,256,256).to(device)
    net.to(device)
    
    x=net(x).detach().cpu().numpy()
    print(np.all(x <1))
    # summary(net,(3,256,256),batch_size = 1)
    
    
        
        

        

