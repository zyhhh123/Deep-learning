# time:2021.10.29

# first linear
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
class SimpleNet(nn.Module):
    def __init__(self) :
        super.__init__()
        self.input = nn.Linear(1,4,bias=True)
        self.out = nn.Linear(4,1,bias=True)
        
    # three layers
    # forward
    
    def forward(self,x):
        x = torch.relu(self.input(x))
        x = self.out(x) 
        return x

if __name__ =='__main__':
    plt.ion()
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x.pow(2)+torch.rand(x.size())*0.2
    
    net = SimpleNet()
    
    
    optimizer = SGD(net.parameters(),lr= 0.5)
    loss_func = nn.MSELoss()
   
    for i in range(300):
        prediction = net(x)
        
        train_loss = loss_func(prediction,y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if i%10 == 9:
            plt.cla()
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),prediction.data.numpy(),'r-', lw=5)
            plt.show()
            plt.pause(0.1)
            plt.close()

            
    
    
    
    
