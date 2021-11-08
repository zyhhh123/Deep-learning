from torch import optim

from model.FC_EF import *
from OCOD_dataload import *
# model training
import torch
import torch.nn.functional as F
# from utils.metrics import Metrics
def train_loop(dataload,net,loss_fn,optimizer,device):
    running_loss = 0
    # metrics = Metrics(range(2))
    for T1,T2,masks in dataload:
            
        T1 = T1.to(device)
        T2 = T2.to(device)
        x = torch.cat((T1,T2),dim=1).to(device)
        masks = masks.to(device)
        outputs = net(x)
        
        loss = loss_fn(outputs,masks)
        running_loss +=loss.item()
            # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return running_loss
def test_loop(dataload,net,loss_fn,device): 
    test_loss = 0
    with torch.no_grad():
        for T1,T2, y in dataload:
            
            T1 = T1.to(device)
            T2 = T2.to(device)
            x = torch.cat((T1,T2),dim=1).to(device)
            y = y.to(device)
            pred = net(x)
            test_loss += loss_fn(pred, y).item()          
    return test_loss



        
    


    

if __name__ =="__main__":
    root ='D:\hello\OCOD'
    l1,l2 = OCOD(root)
    # train = l1[:10]
    data = OCOD_Dataset(root,l1,320)
    
    # test_data = OCOD_Dataset(root,False,train,test,320)
    train_size = int(0.8*len(data))
    test_size = len(data) - train_size
    train_data,test_data = torch.utils.data.random_split(data,[train_size,test_size])
    
    dataload_train = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=2,pin_memory=False)
    dataload_test = DataLoader(test_data,batch_size=1,shuffle=True,num_workers=2,pin_memory=False)
    
    net = FC_EF(26,2)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    net.cuda(device)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 10
    for epoch in range(num_epoch):
        training_loss = train_loop(dataload_train,net,loss_fn,opt,device)
        # testloss = test_loop(dataload_test,net,loss_fn,device)
        print('Epoch:{}/{}'.format(epoch,num_epoch))
        print('training loss:{}\n'.format(training_loss))
        print('-'*10)
    torch.save(net.state_dict(),'model_weight.pth')   
    pass
    