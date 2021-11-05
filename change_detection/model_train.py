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
        B= T1.cpu().numpy()
        print(np.max(B))
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
    net = FC_EF(26,1)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    net.cuda(device)
    root  = 'D:\hello\OCOD'
    city,test_city = OCOD(root)
    # # 10
    # # 4
    train_city = city[:10]
    test_city = city[10:]
    
    img_dir = 'D:\hello\OCOD\images\Onera Satellite Change Detection dataset - Images'
    label_dir = 'D:\hello\OCOD\\train_labels\Onera Satellite Change Detection dataset - Train Labels'
    
    p = params(3,batch_size=2)
    dataload_train = OCOD_DataLoader(img_dir,label_dir,train_city,p)
    dataload_test = OCOD_DataLoader(img_dir,label_dir,test_city,p)
    
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 10
    for epoch in range(num_epoch):
        training_loss = train_loop(dataload_train,net,loss_fn,opt,device)
        testloss = test_loop(dataload_test,net,loss_fn,device)
        print('Epoch:{}/{}'.format(epoch,num_epoch))
        print('training loss:{}\ntestloss:{}'.format(training_loss,testloss))
        print('-'*10)
    torch.save(net.state_dict(),'model_weight.pth')   
    pass
    