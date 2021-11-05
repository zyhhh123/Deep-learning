
from torch.nn.modules import dropout


class params:

    def __init__(self,channels,num_workers=0,pin_memory=False,batch_size=64
                ,dropout_rate = 0.6  ) :
        self.batch_size =batch_size
        self.num_workers = num_workers
        self.pin_memory =pin_memory
        self.channels = channels
        self.dropout_rate = dropout_rate
        pass