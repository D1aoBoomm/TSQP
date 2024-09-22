import torch
import torch.nn.quantized as nnq
import torch.nn as nn
import time

class ResNet18(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(ResNet18, self).__init__()
        self.x1 = [batch_size, 64, 112, 112]
        self.OP1 = nn.MaxPool2d(3,2,1,1)
        
        self.x3 = [batch_size, 64, 56, 56]
        self.OP3 = nn.ReLU()
        
        self.x6 = [batch_size, 64, 56, 56]
        self.OP6 = nn.ReLU()
        
        self.x9 = [batch_size, 128, 28, 28]
        self.OP9 = nn.ReLU()
        
        self.x13 = [batch_size, 128, 28, 28]
        self.OP13 = nn.ReLU()
        
        self.x16 = [batch_size, 256, 14, 14]
        self.OP16 = nn.ReLU()
        
        self.x20 = [batch_size, 256, 14, 14]
        self.OP20 = nn.ReLU()
        
        self.x23 = [batch_size, 512, 7, 7]
        self.OP23 = nn.ReLU()
        
        self.x27 = [batch_size, 512, 4, 4]
        self.OP27 = nn.ReLU()
        
        self.time = {}

    def forward(self, i):
        if hasattr(self, 'OP{}'.format(i)):
            x = torch.quantize_per_tensor(torch.rand(getattr(self, 'x{}'.format(i))),scale = 0.0472, zero_point = 64, dtype=torch.quint8)
            op = getattr(self, 'OP{}'.format(i))
            
            start = time.time()
            op(x)
            stop = time.time()
            
            self.time['OP{}'.format(i)]=stop-start
            
        else:
            print('no {}'.format(i))

net=ResNet18(128)
for i in range(0,30):
    net(i)
print(net.time)