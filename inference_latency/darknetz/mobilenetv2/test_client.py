import torch
import torch.nn.quantized as nnq
import torch.nn as nn
import time

class MobileNetv2(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(MobileNetv2, self).__init__()
        self.x1 = [batch_size, 32, 112, 112]
        self.OP1 = nn.ReLU6()
        
        self.x3 = [batch_size, 32, 112, 112]
        self.OP3 = nn.ReLU6()
        
        self.x6 = [batch_size, 96, 112, 112]
        self.OP6 = nn.ReLU6()
        
        self.x8 = [batch_size, 96, 56, 56]
        self.OP8 = nn.ReLU6()
        
        self.x11 = [batch_size, 144, 56, 56]
        self.OP11 = nn.ReLU6()
        
        self.x13 = [batch_size, 144, 56, 56]
        self.OP13 = nn.ReLU6()
        
        self.x16 = [batch_size, 144, 56, 56]
        self.OP16 = nn.ReLU6()
        
        self.x18 = [batch_size, 144, 28, 28]
        self.OP18 = nn.ReLU6()
        
        self.x21 = [batch_size, 192, 28, 28]
        self.OP21 = nn.ReLU6()
        
        self.x23 = [batch_size, 192, 28, 28]
        self.OP23 = nn.ReLU6()
        
        self.x25 = [batch_size, 192, 28, 28]
        self.OP25 = nn.ReLU6()
        
        self.x27 = [batch_size, 192, 28, 28]
        self.OP27 = nn.ReLU6()
        
        self.x30 = [batch_size, 192, 28, 28]
        self.OP30 = nn.ReLU6()
        
        self.x32 = [batch_size, 192, 14, 14]
        self.OP32 = nn.ReLU6()
        
        self.x35 = [batch_size, 384, 14, 14]
        self.OP35 = nn.ReLU6()
        
        self.x37 = [batch_size, 384, 14, 14]
        self.OP37 = nn.ReLU6()
        
        self.x40 = [batch_size, 384, 14, 14]
        self.OP40 = nn.ReLU6()
        
        self.x42 = [batch_size, 384, 14, 14]
        self.OP42 = nn.ReLU6()
        
        self.x45 = [batch_size, 384, 14, 14]
        self.OP45 = nn.ReLU6()
        
        self.x47 = [batch_size, 384, 14, 14]
        self.OP47 = nn.ReLU6()
        
        self.x50 = [batch_size, 384, 14, 14]
        self.OP50 = nn.ReLU6()
        
        self.x52 = [batch_size, 384, 14, 14]
        self.OP52 = nn.ReLU6()
        
        self.x55 = [batch_size, 576, 14, 14]
        self.OP55 = nn.ReLU6()
        
        self.x57 = [batch_size, 576, 14, 14]
        self.OP57 = nn.ReLU6()
        
        self.x60 = [batch_size, 576, 14, 14]
        self.OP60 = nn.ReLU6()
        
        self.x62 = [batch_size, 576, 14, 14]
        self.OP62 = nn.ReLU6()
        
        self.x65 = [batch_size, 576, 14, 14]
        self.OP65 = nn.ReLU6()
        
        self.x67 = [batch_size, 576, 7, 7]
        self.OP67 = nn.ReLU6()
        
        self.x70 = [batch_size, 960, 7, 7]
        self.OP70 = nn.ReLU6()
        
        self.x72 = [batch_size, 960, 7, 7]
        self.OP72 = nn.ReLU6()
        
        self.x75 = [batch_size, 960, 7, 7]
        self.OP75 = nn.ReLU6()
        
        self.x79 = [batch_size, 960, 7, 7]
        self.OP79 = nn.ReLU6()
        
        self.x81 = [batch_size, 960, 7, 7]
        self.OP81 = nn.ReLU6()
        
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

net=MobileNetv2(128)
for i in range(0,90):
    net(i)
print(net.time)