import torch
import torch.nn.quantized as nnq
import torch.nn as nn

import time

class ResNet18(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(ResNet18, self).__init__()
        self.x0 = [batch_size, 3, 224,224]
        self.OP0 = nnq.Conv2d(in_channels=3, out_channels=64, kernel_size=[7,7], stride=[2,2], padding=[3,3])
        
        self.x2 = [batch_size, 64, 56, 56]
        self.OP2 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x4 = [batch_size, 64, 56, 56]
        self.OP4 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x5 = [batch_size, 64, 56, 56]
        self.OP5 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x7 = [batch_size, 64, 56, 56]
        self.OP7 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x8 = [batch_size, 64, 28, 28]
        self.OP8 = nnq.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x10 = [batch_size, 128, 28, 28]
        self.OP10 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x11 = [batch_size, 64, 28, 28]
        self.OP11 = nnq.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x12 = [batch_size, 128, 28, 28]
        self.OP12 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x14 = [batch_size, 128, 28, 28]
        self.OP14 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x15 = [batch_size, 128, 28, 28]
        self.OP15 = nnq.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x17 = [batch_size, 256, 14, 14]
        self.OP17 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x18 = [batch_size, 128, 28, 28]
        self.OP18 = nnq.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x19 = [batch_size, 256, 14, 14]
        self.OP19 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x21 = [batch_size, 256, 14, 14]
        self.OP21 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x22 = [batch_size, 256, 14, 14]
        self.OP22 = nnq.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x24 = [batch_size, 512, 7, 7]
        self.OP24 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x25 = [batch_size, 256, 14, 14]
        self.OP25 = nnq.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x26 = [batch_size, 512, 7, 7]
        self.OP26 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])

        self.x28 = [batch_size, 512, 4, 4]
        self.OP28 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])

        self.x29 = [batch_size, 512, 2, 2]
        self.OP29 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.x30 = [batch_size, 512]
        self.OP30 = nnq.Linear(512,1000)

        self.time = {}

    def forward(self, i):
        if hasattr(self, 'OP{}'.format(i)):
            x = torch.quantize_per_tensor(torch.rand(getattr(self, 'x{}'.format(i))),scale = 0.0472, zero_point = 64, dtype=torch.quint8)
            op = getattr(self, 'OP{}'.format(i))
            for w in range(0,5):
                op(x) # warmup
            
            start = time.time()
            op(x)
            stop = time.time()
            
            self.time['OP{}'.format(i)]=stop-start
            
        else:
            print('no {}'.format(i))
        

net=ResNet18(128)
for i in range(0,31):
    net(i)
print(net.time)