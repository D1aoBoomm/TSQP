import torch
import torch.nn.quantized as nnq
import torch.nn as nn

import time

class VGG19(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(VGG19, self).__init__()
        self.x0 = [batch_size, 3, 224,224]
        self.OP0 = nnq.Conv2d(in_channels=3, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x2 = [batch_size, 64, 224,224]
        self.OP2 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x5 = [batch_size, 64, 112,112]
        self.OP5 = nnq.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x7 = [batch_size, 128, 112,112]
        self.OP7 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x10 = [batch_size, 128, 56,56]
        self.OP10 = nnq.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x12 = [batch_size, 256, 56,56]
        self.OP12 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x14 = [batch_size, 256, 56,56]
        self.OP14 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x16 = [batch_size, 256, 56,56]
        self.OP16 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x19 = [batch_size, 256, 56,56]
        self.OP19 = nnq.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x21 = [batch_size, 512, 28,28]
        self.OP21 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x23 = [batch_size, 512, 28,28]
        self.OP23 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x25 = [batch_size, 512, 28,28]
        self.OP25 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x28 = [batch_size, 512, 14,14]
        self.OP28 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x30 = [batch_size, 512, 14,14]
        self.OP30 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x32 = [batch_size, 512, 14,14]
        self.OP32 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x34 = [batch_size, 512, 14,14]
        self.OP34 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x38 = [batch_size, 512*7*7]
        self.OP38 = nnq.Linear(25088,4096)

        self.x40 = [batch_size,4096]
        self.OP40 = nnq.Linear(4096,4096)
        
        self.x42 = [batch_size,4096]
        self.OP42 = nnq.Linear(4096,1000)

        self.x37 = [batch_size, 512, 7,7]
        self.OP37 = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
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

net=VGG19(128)
for i in range(0,45):
    net(i)
print(net.time)