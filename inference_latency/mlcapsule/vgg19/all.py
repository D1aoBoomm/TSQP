import torch
import torch.nn.quantized as nnq
import torch.nn as nn
import time

class VGG19(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(VGG19, self).__init__()
        self.x1 =[batch_size, 64, 224,224]
        self.OP1 = nn.ReLU()
        
        self.x3 =[batch_size, 64, 224,224]
        self.OP3 = nn.ReLU()
        
        self.x4 =[batch_size, 64, 224,224]
        self.OP4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x6 =[batch_size, 128, 112,112]
        self.OP6 = nn.ReLU()
        
        self.x8 =[batch_size, 128, 112,112]
        self.OP8 = nn.ReLU()
        
        self.x9 =[batch_size, 128, 112,112]
        self.OP9 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x11 =[batch_size, 256, 56,56]
        self.OP11 = nn.ReLU()
        
        self.x13 =[batch_size, 256, 56,56]
        self.OP13 = nn.ReLU()
        
        self.x15 =[batch_size, 256, 56,56]
        self.OP15 = nn.ReLU()
        
        self.x17 =[batch_size, 256, 56,56]
        self.OP17 = nn.ReLU()
        
        self.x18 =[batch_size, 256, 56,56]
        self.OP18 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x20 =[batch_size, 512, 28,28]
        self.OP20 = nn.ReLU()
        
        self.x22 =[batch_size, 512, 28,28]
        self.OP22 = nn.ReLU()
        
        self.x24 =[batch_size, 512, 28,28]
        self.OP24 = nn.ReLU()
        
        self.x26 =[batch_size, 512, 28,28]
        self.OP26 = nn.ReLU()
        
        self.x27 =[batch_size, 512, 28,28]
        self.OP27 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x29 =[batch_size, 512, 14,14]
        self.OP29 = nn.ReLU()
        
        self.x31 =[batch_size, 512, 14,14]
        self.OP31 = nn.ReLU()
        
        self.x33 =[batch_size, 512, 14,14]
        self.OP33 = nn.ReLU()
        
        self.x35 =[batch_size, 512, 14,14]
        self.OP35 = nn.ReLU()
        
        self.x36 =[batch_size, 512, 14,14]
        self.OP36 = nn.MaxPool2d(2,2)
        
        self.x39 =[batch_size, 4096]
        self.OP39 = nn.ReLU()
        
        self.x41 =[batch_size, 4096]
        self.OP41 = nn.ReLU()
        
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
f = open('./result.txt','w')
for i in range(0,45):
    print(i)
    net(i)
print(net.time, file=f)