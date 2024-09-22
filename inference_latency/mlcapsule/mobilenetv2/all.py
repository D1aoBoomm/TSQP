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
        
        self.x0 = [batch_size, 3, 224,224]
        self.OP0 = nnq.Conv2d(in_channels=3, out_channels=32, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x2 = [batch_size, 32, 112,112]
        self.OP2 = nnq.Conv2d(in_channels=32, out_channels=32, kernel_size=[3,3], stride=[1,1], padding=[1,1],groups=32)
        
        self.x4 = [batch_size, 32, 112,112]
        self.OP4 = nnq.Conv2d(in_channels=32, out_channels=16, kernel_size=[1,1], stride=[1,1])
        
        self.x5 = [batch_size, 16, 112,112]
        self.OP5 = nnq.Conv2d(in_channels=16, out_channels=96, kernel_size=[1,1], stride=[1,1])

        self.x7 = [batch_size, 96, 112,112]
        self.OP7 = nnq.Conv2d(in_channels=96, out_channels=96, kernel_size=[3,3], stride=[2,2], padding=[1,1],groups=96)
        
        self.x9 = [batch_size, 96, 56,56]
        self.OP9 = nnq.Conv2d(in_channels=96, out_channels=24, kernel_size=[1,1], stride=[1,1])

        self.x10 = [batch_size, 24, 56,56]
        self.OP10 = nnq.Conv2d(in_channels=24, out_channels=144, kernel_size=[1,1], stride=[1,1])
        
        self.x12 = [batch_size, 144, 56,56]
        self.OP12 = nnq.Conv2d(in_channels=144, out_channels=144, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=144)
        
        self.x14 = [batch_size, 144, 56,56]
        self.OP14 = nnq.Conv2d(in_channels=144, out_channels=24, kernel_size=[1,1], stride=[1,1])
        
        self.x15 = [batch_size, 24, 56,56]
        self.OP15 = nnq.Conv2d(in_channels=24, out_channels=144, kernel_size=[1,1], stride=[1,1])
        
        self.x17 = [batch_size, 144, 56,56]
        self.OP17 = nnq.Conv2d(in_channels=144, out_channels=144, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=144)
        
        self.x19 = [batch_size, 144, 28,28]
        self.OP19 = nnq.Conv2d(in_channels=144, out_channels=32, kernel_size=[1,1], stride=[1,1])
        
        self.x20 = [batch_size, 32, 28,28]
        self.OP20 = nnq.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1])
        
        self.x22 = [batch_size, 192, 28,28]
        self.OP22 = nnq.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=192)
        
        self.x23 = [batch_size, 192, 56,56]
        self.OP23 = nnq.Conv2d(in_channels=192, out_channels=32, kernel_size=[1,1], stride=[1,1])

        self.x24 = [batch_size, 32, 28,28]
        self.OP24 = nnq.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1])
        
        self.x26 = [batch_size, 192, 28,28]
        self.OP26 = nnq.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=192)
        
        self.x28 = [batch_size, 192, 28,28]
        self.OP28 = nnq.Conv2d(in_channels=192, out_channels=32, kernel_size=[1,1], stride=[1,1])
        
        self.x29 = [batch_size, 32, 28,28]
        self.OP29 = nnq.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1])
        
        self.x31 = [batch_size, 192, 28,28]
        self.OP31 = nnq.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=192)
        
        self.x33 = [batch_size, 192, 14,14]
        self.OP33 = nnq.Conv2d(in_channels=192, out_channels=64, kernel_size=[1,1], stride=[1,1])

        self.x34 = [batch_size, 64, 14,14]
        self.OP34 = nnq.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1])
        
        self.x36 = [batch_size, 384, 14,14]
        self.OP36 = nnq.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384)
        
        self.x38 = [batch_size, 384, 14,14]
        self.OP38 = nnq.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1])

        self.x39 = [batch_size, 64, 14,14]
        self.OP39 = nnq.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1])
        
        self.x41 = [batch_size, 384, 14,14]
        self.OP41 = nnq.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384)
        
        self.x43 = [batch_size, 384, 14,14]
        self.OP43 = nnq.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1])
        
        self.x44 = [batch_size, 64, 14,14]
        self.OP44 = nnq.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1])
        
        self.x46 = [batch_size, 384, 14,14]
        self.OP46 = nnq.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384)
        
        self.x48 = [batch_size, 384, 14,14]
        self.OP48 = nnq.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1])
        
        self.x49 = [batch_size, 64, 14,14]
        self.OP49 = nnq.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1])
        
        self.x51 = [batch_size, 384, 14,14]
        self.OP51 = nnq.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384)
        
        self.x53 = [batch_size, 384, 14,14]
        self.OP53= nnq.Conv2d(in_channels=384, out_channels=96, kernel_size=[1,1], stride=[1,1])
        
        self.x54 = [batch_size, 96, 14,14]
        self.OP54 = nnq.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1])
        
        self.x56 = [batch_size, 576, 14,14]
        self.OP56 = nnq.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=576)
        
        self.x58 = [batch_size, 576, 14,14]
        self.OP58= nnq.Conv2d(in_channels=576, out_channels=96, kernel_size=[1,1], stride=[1,1])
        
        self.x59 = [batch_size, 96, 14,14]
        self.OP59 = nnq.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1])
        
        self.x61 = [batch_size, 576, 14,14]
        self.OP61 = nnq.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=576)
        
        self.x63 = [batch_size, 576, 14,14]
        self.OP63= nnq.Conv2d(in_channels=576, out_channels=96, kernel_size=[1,1], stride=[1,1])

        self.x64 = [batch_size, 96, 14,14]
        self.OP64 = nnq.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1])
        
        self.x66 = [batch_size, 576, 14,14]
        self.OP66 = nnq.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=576)
        
        self.x68 = [batch_size, 576, 7,7]
        self.OP68= nnq.Conv2d(in_channels=576, out_channels=160, kernel_size=[1,1], stride=[1,1])
        
        self.x69 = [batch_size, 160, 7,7]
        self.OP69 = nnq.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1])
        
        self.x71 = [batch_size, 960, 7,7]
        self.OP71 = nnq.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960)
        
        self.x73 = [batch_size, 960, 7,7]
        self.OP73 = nnq.Conv2d(in_channels=960, out_channels=160, kernel_size=[1,1], stride=[1,1])
        
        self.x74 = [batch_size, 160, 7,7]
        self.OP74 = nnq.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1])
        
        self.x76 = [batch_size, 960, 7,7]
        self.OP76 = nnq.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960)
        
        self.x77 = [batch_size, 960, 7,7]
        self.OP77 = nnq.Conv2d(in_channels=960, out_channels=160, kernel_size=[1,1], stride=[1,1])
        
        self.x78 = [batch_size, 160, 7,7]
        self.OP78 = nnq.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1])
        
        self.x80 = [batch_size, 960, 7,7]
        self.OP80 = nnq.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960)
        
        self.x82 = [batch_size, 960, 7,7]
        self.OP82 = nnq.Conv2d(in_channels=960, out_channels=320, kernel_size=[1,1], stride=[1,1])

        self.x83 = [batch_size, 320, 7,7]
        self.OP83 = nnq.Conv2d(in_channels=320, out_channels=1280, kernel_size=[1,1], stride=[1,1])

        self.x84 = [batch_size, 1280]
        self.OP84 = nnq.Linear(1280,1000)
        
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
f = open('./result.txt','w')
for i in range(0,90):
    print(i)
    net(i)
print(net.time, file=f)