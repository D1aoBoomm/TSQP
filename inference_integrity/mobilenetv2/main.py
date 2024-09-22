import torch
import torch.nn as nn
import tqdm

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
        self.OP0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3,3], stride=[2,2], padding=[1,1],bias=False)
        
        self.x2 = [batch_size, 32, 112,112]
        self.OP2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3,3], stride=[1,1], padding=[1,1],groups=32,bias=False)
        
        self.x4 = [batch_size, 32, 112,112]
        self.OP4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x5 = [batch_size, 16, 112,112]
        self.OP5 = nn.Conv2d(in_channels=16, out_channels=96, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x7 = [batch_size, 96, 112,112]
        self.OP7 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3,3], stride=[2,2], padding=[1,1],groups=96,bias=False)
        
        self.x9 = [batch_size, 96, 56,56]
        self.OP9 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x10 = [batch_size, 24, 56,56]
        self.OP10 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x12 = [batch_size, 144, 56,56]
        self.OP12 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=144,bias=False)
        
        self.x14 = [batch_size, 144, 56,56]
        self.OP14 = nn.Conv2d(in_channels=144, out_channels=24, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x15 = [batch_size, 24, 56,56]
        self.OP15 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x17 = [batch_size, 144, 56,56]
        self.OP17 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=144,bias=False)
        
        self.x19 = [batch_size, 144, 28,28]
        self.OP19 = nn.Conv2d(in_channels=144, out_channels=32, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x20 = [batch_size, 32, 28,28]
        self.OP20 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x22 = [batch_size, 192, 28,28]
        self.OP22 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=192,bias=False)
        
        self.x23 = [batch_size, 192, 56,56]
        self.OP23 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x24 = [batch_size, 32, 28,28]
        self.OP24 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x26 = [batch_size, 192, 28,28]
        self.OP26 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=192,bias=False)
        
        self.x28 = [batch_size, 192, 28,28]
        self.OP28 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x29 = [batch_size, 32, 28,28]
        self.OP29 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x31 = [batch_size, 192, 28,28]
        self.OP31 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=192,bias=False)
        
        self.x33 = [batch_size, 192, 14,14]
        self.OP33 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x34 = [batch_size, 64, 14,14]
        self.OP34 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x36 = [batch_size, 384, 14,14]
        self.OP36 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384,bias=False)
        
        self.x38 = [batch_size, 384, 14,14]
        self.OP38 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x39 = [batch_size, 64, 14,14]
        self.OP39 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x41 = [batch_size, 384, 14,14]
        self.OP41 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384,bias=False)
        
        self.x43 = [batch_size, 384, 14,14]
        self.OP43 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x44 = [batch_size, 64, 14,14]
        self.OP44 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x46 = [batch_size, 384, 14,14]
        self.OP46 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384,bias=False)
        
        self.x48 = [batch_size, 384, 14,14]
        self.OP48 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x49 = [batch_size, 64, 14,14]
        self.OP49 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x51 = [batch_size, 384, 14,14]
        self.OP51 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=384,bias=False)
        
        self.x53 = [batch_size, 384, 14,14]
        self.OP53= nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x54 = [batch_size, 96, 14,14]
        self.OP54 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x56 = [batch_size, 576, 14,14]
        self.OP56 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=576,bias=False)
        
        self.x58 = [batch_size, 576, 14,14]
        self.OP58= nn.Conv2d(in_channels=576, out_channels=96, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x59 = [batch_size, 96, 14,14]
        self.OP59 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x61 = [batch_size, 576, 14,14]
        self.OP61 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=576,bias=False)
        
        self.x63 = [batch_size, 576, 14,14]
        self.OP63= nn.Conv2d(in_channels=576, out_channels=96, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x64 = [batch_size, 96, 14,14]
        self.OP64 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x66 = [batch_size, 576, 14,14]
        self.OP66 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[3,3], stride=[2,2],padding=[1,1],groups=576,bias=False)
        
        self.x68 = [batch_size, 576, 7,7]
        self.OP68= nn.Conv2d(in_channels=576, out_channels=160, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x69 = [batch_size, 160, 7,7]
        self.OP69 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x71 = [batch_size, 960, 7,7]
        self.OP71 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960,bias=False)
        
        self.x73 = [batch_size, 960, 7,7]
        self.OP73 = nn.Conv2d(in_channels=960, out_channels=160, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x74 = [batch_size, 160, 7,7]
        self.OP74 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x76 = [batch_size, 960, 7,7]
        self.OP76 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960,bias=False)
        
        self.x77 = [batch_size, 960, 7,7]
        self.OP77 = nn.Conv2d(in_channels=960, out_channels=160, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x78 = [batch_size, 160, 7,7]
        self.OP78 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=[1,1], stride=[1,1],bias=False)
        
        self.x80 = [batch_size, 960, 7,7]
        self.OP80 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=960,bias=False)
        
        self.x82 = [batch_size, 960, 7,7]
        self.OP82 = nn.Conv2d(in_channels=960, out_channels=320, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x83 = [batch_size, 320, 7,7]
        self.OP83 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=[1,1], stride=[1,1],bias=False)

        self.x84 = [batch_size, 1280]
        self.OP84 = nn.Linear(1280,1000,bias=False)
        
        self.time = {}

batch_size = 1
correct = 0
fail = 0
exp_num = 1000

net = MobileNetv2(batch_size=1).to('cuda')
state_dict = net.state_dict()
for name in state_dict.keys():
    state_dict[name] = torch.randint(-128,127,state_dict[name].shape).to(torch.float32)
net.load_state_dict(state_dict)

print("Starting test integrity detection successful rate | WITHOUT ATTACK")
for i in tqdm.tqdm(range(0,exp_num)):
    for name in state_dict.keys():
        id = name[:-6][2:][:-1]
        real_input = torch.randint(0,127,getattr(net, 'x{}'.format(id))).to(torch.float32)
        mu = torch.mean(real_input)
        var = torch.var(real_input)
        
        fp_1 = torch.normal(mean=mu/2, std=(var**0.5)/(2**0.5), size=real_input.shape).to(torch.int32).to(torch.float32)
        fp_2 = torch.normal(mean=mu/2, std=(var**0.5)/(2**0.5), size=real_input.shape).to(torch.int32).to(torch.float32)
        
        rr_1 = getattr(net, 'OP{}'.format(id))(fp_1.to('cuda'))
        rr_2 = getattr(net, 'OP{}'.format(id))(fp_2.to('cuda'))
        
        resp = getattr(net, 'OP{}'.format(id))((fp_1+fp_2).to('cuda'))
        
        if torch.all(resp == rr_1+rr_2) != True:
            fail += 1
            print('Integrity detection failed for {} times'.format(fail))
        else:
            correct += 1

print('Finished! Successful Rate: {}'.format(correct/correct+fail))

print("Starting test integrity detection successful rate | WITH ATTACK")
correct = 0
fail = 0

for i in tqdm.tqdm(range(0,exp_num)):
    for name in state_dict.keys():
        id = name[:-6][2:][:-1]
        real_input = torch.randint(0,127,getattr(net, 'x{}'.format(id))).to(torch.float32)
        mu = torch.mean(real_input)
        var = torch.var(real_input)
        
        fp_1 = torch.normal(mean=mu/2, std=(var**0.5)/(2**0.5), size=real_input.shape).to(torch.int32).to(torch.float32)
        fp_2 = torch.normal(mean=mu/2, std=(var**0.5)/(2**0.5), size=real_input.shape).to(torch.int32).to(torch.float32)
        
        rr_1 = getattr(net, 'OP{}'.format(id))(fp_1.to('cuda'))
        rr_2 = getattr(net, 'OP{}'.format(id))(fp_2.to('cuda'))
        
        resp = getattr(net, 'OP{}'.format(id))((fp_1+fp_2).to('cuda'))
        resp = resp + torch.randint(0, 127, resp.shape).to('cuda') # manipulation
        
        if torch.all(resp == rr_1+rr_2) != True:
            correct += 1
            
        else:
            fail += 1
            print('\n')
            print('Integrity detection failed for {} times'.format(fail))
print('Finished! Successful Rate: {}'.format(correct/correct+fail))