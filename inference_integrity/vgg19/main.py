import torch
import torch.nn as nn
import tqdm

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
        self.OP0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x2 = [batch_size, 64, 224,224]
        self.OP2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)

        self.x5 = [batch_size, 64, 112,112]
        self.OP5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x7 = [batch_size, 128, 112,112]
        self.OP7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)

        self.x10 = [batch_size, 128, 56,56]
        self.OP10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x12 = [batch_size, 256, 56,56]
        self.OP12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x14 = [batch_size, 256, 56,56]
        self.OP14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x16 = [batch_size, 256, 56,56]
        self.OP16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x19 = [batch_size, 256, 56,56]
        self.OP19 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x21 = [batch_size, 512, 28,28]
        self.OP21 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x23 = [batch_size, 512, 28,28]
        self.OP23 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x25 = [batch_size, 512, 28,28]
        self.OP25 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x28 = [batch_size, 512, 14,14]
        self.OP28 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x30 = [batch_size, 512, 14,14]
        self.OP30 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x32 = [batch_size, 512, 14,14]
        self.OP32 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)
        
        self.x34 = [batch_size, 512, 14,14]
        self.OP34 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False)

        self.x38 = [batch_size, 512*7*7]
        self.OP38 = nn.Linear(25088,4096, bias=False)

        self.x40 = [batch_size,4096]
        self.OP40 = nn.Linear(4096,4096, bias=False)
        
        self.x42 = [batch_size,4096]
        self.OP42 = nn.Linear(4096,1000, bias=False)

        self.x37 = [batch_size, 512, 7,7]
        self.OP37 = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
        self.x43 = [batch_size,1000]
        
        self.time = {}

batch_size = 1
correct = 0
fail = 0
exp_num = 1000

net = VGG19(batch_size=1).to('cuda')
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