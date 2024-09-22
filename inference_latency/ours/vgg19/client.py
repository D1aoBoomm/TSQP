import grpc
 
import msg_pb2
import msg_pb2_grpc
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
        
        self.time = {}

    def forward(self, i):
        while True:
            if hasattr(self, 'OP{}'.format(i)):
            
                x = torch.quantize_per_tensor(torch.rand(getattr(self, 'x{}'.format(i))),scale = 0.0472, zero_point = 64, dtype=torch.quint8)
                op = getattr(self, 'OP{}'.format(i))
                for w in range(0,5):
                    op(x) # warmup
                
                start = time.time()
                op(x)
                stop = time.time()
                
                self.time['OP{}'.format(i)]=stop-start
                i = i + 1
            else:
                return i
        
def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # global vgg
    # vgg = VGG19(batch_size=32)
    
    i = -1
    # send warmup
    with grpc.insecure_channel('localhost:50051') as channel:
            stub = msg_pb2_grpc.MsgServiceStub(channel)
            response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(i)))
            i = int(response.msg)
    while True:
        i = vgg(i)
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = msg_pb2_grpc.MsgServiceStub(channel)
            response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(i)))
            print("Client Received: {}".format(response.msg))
            i = int(response.msg)
            if i == 999:
                break
            elif i > 42:
                response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(999)))
                break

    print('TEE Finished',file=f)
    print('TEE consumes: {}'.format(vgg.time),file=f)
 
if __name__ == '__main__':
    global vgg,f
    f = open('./result.txt', 'w')
    vgg = VGG19(batch_size=1)
    for i in range(1):
        run()