import grpc
 
import msg_pb2
import msg_pb2_grpc
import torch

import torch.nn.quantized as nnq
import torch.nn as nn
import time

class MobileNetv2(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(MobileNetv2, self).__init__()
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
    
    i = -1
    # send warmup
    with grpc.insecure_channel('localhost:50051') as channel:
            stub = msg_pb2_grpc.MsgServiceStub(channel)
            response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(i)))
            i = int(response.msg)
    while True:
        i = mobilenet(i)
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = msg_pb2_grpc.MsgServiceStub(channel)
            response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(i)))
            print("Client Received: {}".format(response.msg))
            i = int(response.msg)
            if i == 999:
                break
            elif i > 84:
                response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(999)))
                break

    print('TEE Finished',file=f)
    print('TEE consumes: {}'.format(mobilenet.time),file=f)
 
if __name__ == '__main__':
    global mobilenet,f
    f = open('./result.txt', 'w')
    mobilenet = MobileNetv2(batch_size=1)
    for i in range(1):
        run()