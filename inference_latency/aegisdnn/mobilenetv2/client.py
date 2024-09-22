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
        self.x1 = [batch_size, 32, 112, 112]
        self.OP1 = nn.ReLU6()
        
        self.x3 = [batch_size, 32, 112, 112]
        self.OP3 = nn.ReLU6()
        
        self.x6 = [batch_size, 96, 112, 112]
        self.OP6 = nn.ReLU6()
        
        self.x0 = [batch_size, 3, 224,224]
        self.OP0 = nnq.Conv2d(in_channels=3, out_channels=32, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x2 = [batch_size, 32, 112,112]
        self.OP2 = nnq.Conv2d(in_channels=32, out_channels=32, kernel_size=[3,3], stride=[1,1], padding=[1,1],groups=32)
        
        self.x4 = [batch_size, 32, 112,112]
        self.OP4 = nnq.Conv2d(in_channels=32, out_channels=16, kernel_size=[1,1], stride=[1,1])
        
        self.x5 = [batch_size, 16, 112,112]
        self.OP5 = nnq.Conv2d(in_channels=16, out_channels=96, kernel_size=[1,1], stride=[1,1])
        
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