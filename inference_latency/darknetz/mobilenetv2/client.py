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