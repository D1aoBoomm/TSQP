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
        self.x12 = [batch_size, 256, 56,56]
        self.OP12 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x2 = [batch_size, 64, 224,224]
        self.OP2 = nnq.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1])

        self.x5 = [batch_size, 64, 112,112]
        self.OP5 = nnq.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x7 = [batch_size, 128, 112,112]
        self.OP7 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
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