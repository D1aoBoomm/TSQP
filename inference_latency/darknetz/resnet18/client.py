import grpc
 
import msg_pb2
import msg_pb2_grpc
import torch

import torch.nn.quantized as nnq
import torch.nn as nn
import time

class ResNet18(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(ResNet18, self).__init__()

        
        self.x23 = [batch_size, 512, 7, 7]
        self.OP23 = nn.ReLU()
        
        self.x27 = [batch_size, 512, 4, 4]
        self.OP27 = nn.ReLU()
        
        self.x22 = [batch_size, 256, 14, 14]
        self.OP22 = nnq.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x24 = [batch_size, 512, 7, 7]
        self.OP24 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x25 = [batch_size, 256, 14, 14]
        self.OP25 = nnq.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x26 = [batch_size, 512, 7, 7]
        self.OP26 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])

        self.x28 = [batch_size, 512, 4, 4]
        self.OP28 = nnq.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1])

        self.x29 = [batch_size, 512, 2, 2]
        self.OP29 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.x30 = [batch_size, 512]
        self.OP30 = nnq.Linear(512,1000)
        
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
        i = resnet(i)
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = msg_pb2_grpc.MsgServiceStub(channel)
            response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(i)))
            print("Client Received: {}".format(response.msg))
            i = int(response.msg)
            if i == 999:
                break
            elif i > 30:
                response = stub.GetMsg(msg_pb2.MsgRequest(msg='{}'.format(999)))
                break

    print('TEE Finished',file=f)
    print('TEE consumes: {}'.format(resnet.time),file=f)
 
if __name__ == '__main__':
    global resnet,f
    f = open('./result.txt', 'w')
    resnet = ResNet18(batch_size=32)
    for i in range(1):
        run()