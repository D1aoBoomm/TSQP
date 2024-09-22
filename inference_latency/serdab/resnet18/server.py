import grpc
import msg_pb2
import msg_pb2_grpc
import time
from concurrent import futures

import tqdm

import torch
import torch.nn.quantized as nnq
import torch.nn as nn

class ResNet18(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(ResNet18, self).__init__()
        self.x9 = [batch_size, 128, 28, 28]
        self.OP9 = nn.ReLU()
        
        self.x13 = [batch_size, 128, 28, 28]
        self.OP13 = nn.ReLU()
        
        self.x16 = [batch_size, 256, 14, 14]
        self.OP16 = nn.ReLU()
        
        self.x20 = [batch_size, 256, 14, 14]
        self.OP20 = nn.ReLU()
        
        self.x23 = [batch_size, 512, 7, 7]
        self.OP23 = nn.ReLU()
        
        self.x27 = [batch_size, 512, 4, 4]
        self.OP27 = nn.ReLU()
        
        self.x10 = [batch_size, 128, 28, 28]
        self.OP10 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x11 = [batch_size, 64, 28, 28]
        self.OP11 = nnq.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x12 = [batch_size, 128, 28, 28]
        self.OP12 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x14 = [batch_size, 128, 28, 28]
        self.OP14 = nnq.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x15 = [batch_size, 128, 28, 28]
        self.OP15 = nnq.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x17 = [batch_size, 256, 14, 14]
        self.OP17 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x18 = [batch_size, 128, 28, 28]
        self.OP18 = nnq.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        
        self.x19 = [batch_size, 256, 14, 14]
        self.OP19 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
        self.x21 = [batch_size, 256, 14, 14]
        self.OP21 = nnq.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        
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
        if i == -1:
            print('warm up for 1000 times')
            for j in range(0,1000):
                if j % 100 ==0:
                    print('warm {} times'.format(j))
                for k in range(0,85):
                    try:
                        getattr(self, 'OP{}'.format(k))(getattr(self, 'x{}'.format(i)))
                    except:
                        pass
            print('warmup finished')
            return 0
        
        #else
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
        
class MsgServicer(msg_pb2_grpc.MsgServiceServicer):
    
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet18(32)
        
    
    def GetMsg(self, request, context):
        i = int(request.msg)
        print("REE Received name: {}".format(i))
        i = self.resnet(i)
        
        if i == 999:
            print('Inference Finished')
            print('REE consumes: {}'.format(self.resnet.time))
            return msg_pb2.MsgResponse(msg='{}'.format(999))
        
        elif i > 30:
            print('Inference Finished')
            print('REE consumes: {}'.format(self.resnet.time))
            return msg_pb2.MsgResponse(msg='{}'.format(999))
        
        return msg_pb2.MsgResponse(msg='{}'.format(i))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    msg_pb2_grpc.add_MsgServiceServicer_to_server(MsgServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('REE Server Started')
    time.sleep(60000)
    
if __name__ == '__main__':
    serve()