import grpc
import msg_pb2
import msg_pb2_grpc
import time
from concurrent import futures

import tqdm

import torch
import torch.nn.quantized as nnq
import torch.nn as nn

class MobileNetv2(torch.nn.Module):
    
    def __init__(self,batch_size=8):
        super(MobileNetv2, self).__init__()
        
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
        
        

        self.x7 = [batch_size, 96, 112,112]
        self.OP7 = nnq.Conv2d(in_channels=96, out_channels=96, kernel_size=[3,3], stride=[2,2], padding=[1,1],groups=96)
        
        
        
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
        
        

        self.x83 = [batch_size, 320, 7,7]
        self.OP83 = nnq.Conv2d(in_channels=320, out_channels=1280, kernel_size=[1,1], stride=[1,1])

        self.x84 = [batch_size, 1280]
        self.OP84 = nnq.Linear(1280,1000)

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
        self.mobilenet = MobileNetv2(1)
        
    
    def GetMsg(self, request, context):
        i = int(request.msg)
        print("REE Received name: {}".format(i))
        i = self.mobilenet(i)
        
        if i == 999:
            print('Inference Finished')
            print('REE consumes: {}'.format(self.mobilenet.time))
            return msg_pb2.MsgResponse(msg='{}'.format(999))
        
        elif i > 84:
            print('Inference Finished')
            print('REE consumes: {}'.format(self.mobilenet.time))
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