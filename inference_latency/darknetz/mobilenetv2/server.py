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
        
        self.x22 = [batch_size, 192, 28,28]
        self.OP22 = nnq.Conv2d(in_channels=192, out_channels=192, kernel_size=[3,3], stride=[1,1],padding=[1,1],groups=192)
        
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
        
        self.x58 = [batch_size, 576, 14,14]
        self.OP58= nnq.Conv2d(in_channels=576, out_channels=96, kernel_size=[1,1], stride=[1,1])
        
        self.x59 = [batch_size, 96, 14,14]
        self.OP59 = nnq.Conv2d(in_channels=96, out_channels=576, kernel_size=[1,1], stride=[1,1])
        
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