from torch.nn import Module
from torch import nn


# 构建网络
# mnist
class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 输出为6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 输出为16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
        )
        self.block_2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):  # 正向传播过程
        x = self.block_1(x)
        x = x.view(-1,16*5*5)
        x = self.block_2(x)
        return x
