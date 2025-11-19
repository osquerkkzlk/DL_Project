import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

class ResNet(nn.Module):
    def __init__(self,channel_in,channel_out,use_1x1_conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(channel_in,channel_out,kernel_size=3,
                             padding=1,stride=strides)
        self.conv2=nn.Conv2d(channel_out,channel_out,kernel_size=3,padding=1)
        if use_1x1_conv:
            self.conv3=nn.Conv2d(channel_in,channel_out,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(channel_out)
        self.bn2=nn.BatchNorm2d(channel_out)
    def forward(self,x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y+=x
        return F.relu(y)

def resnet_block(channel_in,channel_out,num_resnet,first_block=True):
    blk=[]
    for i in range(num_resnet):
        if i==0 and first_block:
            blk.append(ResNet(channel_in,channel_out,use_1x1_conv=True,strides=2))
        else:
            blk.append(ResNet(channel_out,channel_out))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7,stride=2,padding=3),
                 nn.BatchNorm2d(64),nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b2=nn.Sequential(*resnet_block(64,64,2,first_block=False))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

#nn.AdaptiveAvgPool2d(output_size) 将输入张量的空间尺寸（高度和宽度）池化为指定的 output_size。
net=nn.Sequential(b1,b2,b3,b4,b5,
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),
                  nn.Linear(512,10))

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())