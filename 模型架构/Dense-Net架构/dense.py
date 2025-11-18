import torch
from torch import nn
from d2l import torch as d2l

from 模型架构.AlexNet架构.alex import batch_size


#DenseNet使用了ResNet改良版的 " 批量规范化、激活和卷积 " 架构



def conv_block(channel_in,channel_out):
    return nn.Sequential(
        nn.BatchNorm2d(channel_in),
        nn.ReLU(),
        nn.Conv2d(channel_in,channel_out,kernel_size=3,padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self,num_convs,channel_in,channel_out):
        super(DenseBlock,self).__init__()
        layer=[]
        for i in range(num_convs):
            layer.append(conv_block(channel_out*i+channel_in,channel_in))
        self.net=nn.Sequential(*layer)

    def forward(self,x):
        for blk in self.net:
            y=blk(x)
            x=torch.cat((x,y),dim=1)
        return x

# 一位的拼接，只会使得网络的复杂度不断增大，随之而来的过拟合风险也会增大，
#  所以我们引入了过渡层，利用步幅为 2 的汇聚层减半高和宽，减小模型复杂度

def transition_block(channel_in,channel_out):
    return nn.Sequential(
        nn.BatchNorm2d(channel_in),nn.ReLU(),
    # 注意在这里，可以通过控制输入通道数和输出通道数，达到减小通道数的目的，因为我们设置了大小为1的卷积核
        nn.Conv2d(channel_in,channel_out,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )


# 开始构建
# DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层。
layer1=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

num_channels, num_growth=64,32
num_convs_list=[4]*4
blks=[]
for i,num_conv in enumerate(num_convs_list):
    blks.append(DenseBlock(num_conv,num_channels,num_growth))
    num_channels+=num_growth*num_channels
    if i!=len(num_convs_list)-1:
        blks.append(transition_block(num_channels,num_channels//2))
        num_channels=num_channels//2


net=nn.Sequential(
    layer1,*blks,
    nn.BatchNorm2d(num_channels),nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.LazyLinear(10)
)

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())