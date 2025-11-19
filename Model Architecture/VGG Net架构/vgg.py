import torch
from torch import nn
from d2l import torch as d2l
# VGG NET的特殊之处在于：该网络架构使用了“块”的概念，对层进行了封装，有集成化的思想
def block(num_convs,channel_in,channel_out):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(channel_in,channel_out,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        channel_in=channel_out
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks=[]
    channel_in=1
    for (num_convs,channel_out) in conv_arch:
        conv_blks.append(block(num_convs,channel_in,channel_out))
        channel_in=channel_out
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
    )

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr,num_epochs,batch_size=0.01,10,128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
