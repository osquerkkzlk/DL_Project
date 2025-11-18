import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torchvision
from AI_ToolBox import train,load_fashion_mnist

#参数设置

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
drop_=0.3
epochs=10
lr=0.01
batch_size=256
num_works=4

# 构建模型
net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,10)
)

# 权重初始化
def init(layer):
    if type(layer)==nn.Linear:
        nn.init.normal_(layer.weight,std=0.01)
net.apply(init)

# 开始训练
def main():
    train_iter, test_iter = deal_data(batch_size)
    loss=nn.CrossEntropyLoss(reduction="none")
    optimizer=torch.optim.SGD([
            {"params": net[1].weight, 'weight_decay': 3},
            {"params": net[1].bias}], lr=lr)
    train(net,train_iter,test_iter,loss,epochs,optimizer)

if __name__=="__main__":
    main()


