import torch
from torch import nn

# 顺序块
class Sequential_(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            self._modules[str(idx)]=(module)
    def forward(self,x):
        for block in self._modules:
            x=block(x)
        return x
    def __getitem__(self, item):
        return self._modules[str(item)]

class FixedHiddenMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight=torch.randn((20,20),requires_grad=True)
        self.linear=nn.Linear(20,20)
    def forward(self,x):
        x=self.linear(x)
        x=nn.functional.relu(torch.mm(x,self.rand_weight)+1)
        x=self.linear(x)
        while x.abs().sum()>1:
            x/=2
        return x.sum()

class Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = FixedHiddenMlp()
        self.net2 = Sequential_(nn.Linear(20, 20), nn.ReLU(),
                           nn.Linear(20, 20))

    def forward(self,x):
        x1=self.net2.forward(x)
        x2=self.net1.forward(x1)
        return x2

# x= torch.rand(2, 20)

# 参数初始化
def init_weight(layer):
    if type(layer)==nn.Linear:
        nn.init.normal_(layer.bias,mean=0,std=0.01)
        nn.init.constant_(layer.bias,1)
# 自定义初始化函数

def init(layer):
    if isinstance(layer,nn.Linear):
        print("Init")
        nn.init.uniform_(layer.weight,-10,10)
        layer.weight.data *= (layer.weight.data.abs()>=5)


# torch.save(net.state_dict(),"mlp.params")
net=Sequential_(nn.Linear(20, 20), nn.ReLU(),
                           nn.Linear(20, 1))
net.load_state_dict(torch.load("mlp.params"))
net.eval()