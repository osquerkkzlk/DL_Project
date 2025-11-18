import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(x,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not torch.is_grad_enabled():
        x_preds=(x-moving_mean)/torch.sqrt_(moving_var+eps)
    else:
        assert len(x.shape) in (2,4)
        if len(x.shape)==2:
            mean=x.mean(dim=0)
            var=((x-mean)**2).mean(dim=0)
        else:
            mean=x.mean(dim=(0,2,3),keepdim=True)
            var=((x-mean)**2).mean(dim=(0,2,3),keepdim=True)
        # 在训练模式下，用当前的均值和方差做标准化
        x_preds=(x-mean)/torch.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1-momentum)*mean
        moxing_var=momentum*moving_var+(1-momentum)*var
    y=gamma*x_preds+beta
    return y,moving_mean.detach(),moving_var.detach()

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super().__init__()
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.ones(shape)

    def forward(self,x):
        # 一定要记住，他并不是就地操作
        self.moving_mean=self.moving_mean.to(x.device)
        self.moving_var=self.moving_var.to(x.device)

        y,self.moving_mean,self.moving_var=batch_norm(x,self.gamma,self.beta,self.moving_mean,
                                                      self.moving_var,eps=1e-5,momentum=0.9)
        return y
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))