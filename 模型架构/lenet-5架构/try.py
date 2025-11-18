from random import triangular

import torch
from torch import nn
from ai import accumulaters,accuracy,load_fashion_mnist,MatplotAnimator,Timer
def main():
    # 模型架构
    net=nn.Sequential(
        nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(16*5*5,120),nn.Sigmoid(),
        nn.Linear(120,84),nn.Sigmoid(),
        nn.Linear(84,10)
    )

    x=torch.rand(size=(1,1,28,28),dtype=torch.float)
    for layer in net:
        x=layer(x)
        print(layer.__class__.__name__,x.shape)

    batch_size=256
    train_iter,test_iter=load_fashion_mnist(batch_size=batch_size)

    def evaluate_accuracy(net,data_iter,device=None):
        # 把数据转移到gpu
        if isinstance(net,nn.Module):
            net.eval()
            if not device:
                device=next(iter(net.parameters())).device
            metric=accumulaters(2)
            with torch.no_grad():
                for x,y in data_iter:
                    if isinstance(x,list):
                        x=[x_.to(device) for x_ in x]
                    else:
                        x=x.to(device)
                    y=y.to(device)
                    metric.add(accuracy(net(x),y),y.numel())
            return metric[0]/metric[1]

    def train_ch6(net,train_iter,tet_iter,num_epochs,lr,device):
        def init_weight(layer):
            if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        net.apply(init_weight)
        print("training on",device)
        net.to(device)
        optimizer=torch.optim.SGD(net.parameters(),lr=lr)
        loss=nn.CrossEntropyLoss()
        animator=MatplotAnimator(xlabel="epoch",xlim=[1,num_epochs+1],
                              legend=["train_loss","train_acc","test_acc"])
        timer,num_batches=Timer(),len(train_iter)
        for epoch in range(num_epochs):
            metric=accumulaters(3)
            net.train()
            for i ,(x,y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                x,y=x.to(device),y.to(device)
                y_preds=net(x)
                l=loss(y_preds,y)
                # 对损失函数做反向传播
                l.backward()
                optimizer.step()
                #只对“纯日志 / 纯指标”块用 torch.no_grad()；
                with torch.no_grad():
                    metric.add(l*x.shape[0],accuracy(y_preds,y),x.shape[0])
                timer.stop()
                train_l=metric[0]/metric[-1]
                train_acc=metric[1]/metric[-1]
                if (i+1) %(num_batches //5 )==0 or i==num_batches-1:
                    animator.add(epoch+(i+1)/num_batches,
                                 (train_l,train_acc,None))
            test_acc=evaluate_accuracy(net,test_iter)
            animator.add(epoch+1,(None,None,test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')

    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr,torch.device(f"cuda:{0}"))

if __name__ =="__main__":
    main()