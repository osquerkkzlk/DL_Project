import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import numpy as np
from IPython import display
import matplotlib
import time
from torch import nn
#前置
from tqdm import tqdm
num_works=4
matplotlib.use('QtAgg')


#显示进程
def with_progress_bar(iterable, desc="处理中"):
    return tqdm(iterable, desc=desc, ascii=False, ncols=100)
# 小批量训练
def train_epochs(net,train_iter,loss,optimizer):
    net.to("cuda")
    metric=accumulaters(3)
    for x, y in train_iter:
        x,y=x.to("cuda"),y.to("cuda")
        y_hat=net(x)
        loss_=loss(y_hat,y)
        optimizer.zero_grad()
        loss_.mean().backward()
        optimizer.step()
        metric.add(float(loss_.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

# 记录关键数据
class accumulaters:
    def __init__(self,n):
        self.metric=[0.]*n
    def add(self,*args):
        self.metric=[a + float(b) for a,b in zip(self.metric,args)]
    def __getitem__(self,i):
        return self.metric[i]

#正式训练
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    net.to("cuda")
    animator = MatplotAnimator(xlabel='epoch', xlim=[1, num_epochs],
                               ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    for epoch in with_progress_bar(range(num_epochs)):
        train_metrics = train_epochs(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        time.sleep(0.001)
    animator.close()
    print(f"Final: train loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}")
    return net

# 求出精度
def evaluate_accuracy(net, data_iter):
    net.to("cuda")
    """计算在指定数据集上模型的精度"""
    net.eval()  # 将模型设置为评估模式
    metric = accumulaters(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X,y=X.to("cuda") , y.to("cuda")
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def accuracy(y_hat,y):
    y1=y_hat.argmax(axis=1)
    return sum(y==y1).float().sum().item()

# 动态绘图类
class MatplotAnimator:
    def __init__(self, xlabel=None, ylabel=None, legend=None,
                 xlim=None, ylim=None, fmts=('-', 'm--', 'g-.', 'r:'),
                 figsize=(6, 4)):
        # ——关键：打开交互模式——
        plt.ion()                              # 不阻塞主线程
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel, self.ylabel, self.legend = xlabel, ylabel, legend
        self.fmts, self.xlim, self.ylim = fmts, xlim, ylim
        self.X, self.Y = [], []

    def add(self, x, ys):
        if not hasattr(ys, "__len__"):
            ys = [ys]
        if not self.X:
            self.X = [[] for _ in ys]
            self.Y = [[] for _ in ys]
        for i, y in enumerate(ys):
            self.X[i].append(x)
            self.Y[i].append(y)

        self.ax.cla()                          # 清除旧曲线
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.ax.plot(x_vals, y_vals, fmt)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim: self.ax.set_xlim(*self.xlim)
        if self.ylim: self.ax.set_ylim(*self.ylim)
        if self.legend: self.ax.legend(self.legend)

        self.fig.canvas.draw()                 # 刷新
        self.fig.canvas.flush_events()         # 立即更新

    def close(self):
        plt.ioff()         # 关闭交互模式
        plt.show()          # 训练结束后阻塞显示

# 生成 fashion-mnist 数据
def load_fashion_mnist(batch_size,resize=None):
    """
    :param batch_size: 批量大小
    :param resize: 将图像重设尺寸
    :return: 训练数据和测试数据
    """
    transform=[torchvision.transforms.ToTensor()]
    if resize:
        transform.insert(0,torchvision.transforms.Resize())
    #  把前面收集到的一系列变换组合，用于后续处理
    transform=torchvision.transforms.Compose(transform)
    train_data=torchvision.datasets.FashionMNIST(root=r"E:\fashion mnist",
                                                 transform=transform,train=True,download=True)
    test_data = torchvision.datasets.FashionMNIST(root=r"E:\fashion mnist",
                                                   transform=transform,train=False, download=True)
    return (
        torch.utils.data.DataLoader(train_data,batch_size,shuffle=True,num_workers=num_works),
        torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=num_works)
    )

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
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

    def evaluate_accuracy(net,data_iter,device=None):
        # 把数据转移到gpu
        if isinstance(net,nn.Module):
            net.eval()
            if not device:
                device=next(iter(net.parameters())).device
            with torch.no_grad():
                for x,y in data_iter:
                    metric=accumulaters(2)
                    if isinstance(x,list):
                        x=[x_.to(device) for x_ in x]
                    else:
                        x=x.to(device)
                    y=y.to(device)
                    metric.add(accuracy(net(x),y),y.numel())
            return metric[0]/metric[1]