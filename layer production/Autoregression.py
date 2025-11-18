import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,))
d2l.plot(time,x,"time","x",xlim=[1,T+1],figsize=(6,3))

interval=4
features=torch.zeros((T-interval,interval))
for i in range(interval):
    features[:,i]=x[i:T-interval+i]
labels=x[interval:].reshape(-1,1)
batch_size,n_train=16,600
# 只取前n_train个数据训练
train_iter=d2l.load_array((features[:n_train],labels[:n_train]),
                          batch_size=batch_size,is_train=True)


# 初始化网络权重
def init_weight(layer):
    if type(layer)==nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
def net():
    net=nn.Sequential(nn.Linear(4,256),
                      nn.ReLU(),
                      nn.Linear(256,10),
                      nn.ReLU(),
                      nn.Linear(10,1))
    net.apply(init_weight)
    return net

# 需要注意的是，pytorch中的损失计算，“none”表示返回每个元素各自的损失，是一个张量，
# “mean”会返回平均损失，是一个标量
loss=nn.MSELoss(reduction="none")

def train(net,train_iter,loss,epochs,lr):
    optimizer=torch.optim.Adam(net.parameters(),lr)
    for epoch in range(epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l=loss(net(x),y)
            l.sum().backward()
            optimizer.step()
        print(f"epoch:{epoch+1},"
              f"loss:{d2l.evaluate_loss(net,train_iter,loss):f}")
net=net()
train(net,train_iter,loss,5,0.01)

# 进行模型预测
preds=net(features)
d2l.plot([time,time[interval:]],
          [x.detach().numpy(),preds.detach().numpy()],
         "time","x",legend=["data","preds"],
         xlim=[1,T],figsize=(6,3))
plt.show()

multiple_step_preds=torch.zeros(T)
multiple_step_preds[:n_train+interval]=x[:n_train+interval]
for i in range(n_train+interval,T):
    multiple_step_preds[i]=net(
        multiple_step_preds[i-interval:i].reshape((1,-1))
    )
d2l.plot([time,time[interval:],time[n_train+interval:],],
         [x,preds.detach().numpy(),multiple_step_preds[n_train+interval:].detach().numpy()],
         "time","x",legend=["data","one-step-preds",f"{interval}-steps-preds",],
         xlim=[1,1000],figsize=(6,3))
plt.show()

max_steps=64
features=torch.zeros((T-interval-max_steps+1,interval+max_steps))
for i in range(interval):
    features[:,i]=x[i:i+T-interval-max_steps+1]
for i in range(interval,interval+max_steps):
    features[:,i]=net(features[:,i-interval:i]).reshape((1,-1))

steps=(1,4,4,16,64)
# 预测 i 步之后的数据，time时间要与之相对应才可以，还得是grok
d2l.plot([time[interval+i-1 : T-max_steps+i]for i in steps],
         [features[:,interval+i-1].detach().numpy() for i in steps],
         "time","x",legend=[f"{i}-step-preds" for i in steps],
         xlim=[5,1000],figsize=(6,3))
plt.show()
