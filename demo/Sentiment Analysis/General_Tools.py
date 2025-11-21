import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import truncate_pad
import glob

class Accumulator():
    def __init__(self,num):
        self.metric=[0]*num

    def add(self,*args):
        if args:
            for i,num in enumerate(args):
                self.metric[i]+=num
    def __getitem__(self, item):
        return self.metric[item]


class Recoder():
    def __init__(self,num):
        self.metric=[[]for _ in range(num)]

    def add(self,*args):
        if args:
            for i,num in enumerate(args):
                self.metric[i].append(num)
    def __getitem__(self, item):
        return self.metric[item]

def eval(net,criterion,test_iter,device):
    net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for x,y in test_iter:
            x,y=x.to(device),y.to(device)
            y_pred=net(x)
            loss=criterion(y_pred,y).sum().cpu().item()
            metric.add(loss,len(y))
    net.train()
    return metric[0]/metric[1]

def display(train_loss,test_loss,name):
    plt.plot(range(1,len(train_loss)+1),train_loss,"r-o",label="train")
    plt.plot(range(1,len(test_loss)+1),test_loss,"b-o",label="test")
    plt.legend()
    plt.title("Overall Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"./Storage/Overall_Loss_Curve_{name}.png")
    plt.show()


def train(net,optim,criterion,num_epochs,train_iter,test_iter,device,save=True,path="./Storage/LSTM.pth"):
    net.to(device)
    metric_all=Recoder(2)
    pbar=tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        metric=Accumulator(2)
        for x,y in train_iter:
            x,y=x.to(device),y.to(device)
            optim.zero_grad()
            y_pred=net(x)
            l=criterion(y_pred,y)
            l.sum().backward()
            optim.step()
            metric.add(l.sum().cpu().item(),len(y))

        train_loss=metric[0]/metric[1]
        test_loss=eval(net,criterion,test_iter,device)
        metric_all.add(train_loss,test_loss)

        pbar.update(1)
        pbar.set_description(f"<train loss> {train_loss},<test loss> {test_loss}")
    # return train_loss,test_loss
    if save:
        save_model(net,path=path)
    return metric_all[0],metric_all[1]

def save_model(net,path="./Storage/LSTM.pth"):
    torch.save(net.state_dict(),path)

def load_model(net,path="./Storage/LSTM.pth"):
    temp=None
    paths=glob.glob("./Storage/*pth")
    for mathed_path in paths:
        if os.path.basename(mathed_path)==os.path.basename(path):
            temp=True
    if temp:
        print("模型匹配成功")
        net.load_state_dict(torch.load(path,map_location="cpu"))
    return net

def predict(net,vocab,device,num_steps,examples=None):
    if not examples:
        examples=["this moive is bad",\
                  "I love you",\
                  "you are my sweet",\
                  "you are a very bad man!",\
                  "fuck you,man"]
    data=torch.tensor([truncate_pad(vocab[line.lower().split()],num_steps,vocab["<unk>"])\
                         for line in examples],device=device)
    net.to(device)
    labels=torch.argmax(net(data),dim=-1).cpu().numpy().tolist()
    labels=["positive" if label==1 else "negative" for label in labels]
    for example,label in zip(examples,labels):
        print(example,f"\n{label}\n")
    return labels





