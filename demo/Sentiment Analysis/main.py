import torch
from torch import nn
from LSTM import LSTM_model
from TextCNN import TextCNN_model
from General_Tools import display,train,load_model,predict
from data import load_imdb
from configure import configure
import os

if __name__ == '__main__':
    [x1,x2,x3,x4]=configure()

    path=None
    embed_size,num_hiddens,num_layers,num_steps=100,100,2,500
    device="cuda" if torch.cuda.is_available() else "cpu"
    batch_size ,num_epochs,lr = 64,5,0.01
    is_show,is_train,is_predict=x2,x3,x4
    train_iter, test_iter, vocab = load_imdb(batch_size=batch_size,num_steps=num_steps)

    os.makedirs("./Storage",exist_ok=True)

    if x1:
        # LSTM
        net=LSTM_model(embed_size,num_hiddens,num_layers,vocab,device)
        path = "./Storage/LSTM.pth"
    else:
        # TextCNN
        net=TextCNN_model(vocab,embed_size,kernel_sizes=[3,4,5],num_channels=[100,100,100],device=device)
        path = "./Storage/TextCNN.pth"

    print("\nloading...\n")
    net = load_model(net, path)
    if is_train:
        print("\ntraining..\n")
        optim=torch.optim.Adam(net.parameters(),lr=lr)
        criterion=nn.CrossEntropyLoss(reduction="none")
        train_loss,test_loss=train(net, optim, criterion, num_epochs, train_iter, test_iter, device,True,path)
        if is_show:
            display(train_loss,test_loss,os.path.basename(path).split(".")[0])

    if is_predict:
        print("\nPredicting...\n")
        predict(net,vocab,device,num_steps)