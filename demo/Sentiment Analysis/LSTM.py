import torch
from torch import nn
from data import TokenEmbedding


class BiRNN(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,**kwargs):
        super(BiRNN,self).__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.encoder=nn.LSTM(embed_size,num_hiddens,num_layers=num_layers,bidirectional=True)
        self.decoder=nn.Linear(4*num_hiddens,2)

    def forward(self,x):
        # <x shape> (batch,num_steps)
        # 时间序列要求：输入形状： (seq_len, batch_size, input_size)
                    # 输出形状： (seq_len, batch_size, num_directions * hidden_size)
        # 但是 transformer并不进行要求，还是batch在第一维度，不要搞混
        embeddings=self.embedding(x.T)
        #PyTorch 的 RNN/LSTM 可以用 底层的 cudnn 实现 来加速计算，要求权重张量应该是连续的内存块
        #flatten_parameters 的作用将所有权重张量合并成一个连续的内存块
        self.encoder.flatten_parameters()
        # 在这里，我们丢掉最后一个时刻的隐状态，一般在状态传递的时候会使用他们。
        # <outputs shape> (seqlen,batch,num_hiddens*2) , LSTM的机制是当前层的输入来自上一层，当前层同时覆盖上一层
        outputs,_=self.encoder(embeddings)
        # 双向LSTM 第一个输出既是正向的开头也是反向的结尾；最后一个输出既是反向的开头也是正向的结尾。
        encoding=torch.cat((outputs[0],outputs[-1]),dim=1)
        outs=self.decoder(encoding)
        return outs

def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m,nn.LSTM):
        for param in m._flat_weights_names:
            if "weight_ih" in param:
                nn.init.xavier_uniform_(m._parameters[param])
            elif "weight_hh" in param:
                nn.init.orthogonal_(m._parameters[param])
            else :
                nn.init.zeros_(m._parameters[param])

def LSTM_model(embed_size,num_hiddens,num_layers,vocab,device):
    # 要求 embed_size和数据维度一致
    assert embed_size==100
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    net.apply(init_weight)
    embeding=TokenEmbedding()
    embeds=embeding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad=False
    net.to(device)
    return net




