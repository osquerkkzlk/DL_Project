import torch
from torch import nn
from data import TokenEmbedding


class TextCNN(nn.Module):
    def __init__(self,vocab_size,embed_size,kernel_sizes,num_channels,**kwargs):
        super(TextCNN,self).__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.constant_embedding=nn.Embedding(vocab_size,embed_size)
        self.dropout=nn.Dropout(0.5)
        self.decoder=nn.Linear(sum(num_channels),2)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.relu=nn.ReLU()
        self.convs=nn.ModuleList()
        for c,k in zip(num_channels,kernel_sizes):
            self.convs.append(nn.Conv1d(2*embed_size,c,k))
    def forward(self,x):
        embeddings=torch.cat([self.embedding(x),self.constant_embedding(x)],dim=2)
        # cov1d 要求的shape (batch_size, channels, sequence_length)
        embeddings=embeddings.permute(0,2,1)
        encoding=torch.cat([self.relu(self.pool(conv(embeddings))).squeeze(-1) for conv in self.convs],dim=-1)
        outputs=self.decoder(self.dropout(encoding))
        return outputs

def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m,nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

def TextCNN_model(vocab,embed_size,kernel_sizes,num_channels,device):
    net=TextCNN(len(vocab),embed_size,kernel_sizes,num_channels)
    net.apply(init_weight)
    embedding=TokenEmbedding()
    embeds=embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    # embedding 没有 bias，只有 weight
    net.constant_embedding.weight.requires_grad=False
    net.to(device)
    return net


