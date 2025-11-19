from torch import nn
import math
from Function.General_tools import MultiHeadAttention,AddNorm,FFN,PositionalEncoding

#✔️
class EncoderBlock(nn.Module):
    """编码块"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,
                norm_shape,ffn_num_inputs,ffn_num_hiddens,
                num_heads,dropout,use_bias=False,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention=MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=FFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lens):
        # 要求 key_size==num_hiddens
        y=self.addnorm1(X,self.attention(X,X,X,valid_lens))
        # <output>shape: (batch,query_num,num_hiddens)
        return self.addnorm2(y,self.ffn(y))

#✔️
class TransformerEncoder(nn.Module):
    """Encoer( core )🤫"""
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,
                norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias=False,**kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}",EncoderBlock(key_size,query_size,value_size,num_hiddens,
                                                          norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,
                                                          dropout,use_bias))

    def forward(self,X,valid_lens,*args):
        # 防止输入信号由顺序信号占主导，导致噪声特别大无法训练
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X=blk(X,valid_lens)
        return X




