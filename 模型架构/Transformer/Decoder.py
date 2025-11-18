import torch
from torch import nn
import math
from Function.General_tools import MultiHeadAttention,AddNorm,FFN,PositionalEncoding

#✔️
class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_inputs, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """解码器需要编码器提取的全局特征state以及当前输入特征"""
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat([state[2][self.i], X], axis=1)
        state[2][self.i] = key_values
        # training属性是nn.Module自动维护的，当你shezhiModel.train()的时候该属性就会被自动设置为True
        # module.training = True
        if self.training:
            # 生成 mask 掩码，用于训练的时候遮蔽后面的词元
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+ 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 在这里，X作为 query ，key_values作为键值对，表示 key_values的每个位置对当前查询的影响权重
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        y = self.addnorm1(X, X2)
        # enc_outputs shape:(batch,query_num,num_hiddens)
        y2 = self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(y, y2)
        return self.addnorm3(Z, self.ffn(Z)), state

#✔️
class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,\
                ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,**kwargs):
        super(TransformerDecoder,self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}",DecoderBlock(key_size,query_size,value_size,num_hiddens,\
                                                           norm_shape, ffn_num_inputs,ffn_num_hiddens,\
                                                           num_heads,dropout,i))
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]

    def forward(self,X,state,eval=False):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens),eval)
        for blk in self.blks:
            X,state=blk(X,state)
        return self.dense(X),state
