import torch
from torch import nn
from .General_Tools import  MultiHeadAttention,AddNorm,FFN

def get_tokens_and_segments(tokens_a,tokens_b=None):
    '''
    获取输入序列的词元及片段索引
    :param tokens_a:
    :param tokens_b:
    :return: tokens,segments
    '''
    tokens=["<cls>"]+tokens_a+["<sep>"]
    segments=[0]*(len(tokens_a)+2)
    if tokens_b is not None:
        tokens+=tokens_b+["sep"]
        segments+=[1]*(len(tokens_b)+1)
    return tokens,segments

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

class BertEncoder(nn.Module):
    """Bert 编码器"""
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_inputs,
                 ffn_num_hiddens,num_heads,num_layers,dropout,max_len=1_000,
                 key_size=768,query_size=768,value_size=768,**kwargs):
        super(BertEncoder,self).__init__(**kwargs)
        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding=nn.Embedding(2,num_hiddens)
        self.blks=nn.Sequential()
        self.pos_embedding=nn.Parameter(torch.randn(1,max_len,num_hiddens))
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}",EncoderBlock(key_size,query_size,value_size,
                                                           num_hiddens,norm_shape,ffn_num_inputs,
                                                           ffn_num_hiddens,num_heads,dropout,True))

    def forward(self,tokens,segments,valid_lens):
        # < X shape > (batch,max_len,num_hiddens)
        X=self.token_embedding(tokens)+self.segment_embedding(segments)
        X=X+self.pos_embedding.data[:,:X.shape[1],:]
        for blk in self.blks:
            X=blk(X,valid_lens)
        return X

class MaskLM(nn.Module):
    """BERT的掩蔽模型"""
    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super(MaskLM,self).__init__(**kwargs)
        self.mlp=nn.Sequential(nn.Linear(num_inputs,num_hiddens),
                               nn.ReLU(),
                               nn.LayerNorm(num_hiddens),
                               nn.Linear(num_hiddens,vocab_size))

    def forward(self,X,pred_positions):
        num_pred_positions=pred_positions.shape[1]
        pred_positions=pred_positions.reshape(-1)
        batch_size=X.shape[0]
        batch_idx=torch.arange(batch_size)
        batch_idx=torch.repeat_interleave(batch_idx,num_pred_positions)
        masked_X=X[batch_idx,pred_positions]
        masked_X=masked_X.reshape((batch_size,num_pred_positions,-1))
        mlm_y_pred=self.mlp(masked_X)
        return mlm_y_pred

class NextSentencePred(nn.Module):
    """预测两个句子是否相邻"""
    def __init__(self,num_inputs,**kwargs):
        super(NextSentencePred,self).__init__(**kwargs)
        self.output=nn.Linear(num_inputs,2)
    def forward(self,X):
        return self.output(X)

class BertModel(nn.Module):
    """Bert  Model"""
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_inputs,
                 ffn_num_hiddens,num_heads,num_layers,dropout,max_len=1_000,
                 key_size=768,query_size=768,value_size=768,hid_in_features=768,
                 mlm_in_features=768,nsp_in_features=768):
        super(BertModel,self).__init__()
        self.encoder=BertEncoder(vocab_size,num_hiddens,norm_shape,ffn_num_inputs,
                             ffn_num_hiddens,num_heads,num_layers,dropout,max_len,
                             key_size,query_size,value_size)
        self.hidden=nn.Sequential(nn.Linear(hid_in_features,mlm_in_features),
                                  nn.Tanh())
        self.mlm=MaskLM(vocab_size,num_hiddens,mlm_in_features)
        self.nsp=NextSentencePred(nsp_in_features)

    def forward(self,tokens,segments,valid_lens=None,pred_positions=None):
        encoded_X=self.encoder(tokens,segments,valid_lens)
        if pred_positions is not None:
            mlm_y_pred=self.mlm(encoded_X,pred_positions)
        else:
            mlm_y_pred=None
        # Bert是注意力自回归模型，所以他的任何位置的字符实际上已经包含了整个位置的信息（即注意力权重）
        nsp_y_pred=self.nsp(self.hidden(encoded_X[:,0,:]))
        return encoded_X,mlm_y_pred,nsp_y_pred





