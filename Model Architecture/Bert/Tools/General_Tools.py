import torch
from torch import nn
from tqdm import tqdm
import math


L=[]
# ✔️
def truncate_pad(line, num_steps, padding_token):
    """截断或者填充序列,以保证序列长度一致"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

# ✔️
class FFN(nn.Module):
    """
    基于位置的前馈网络,仅改变输入的最后一个维度
    """

    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

#✔️
class AddNorm(nn.Module):
    """层归一化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(normalized_shape)

    def forward(self, X, y):
        return self.LN(self.dropout(y) + X)

#✔️
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
# ✔️
def masked_softmax(X, valid_lens):
    """
    有掩码的 softmax,以消除pad或者无关因素的影响
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, X.shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
# ✔️
class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        dim = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dim)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# ✔️
def transpose_qkv(X, num_heads):
    """transform shape"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# ✔️
def transpose_output(X, num_heads):
    """反变换"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# ✔️
def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# ✔️
class Accumulator():
    def __init__(self, num):
        self.Record = [0] * num

    def add(self, *args):
        if args:
            for i in range(len(args)):
                self.Record[i] += args[i]

    def __getitem__(self, idx):
        return self.Record[idx]

# ✔️
class MultiHeadAttention(nn.Module):
    """"多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output = transpose_output(output, self.num_heads)
        return self.W_o(output)

def get_loss(net,loss,vocab_size,tokens_X,segments_X,valid_lens_X,\
             pred_positions_X,mlm_weights_X,mlm_y,nsp_y):
    _,mlm_y_pred,nsp_y_pred=net(tokens_X,segments_X,valid_lens_X.reshape(-1),\
                                pred_positions_X)
    mlm_l=loss(mlm_y_pred.reshape(-1,vocab_size),mlm_y.reshape(-1))*\
                mlm_weights_X.reshape(-1,1)
    mlm_l=mlm_l.sum()/(mlm_weights_X.sum()+1e-8)
    nsp_l=loss(nsp_y_pred,nsp_y)
    l=mlm_l+nsp_l
    return mlm_l,nsp_l,l


def train_bert(train_iter, net, loss, vocab_size, device, num_steps,eval=False):
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量
    metric = Accumulator(4)
    num_steps_reached = False
    mlm_L,nsp_L,step=[],[],0
    net=net.to(device)
    if eval:
        step,mlm_L,nsp_L=load_checkpoint(net,trainer)
    if torch.cuda.device_count()>1:
        net=nn.DataParallel(net)

    pbar=tqdm(total=num_steps,desc="<training>")
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y, nsp_y = mlm_Y.to(device), nsp_y.to(device)
            trainer.zero_grad()
            mlm_l, nsp_l, l = get_loss(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0],1)
            step += 1
            pbar.update(1)
            pbar.set_description(f"{step+1}/{num_steps+1}",
                                 f'\tMLM loss {metric[0] / metric[3]:.3f}, '
                                 f'\tNSP loss {metric[1] / metric[3]:.3f}'
                                 )
            # 当导入信息时，应该从下一个step开始
            mlm_L.append(metric[0] / metric[3])
            nsp_L.append(metric[1] / metric[3])
            save_checkpoint(net,trainer,step+1,mlm_L,nsp_L)

            if step == num_steps:
                num_steps_reached = True
                break
    print("\n------------Overall Loss------------")
    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')


# 设置 checkpoint 可以有效防止训练突然中断而导致 state 未被保存
def save_checkpoint(net,optim,step,mlm_L,nsp_L,path="./Storage/checkpoint.pth"):
    torch.save({"model":net.module.state_dict() if torch.cuda.device_count()>1 else net.state_dict(),
                "optim":optim.state_dict(),
                "step":step,
                "mlm_L":mlm_L,
                "nsp_L":nsp_L},path)

def load_checkpoint(net,optim,path="./Storage/checkpoint.pth"):
    import numpy as np
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    print("loading...",end="    ")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    net.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    step=checkpoint["step"]
    mlm_L=checkpoint["mlm_L"]
    nsp_L=checkpoint["nsp_L"]
    print(f"from {step+1} step starting")
    return step,mlm_L,nsp_L
