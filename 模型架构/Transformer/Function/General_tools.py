import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import pickle,json,math
import matplotlib.pyplot as plt
import numpy as np

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
class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        quaries, keys = self.W_q(queries), self.W_k(keys)
        features = quaries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

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

# ✔️
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32). \
                reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.t=0

    def forward(self, X,eval=False):
        X = X + self.P[:, self.t:X.shape[1]+self.t, :].to(X.device)
        if eval:
            self.t+=1
        return self.dropout(X)

# ✔️
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# ✔️
class MaskedSoftmaxLoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数
    """
    def forward(self, pred, label, valid_lens):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_lens)
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxLoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# ✔️
def bleu(pred_seq, label_seq, k):
    """
    计算 BLEU Score
    """
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# ✔️
def train(net, train_iter, lr, num_epochs, src_vocab, tgt_vocab, device, num_steps,samples=None):
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weight)
    net = net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxLoss()
    net.train()

    global L

    for epoch in tqdm(range(num_epochs), desc=f"<training>🤫"):
        metric = Accumulator(2)
        for batch in train_iter:
            net.train()
            optim.zero_grad()
            X, X_valid_lens, y, y_valid_lens = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab["<bos>"]] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], 1)
            y_pred, _ = net(X, dec_input, X_valid_lens)
            l = loss(y_pred, y, y_valid_lens)
            l.sum().backward()
            grad_clipping(net, 1)
            optim.step()
            with torch.no_grad():
                num_tokens = y_valid_lens.sum()
                metric.add(l.sum(), num_tokens)
        temp=metric[0] / metric[1]
        L.append(temp.cpu().numpy())
        if epoch%10==0:
            print(f"Current Loss is {temp:.3f}\n")

    Eval(num_epochs,net, src_vocab, tgt_vocab, num_steps, device,samples)

# ✔️
def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    net.decoder.pos_encoding.t=0
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq=[]
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state,eval)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X=Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    net.train()
    return ' '.join(tgt_vocab.to_tokens(output_seq))


def Eval(num_epochs,net, src_vocab, tgt_vocab, num_steps, device,samples=None):
    if not samples:
        samples = [('go .', 'va !'),
                 ('hi .', 'salut !'),
                 ('run !', 'cours !'),
                 ('hello .', 'bonjour .'),
                 ('i won !', "j'ai gagné !"),
                 ("i'm ok .", 'je vais bien .'),
                 ('thank you .', 'merci .'),
                 ('are you ok ?', 'ça va ?'),
                 ("i'm home .", 'je suis rentré .'),
                 ('we won .', 'nous avons gagné .')]
    metric=Accumulator(2)
    for src,tgt in samples:
        pred=predict(net, src, src_vocab, tgt_vocab, num_steps, device)
        Single_bleu=bleu(pred,tgt,2)
        print(f"{pred:20}-------->,{tgt:20}\t\t{Single_bleu:.4f}")
        metric.add(1,Single_bleu)
    print(f"\nOverall_score is {metric[1]/metric[0]:.4f}")

    # display
    plt.plot(np.arange(1, num_epochs + 1), L, "b-o")
    plt.title("Loss of epoches")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.savefig(r".\image\Loss.png")
    plt.show()

def save(net,src_vocab,tgt_vocab,config=None):
    torch.save(net.state_dict(), r"Storage\model.pth")
    with open(r"Storage\src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(r"Storage\tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    if config:
        with open(r"Storage\config.json", "w") as f:
            json.dump(config, f)