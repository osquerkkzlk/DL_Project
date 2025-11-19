from collections import Counter
from torch.utils.data import TensorDataset,DataLoader
import torch
import os

def read_data():
    """载入“英语－法语”数据集"""
    data_dir = r"E:\DL_files\NLP\fra-eng\fra-eng"
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()

def preprocess(text):
    def no_space(char, prev_char):
        return char in set(',.!?„“;:') and prev_char != ' '
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts)>=2:
            source.append(parts[0].lower().split(' '))
            target.append(parts[1].lower().split(' '))
    return source, target


class Vocab:
    """建立词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """token->idx"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """idx->token"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def count_corpus(self, tokens):
        """统计词元频率"""
        if len(tokens)==0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # tokens应该是一个列表形式
        return Counter(tokens)

def load_array(src_vocab,tgt_vocab,src,tgt,batch_size,num_steps,is_train=True):
    def build_array(lines, vocab, num_steps):
        lines = [vocab[l] for l in lines]
        lines = [l + [vocab['<eos>']] for l in lines]
        array = torch.tensor([
            truncate_pad(l, num_steps, vocab['<pad>']) for l in lines
        ])  # shape: (N, num_steps)

        valid_lens = (array != vocab['<pad>']).sum(1)
        return array, valid_lens


    def truncate_pad(line,num_steps,padding_token):
        """截断或者填充序列，以保证序列长度一致"""
        if len(line)>num_steps:
            return line[:num_steps]
        return line+[padding_token]*(num_steps-len(line))

    # 导入
    src_array,src_valid_lens=build_array(src,src_vocab,num_steps)
    tgt_array,tgt_valid_lens=build_array(tgt,tgt_vocab,num_steps)
    data_arrays=(src_array,src_valid_lens,tgt_array,tgt_valid_lens)

    return DataLoader(TensorDataset(*data_arrays),batch_size,shuffle=is_train)

def load_data(batch_size, num_steps, num_examples=None):
    """
    :param batch_size:
    :param num_steps:
    :param num_examples:
    :return: train_iter,test_iter,val_iter,src_vocab,tgt_vocab
    """
    text = read_data()
    train_src, train_tgt = tokenize(preprocess(text),num_examples)

    # 建立词表，接下来应该转换词元
    src_vocab = Vocab(train_src, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(train_tgt, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])

    train_iter = load_array(src_vocab, tgt_vocab, train_src, train_tgt, batch_size, num_steps, True)
    return train_iter,src_vocab,tgt_vocab



