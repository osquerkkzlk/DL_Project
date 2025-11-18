import collections
import re
from d2l import torch as d2l
import torch

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
def tokenize(lines,token="word"):
    if token=="word":
        return [line.split()for line in lines]
    elif token=="char":
        return [list(line)for line in lines]
    else:
        print(f"错误,{token} 是位置类型")

token=tokenize(lines,"word")

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if not tokens:
            tokens=[]
        if not reserved_tokens:
            reserved_tokens=[]
        counter=count_corpus(tokens)
        self._token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        # 未知词元的索引位0
        # self.idx_to_token：从索引到词元的映射列表。
        # self.token_to_idx：从词元到索引的字典映射。 它们是词汇表的核心，
        # 用于将文本中的词元（token）与唯一的整数索引（index）对应起来。
        self.idx_to_token=["<unk>"]+reserved_tokens
        self.token_to_idx={token:idx for idx,token in enumerate(self.idx_to_token)}

        for token ,freq in self._token_freqs:
            if freq<min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, item):
        if not isinstance(item,(list,tuple)):
            return self.token_to_idx.get(item,self.unk)
        return [self.__getitem__(token) for token in item]

    def to_token(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indice] for indice in indices]

    @property
    def unk(self):
        return 0
    @property
    def token_freaqs(self):
        return self._token_freqs
def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line ]
    return collections.Counter(tokens)
#讲词元列表展平成一个列表
vocab=Vocab(token)


def load_corpors_time_machine(max_tokens=-1):
    lines=read_time_machine()
    tokens=tokenize(lines,"char")
    vocab =Vocab(tokens)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return corpus,vocab

corpus,vocab =load_corpors_time_machine()



