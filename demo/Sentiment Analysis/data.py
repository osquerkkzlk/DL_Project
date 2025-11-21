from torch.utils.data import DataLoader,TensorDataset
from collections import Counter
import os,zipfile,glob,tarfile
import torch


def extract_imdb():
    dir_path=r"..\..\data\aclImdb_v1.tar.gz"
    final_path=r"..\..\data\aclImdb"
    os.makedirs(final_path,exist_ok=True)
    if os.listdir(final_path):
        return final_path

    print("IMDB Extracting...")
    with tarfile.open(dir_path,"r:gz") as tar:
        tar.extractall(path=final_path)
    print("IMDB Succeed...")
    return final_path

def read_imdb(is_train=True):
    datas,labels=[],[]
    dir=extract_imdb()

    for label in ("pos","neg"):
        folder_name=os.path.join(dir,"aclImdb","train"if is_train else"test",label)
        for file in os.listdir(folder_name):
            with open (os.path.join(folder_name,file),"rb") as f:
                temp=f.read().decode("utf-8").replace("\n","")
                datas.append(temp)
                labels.append(1 if label=="pos" else 0)
    return datas,labels

def tokenize(lines, token="word"):
    return [line.split() if token == 'word' else list(line) for line in lines]

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

def truncate_pad(line,num_steps,padding_token):
    """截断或者填充序列，以保证序列长度一致"""
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))


def load_imdb(batch_size,num_steps):
    train,test=read_imdb(True),read_imdb(False)
    train_tokens,test_tokens=tokenize(train[0]),tokenize(test[0])
    vocab=Vocab(train_tokens,min_freq=5)
    train_data=torch.tensor([truncate_pad(vocab[line],num_steps,padding_token=vocab["<pad>"])\
                             for line in train_tokens])
    test_data=torch.tensor([truncate_pad(vocab[line],num_steps,padding_token=vocab["<pad>"])\
                             for line in test_tokens])
    return DataLoader(TensorDataset(train_data,torch.tensor(train[-1],dtype=torch.long)),batch_size,shuffle=True),\
            DataLoader(TensorDataset(test_data,torch.tensor(test[-1],dtype=torch.long)),batch_size,shuffle=False),\
            vocab

class TokenEmbedding:
    def __init__(self):
        self.dir=r"../../data/glove.2024.wikigiga.100d.zip"
        self.idx_to_token,self.idx_to_vec=self.load_embedding()
        self.unk=0
        self.token_to_idx={token:idx for idx,token in enumerate(self.idx_to_token)}

    def extract_file(self,extract_path="../../data/glove"):
        os.makedirs(extract_path,exist_ok=True)
        if os.listdir(extract_path):
            return extract_path
        print("Glove Extracting...")
        with zipfile.ZipFile(self.dir,"r") as z:
            z.extractall(extract_path)
        print("Glove Succeed...")
        return extract_path

    def load_embedding(self):
        idx_to_token,idx_to_vec=["<unk>"],[]

        data_dir=self.extract_file()
        with open (glob.glob(data_dir+"/*.txt")[0],"r",encoding="utf-8") as f:
            for line in f:
                elems=line.rstrip().split(" ")
                token,elems=elems[0],[float(elem) for elem in elems[1:]]
                if len(elems)>1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        # 添加 <unk>向量，全零向量
        idx_to_vec=[[0]*len(idx_to_vec[0])]+idx_to_vec
        return idx_to_token ,torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices=[self.token_to_idx.get(token,self.unk) for token in tokens]
        # idx_to_vec已经是tensor，所以索引也需要转换成tensor
        vecs=self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)