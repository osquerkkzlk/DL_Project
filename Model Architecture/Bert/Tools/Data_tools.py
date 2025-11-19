import torch
import os,random
from .Model import get_tokens_and_segments
from collections import Counter

def tokenize(lines, token='word'):
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


def read_wiki():
    data_dir=r"E:\DL_files\NLP\wikitext-2"
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r',encoding="utf-8") as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def replace_mlm_tokens(tokens,candidate_pred_position,num_mlm_preds,vocab):
    # 防止更改原 tokens
    mlm_input_tokens=[token for token in tokens]
    temp=[]
    random.shuffle(candidate_pred_position)
    for mlm_pred_position in candidate_pred_position:
        if len(temp)>=num_mlm_preds:
            break
        masked_token=None
        if random.random()<0.8:
            masked_token="<mask>"
        else:
            if random.random()<0.5:
                masked_token=tokens[mlm_pred_position]
            else:
                masked_token=random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position]=masked_token
        # （真实位置，真实词元）
        temp.append((mlm_pred_position,tokens[mlm_pred_position]))
    # return: tokens,[(真实位置，真实词元),...]
    return mlm_input_tokens,temp


def get_mlm_data(tokens,vocab):
    # 从整个token中筛选用于遮蔽的数据
    candidate_pred_positions=[]
    for i,token in enumerate(tokens):
        if token in ["<cls>","<sep>"]:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds=max(1,round(len(tokens)*0.15))
    mlm_input_tokens,temp=replace_mlm_tokens(tokens,candidate_pred_positions,num_mlm_preds,vocab)
    temp=sorted(temp,key=lambda x:x[0])

    pred_position=[k[0]for k in  temp]
    mlm_pred_labels=[k[1]for k in temp]
    # return :遮蔽处理后的数据，遮蔽的位置，真实数据
    return vocab[mlm_input_tokens],pred_position,vocab[mlm_pred_labels]


def pad_bert_inputs(examples,max_len,vocab):
    max_num_mlm_preds=round(max_len*0.15)
    all_token_ids,all_segments,valid_lens=[],[],[]
    all_pred_positions,all_mlm_weights,all_mlm_labels=[],[],[]
    nsp_labels=[]
    for (token_ids,pred_positions,mlm_pred_label_ids,segments,is_next) in examples:
        # 遮蔽后的词元（训练数据）
        all_token_ids.append(torch.tensor(token_ids+[vocab["<pad>"]]*(max_len-len(token_ids)),dtype=torch.long))
        # 用于预测段落任务的数据片段
        all_segments.append(torch.tensor(segments+[0]*(max_len-len(segments)),dtype=torch.long))
        # 真实数据的有效长度
        valid_lens.append(torch.tensor(len(token_ids),dtype=torch.float32))
        # 遮蔽的位置（需要预测），填充长度至max_len，保证数据对齐
        all_pred_positions.append(torch.tensor(pred_positions+[0]*(max_num_mlm_preds-len(pred_positions)),dtype=torch.long))
        # ” 用于屏蔽填充位置的损失权重 ,“ 1 为真实 mask 位置，0 为填充位置）
        all_mlm_weights.append(torch.tensor([1.]*len(mlm_pred_label_ids)+[0.]*(max_num_mlm_preds-len(pred_positions)),dtype=torch.float32))
        # 预测填充词元的真实标签（即真实数据）
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids+[0]*(max_num_mlm_preds-len(mlm_pred_label_ids)),dtype=torch.float32))
        # 预测段落任务的标签
        nsp_labels.append(torch.tensor(is_next,dtype=torch.long))

    return (all_token_ids,all_segments,valid_lens,all_pred_positions,\
            all_mlm_weights,all_mlm_labels,nsp_labels)




class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self,paragraphs,max_len):
        paragraphs=[tokenize(paragraph,token="word") for paragraph in paragraphs]
        sentences=[sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab=Vocab(sentences,min_freq=5,reserved_tokens=["<pad>","<mask>","<sep>","<cls>"])

        # 获取下一个句子预测任务的数据
        examples=[]
        for paragraph in paragraphs:
            examples.extend(get_nsp_data_from_paragraph(paragraph,paragraphs,self.vocab,max_len))

        # 获取遮蔽语言模型任务的数据
        examples=[(get_mlm_data(tokens,self.vocab)+(segments,is_next))\
                  for tokens,segments ,is_next in examples]
        (self.all_token_ids,self.all_segments,self.valid_lens,\
         self.all_pred_positions,self.all_mlm_weights,self.all_mlm_labels,\
         self.nsp_labels)=pad_bert_inputs(examples,max_len,self.vocab)

    def __getitem__(self, idx):

        return (self.all_token_ids[idx],self.all_segments[idx],self.valid_lens[idx],\
                self.all_pred_positions[idx],self.all_mlm_weights[idx],self.all_mlm_labels[idx],\
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size,max_len):
    """加载数据集"""
    #< paragraphs shape > :["Adsji(段落)","dsioj(段落)",...]
    paragraphs=read_wiki()
    train_set=WikiTextDataset(paragraphs,max_len)
    train_iter=torch.utils.data.DataLoader(train_set,batch_size,shuffle=True)
    return train_iter,train_set.vocab

