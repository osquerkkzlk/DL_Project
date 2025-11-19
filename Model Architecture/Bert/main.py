import torch
from torch import nn
from Tools.Data_tools import load_data_wiki
from Tools.General_Tools import train_bert
from Tools.Model import BertModel
import os



if __name__ == '__main__':
    os.makedirs("./Storage",exist_ok=True)
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    net = BertModel(len(vocab), num_hiddens=128, norm_shape=[128],
                        ffn_num_inputs=128, ffn_num_hiddens=256, num_heads=2,
                        num_layers=2, dropout=0.2, key_size=128, query_size=128,
                        value_size=128, hid_in_features=128, mlm_in_features=128,
                        nsp_in_features=128)
    device="cuda" if torch.cuda.is_available() else "cpu"
    loss = nn.CrossEntropyLoss(reduction='none')
    train_bert(train_iter, net, loss, len(vocab), device, 50, eval=False)

