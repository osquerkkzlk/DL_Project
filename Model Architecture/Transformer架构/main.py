from Decoder import TransformerDecoder
from Encoder import TransformerEncoder
from Function.General_tools import EncoderDecoder
import torch
from Function.Data_tools import  load_data
from Function.General_tools import train
import os

if __name__ == '__main__':
    # < 训练 > (主函数)
    L = []
    num_hiddens, num_layers, dropout, batch_size, num_steps = 128, 4, 0.1, 128, 20
    lr, num_epochs, device = 0.0005, 103, "cuda" if torch.cuda.is_available() else "cpu"
    ffn_num_inputs, ffn_num_hiddens, num_heads = 128, 256, 8
    key_size, query_size, value_size = 128, 128, 128
    norm_shape = [128]
    samples = None
    train_iter, src_vocab, tgt_vocab = load_data(batch_size, num_steps)
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, \
                                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, \
                                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    # 通过设置参数 Go 来选择是否导入 ./Storage文件夹下的模型、优化器、当前epoch、损失列表
    train(net, train_iter, lr, num_epochs, src_vocab, tgt_vocab, device, num_steps, Go=True)

