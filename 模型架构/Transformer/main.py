from Decoder import TransformerDecoder
from Encoder import TransformerEncoder
from Function.General_tools import EncoderDecoder
import torch
from Function.Data_tools import  load_data
from Function.General_tools import train,save

if __name__ == '__main__':
    # < 训练 > (主函数)
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, "cuda" if torch.cuda.is_available() else "cpu"
    ffn_num_inputs, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    samples=600
    train_iter,src_vocab, tgt_vocab = load_data(batch_size, num_steps)
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, \
                                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, \
                                 ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    train(net, train_iter, lr, num_epochs, src_vocab,tgt_vocab, device,num_steps)
    save(net,src_vocab,tgt_vocab,config= {
                                "num_hiddens": num_hiddens,
                                "num_layers": num_layers,
                                "num_heads": num_heads,
                                "dropout": dropout,
                                "max_len": num_steps,
                                "ffn_num_hiddens":ffn_num_hiddens,
                                "src_vocab_size": len(src_vocab)})

