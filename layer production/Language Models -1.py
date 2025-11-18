import torch
from d2l import torch as d2l
import random
from text_preprocessing import read_time_machine
import matplotlib.pyplot as plt
tokens=d2l.tokenize(read_time_machine())
corpus=[token for line in tokens for token in line]
vocab=d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs=[freq for word,freq in vocab.token_freqs]
d2l.plot(freqs,xlabel="x",ylabel="freq of word",xscale="log",yscale="log")
plt.show()

# 2元语法组合
