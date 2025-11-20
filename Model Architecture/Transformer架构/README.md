
## 概况：
我们从零实现了Transformer架构，深刻理解了其独特的注意力机制以及编码器和解码器的配合机制。同时，我们保存了模型以及词表和关键参数，方便
使用者快速部署和验证。在文件方面，我们提供了 py文件和jupyter文件，供使用者选择。

## 参数设置
    num_hiddens, num_layers, dropout, batch_size, num_steps = 128, 4, 0.1, 128, 20
    lr, num_epochs, device = 0.0005, 100, "cuda" if torch.cuda.is_available() else "cpu"
    ffn_num_inputs, ffn_num_hiddens, num_heads = 128, 256, 8
    key_size, query_size, value_size = 128, 128, 128

## 环境配置
本实验是在 py 3.10 的环境下做的,并且环境内必须安装的包如下:
    `tqdm,torch,matplotlib,numpy`

## 参考结果

![Loss_Curve.png](.\image\Loss.png)

**真实数据对**: \
        samples = \
                [('go .', 'va !'),\
                 ('hi .', 'salut !'),\
                 ('run !', 'cours !'),\
                 ('hello .', 'bonjour .'),\
                 ('i won !', "j'ai gagné !"),\
                 ("i'm ok .", 'je vais bien .'),\
                 ('thank you .', 'merci .'),\
                 ('are you ok ?', 'ça va ?'),\
                 ("i'm home .", 'je suis rentré .'),\
                 ('we won .', 'nous avons gagné .')]

**运行结果**
```python
Resumed from epoch 103 
va !                          -------->,va !                          		1.0000
salut !                       -------->,salut !                       		1.0000
nage !                        -------->,cours !                       		0.0000
bonjour !                     -------->,bonjour .                     		0.0000
j'ai gagné !                  -------->,j'ai gagné !                  		1.0000
je vais bien .                -------->,je vais bien .                		1.0000
merci .                       -------->,merci .                       		1.0000
est-ce que ça va ?            -------->,ça va ?                       		0.6514
je suis chez moi .            -------->,je suis rentré .              		0.5477
nous avons gagné .            -------->,nous avons gagné .            		1.0000

Overall_score is 0.7199
```
## 注意事项
1. 预测的字符串格式与处理文本后的格式不同，导致模型一直无法正确预测，bleu分数始终为0.，最好先将待预测字符串经过内置函数 `Data_tools.preprocess()`处理.
2. 因为数据比较小，所以模型架构如层数、头数等不能设置太大否则很容易过拟合。
3. 另一方面，数据量比较小，模型只能学到很简单的词元组合。
4. 如果epoch设置太大反而引起损失曲线的震荡，在这里，我们设置103。