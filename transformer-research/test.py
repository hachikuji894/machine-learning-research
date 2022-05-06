import torch
import torch.nn as nn
import math
import numpy as np

d_model = 512
max_len = 5000

'''
unsqueeze() 主要改变的就是 torch.Size() 在对应位置插入一个维度
squeeze() 不指定是默认压缩 torch.Size() 中为 1 的全部位置
'''


def positional_encoding_test():
    dropout = nn.Dropout(p=0.1)

    pe = torch.zeros(max_len, d_model)
    print(pe.size())
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    print(position.size())
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    print(div_term.size())
    # torch.Size([5000, 512])/2 = torch.Size([5000, 1]) * torch.Size([256])
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    print(pe.size())

    print(pe)


def index_test():
    a = np.random.rand(5)
    print(a)
    print(a[-1])  # 取最后一个元素
    print(a[:-1])  # 除了最后一个取全部
    print(a[::-1])  # 翻转读取
    print(a[2::-1])  # 翻转读取后从下标为 2 的元素截取
    print(a[0::2])  # 以 2 为步长，及下标为偶数
    print(a[1::2])  # 以 2 为步长，及下标为奇数


# positional_encoding_test()
# index_test()


src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)
print(nn.Embedding(src_vocab_size, d_model).weight)

