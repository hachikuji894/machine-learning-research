import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

import math
import numpy as np

device = 'cpu'
# device = 'cuda'

epochs = 100

sentences = [['我想要一杯啤酒P', 'S i want a beer .', 'i want a beer . E'],
             ['我想要一杯可乐P', 'S i want a coke .', 'i want a coke . E']]

src_vocab = {'P': 0, '我': 1, '想': 2, '要': 3, '一': 4, '杯': 5, '啤': 6, '酒': 7, '可': 8, '乐': 9}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 7
tgt_len = 6

d_model = 512  # embedding
d_ff = 2048  # feedforward
d_k = d_v = 64
n_layers = 6  # layers of encode and decoder
n_heads = 8  # multi-head attention
