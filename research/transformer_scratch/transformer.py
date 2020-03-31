import torch
from torch.autograd import Variable
import math
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np

# Class to perform look up of embedding vector for each word
# Vector will be learned as a parameter by model


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Look up embedding vector
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


# Class to add position encoding to embedding vector
# When processing word, need to know two things
# What does word mean? Where is it in the sentence?


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len],
                         requires_grad=False).cuda()
        return x
