import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from TDNN import *
import parameters
import numpy as np


class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        word_embed = np.load('data/word_embeddings.npy')
        char_embed = np.random.uniform(-1, 1, [self.params.char_vocab_size, self.params.char_embed_size])

        self.word_embed = Variable(t.from_numpy(word_embed))
        self.char_embed = Parameter(t.from_numpy(char_embed))

        self.TDNN = TDNN(self.params)



