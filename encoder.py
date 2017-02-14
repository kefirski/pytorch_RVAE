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

        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).double(), requires_grad=False)
        self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.char_embed.weight = Parameter(t.from_numpy(char_embed).double())

        self.register_parameter('char_embed', self.char_embed.weight)

        self.TDNN = TDNN(self.params)

    def forward(self, word_input, character_input):
        """
        :param word_input: tensor with shape of [batch_size, seq_len] of Long type
        :param character_input: tensor with shape of [batch_size, seq_len, max_word_len] of Long type

        :return: mu and std of latent variable distribution both with shape of [batch_size, latent_variable_size]
        """

        [batch_size, max_seq_len] = word_input.size()

        # character input fisrsly needs to be reshaped to be 2nd rang tensor, then reshaped again
        word_input = self.word_embed(word_input)
        character_input = self.char_embed(character_input.view(-1, self.params.max_word_len))\
            .view(batch_size, max_seq_len, self.params.max_word_len, self.params.char_embed_size)

        character_input = self.TDNN(character_input)

        # for now encoder input is tensor with shape [batch_size, seq_len, word_embed_size + depth_sum]
        encoder_input = t.cat([word_input, character_input], 2)


