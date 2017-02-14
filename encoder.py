import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from TDNN import *
import parameters
from highway import *
import numpy as np


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        word_embed = np.load('data/word_embeddings.npy')
        char_embed = np.random.uniform(-1, 1, [self.params.char_vocab_size, self.params.char_embed_size])

        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).float(), requires_grad=False)
        self.char_embed.weight = Parameter(t.from_numpy(char_embed).float())

        self.TDNN = TDNN(self.params)

        self.rnn = nn.GRU(input_size=self.params.word_embed_size + self.params.sum_depth,
                          hidden_size=self.params.encoder_rnn_size,
                          num_layers=self.params.encoder_num_layers,
                          batch_first=True)

        self.highway = Highway(self.rnn.hidden_size, 4, F.relu)
        self.fc = nn.Linear(self.rnn.hidden_size, self.params.latent_variable_size)

    def forward(self, word_input, character_input):
        """
        :param word_input: tensor with shape of [batch_size, seq_len] of Long type
        :param character_input: tensor with shape of [batch_size, seq_len, max_word_len] of Long type

        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, max_seq_len, max_word_len] = character_input.size()

        assert word_input.size()[:2] == character_input.size()[:2], \
            'Word input and character input must have the same sizes, but {} and {} found'.format(
                word_input.size(), character_input.size())

        assert max_word_len == self.params.max_word_len, \
            'Wrong max_word_len, must be equal to {}, but {} found'.format(self.params.max_word_len, max_word_len)

        # character input fisrsly needs to be reshaped to be 2nd rang tensor, then reshaped again
        word_input = self.word_embed(word_input)

        character_input = character_input.view(-1, self.params.max_word_len)
        character_input = self.char_embed(character_input)
        character_input = character_input.view(batch_size,
                                               max_seq_len,
                                               self.params.max_word_len,
                                               self.params.char_embed_size)

        character_input = self.TDNN(character_input)

        # for now encoder input is tensor with shape [batch_size, seq_len, word_embed_size + depth_sum]
        encoder_input = t.cat([word_input, character_input], 2)

        # unfold rnn with zero initial state and get its final state from last layer
        zero_h = Variable(t.FloatTensor(self.rnn.num_layers, batch_size, self.params.encoder_rnn_size).zero_())
        _, final_state = self.rnn(encoder_input, zero_h)
        final_state = final_state[-1]

        context = self.highway(final_state)
        context = F.relu(self.fc(context))

        return context
