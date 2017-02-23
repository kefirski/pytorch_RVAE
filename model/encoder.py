import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.selfModules.highway import Highway
from utils.selfModules.selflinear import self_Linear
from utils.selfModules.tdnn import TDNN
from utils.functional import *
from utils.selfModules.selfgru import self_GRU


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.TDNN = TDNN(self.params)

        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 4, F.relu)

        self.rnn = self_GRU(input_size=self.params.word_embed_size + self.params.sum_depth,
                            hidden_size=self.params.encoder_rnn_size,
                            num_layers=self.params.encoder_num_layers,
                            batch_first=True)

        self.hw2 = Highway(self.rnn.hidden_size, 4, F.relu)
        self.fc = self_Linear(self.rnn.hidden_size, self.rnn.hidden_size)

    def forward(self, word_input, character_input):
        """
        :param word_input: tensor with shape of [batch_size, seq_len, word_embed_size]
        :param character_input: tensor with shape of [batch_size, seq_len, max_word_len, char_embed_size]

        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, seq_len, _] = word_input.size()

        assert word_input.size()[:2] == character_input.size()[:2], \
            'Word input and character input must have the same sizes, but {} and {} found'.format(
                word_input.size(), character_input.size())

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        character_input = self.TDNN(character_input)

        # for now encoder input is tensor with shape [batch_size, seq_len, word_embed_size + depth_sum]
        encoder_input = t.cat([word_input, character_input], 2)
        encoder_input = self.hw1(encoder_input.view(-1, self.params.word_embed_size + self.params.sum_depth))
        encoder_input = encoder_input.view(batch_size, seq_len, self.params.word_embed_size + self.params.sum_depth)

        # unfold rnn with zero initial state and get its final state from last layer
        _, final_state = self.rnn(encoder_input)
        final_state = final_state[-1]

        context = self.hw2(final_state)
        context = F.relu(self.fc(context))

        return context
