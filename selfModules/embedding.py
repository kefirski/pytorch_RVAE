import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Parameter

from .tdnn import TDNN


class Embedding(nn.Module):
    def __init__(self, params, path='../../../'):
        super(Embedding, self).__init__()

        self.params = params

        word_embed = np.load(path + 'data/word_embeddings.npy')

        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).float(), requires_grad=False)
        self.char_embed.weight = Parameter(
            t.Tensor(self.params.char_vocab_size, self.params.char_embed_size).uniform_(-1, 1))

        self.TDNN = TDNN(self.params)

    def forward(self, word_input, character_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :param character_input: [batch_size, seq_len, max_word_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size + sum_depth]
        """

        assert word_input.size()[:2] == character_input.size()[:2], \
            'Word input and character input must have the same sizes, but {} and {} found'.format(
                word_input.size(), character_input.size())

        [batch_size, seq_len] = word_input.size()

        word_input = self.word_embed(word_input)

        character_input = character_input.view(-1, self.params.max_word_len)
        character_input = self.char_embed(character_input)
        character_input = character_input.view(batch_size,
                                               seq_len,
                                               self.params.max_word_len,
                                               self.params.char_embed_size)

        character_input = self.TDNN(character_input)

        result = t.cat([word_input, character_input], 2)

        return result
