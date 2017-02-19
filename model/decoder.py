import torch.nn as nn
import torch.nn.functional as F
from highway import Highway
from functional import *


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.latent_variable_size, 3, F.relu)

        self.context_to_state = nn.Linear(self.params.latent_variable_size,
                                          self.params.decoder_rnn_size * self.params.decoder_num_layers)

        self.rnn = nn.GRU(input_size=self.params.word_embed_size,
                          hidden_size=self.params.decoder_rnn_size,
                          num_layers=self.params.decoder_num_layers,
                          batch_first=True)

        self.hw2 = Highway(self.params.decoder_rnn_size, 3, F.relu)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def forward(self, decoder_input, z):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_vatiable_size]

        :return: unnormalized logits of setnence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        initial_state = self.hw1(z)
        initial_state = F.tanh(self.context_to_state(initial_state))
        initial_state = initial_state.view(batch_size, self.params.decoder_num_layers, self.params.decoder_rnn_size)

        # for now initial_state is tensor with shape of [num_layers, batch_size, decoder_rnn_size]
        initial_state = initial_state.transpose(0, 1).contiguous()

        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(self.hw2(rnn_out))
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state
