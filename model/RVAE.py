import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from decoder import Decoder
from encoder import Encoder
from functional import *


class RVAE(nn.Module):
    def __init__(self, params):
        super(RVAE, self).__init__()

        self.params = params

        word_embed = np.load('../data/word_embeddings.npy')
        char_embed = np.random.uniform(-1, 1, [self.params.char_vocab_size, self.params.char_embed_size])

        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).float(), requires_grad=False)
        self.char_embed.weight = Parameter(
            t.Tensor(self.params.char_vocab_size, self.params.char_embed_size).uniform_(-1, 1))

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob, encoder_word_input=None, encoder_character_input=None, decoder_input=None, z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_input: An tensor with shape of [batch_size, seq_len + 1] of Long type

        :param drop_prob: probability of an element of context to be zeroed in sence of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of setnence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input, encoder_character_input, decoder_input],
                                  True) \
               or (z is not None and decoder_input is not None), \
            "Ivalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            [batch_size, seq_len] = encoder_word_input.size()

            assert seq_len == decoder_input.size()[1] - 1, \
                """Invalid sequence lenghts. encoder.seq_len must be equal to decoder.seq_len - 1
                   got encoder.seq_len = {}, decoder.seq_len = {}
                """.format(seq_len, decoder_input.size()[1] - 1)

            encoder_word_input = self.word_embed(encoder_word_input)

            encoder_character_input = encoder_character_input.view(-1, self.params.max_word_len)
            encoder_character_input = self.char_embed(encoder_character_input)
            encoder_character_input = encoder_character_input.view(batch_size,
                                                                   seq_len,
                                                                   self.params.max_word_len,
                                                                   self.params.char_embed_size)

            context = self.encoder(encoder_word_input, encoder_character_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = -0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1).squeeze()
            train = True
        else:
            kld = None
            train = False

        z = F.dropout(z, p=drop_prob, training=train)

        decoder_input = self.word_embed(decoder_input)
        out, final_state = self.decoder(decoder_input, z)

        return out, final_state, kld

    def learnable_paramters(self):
        # word_enbedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]
