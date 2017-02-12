import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import parameters
import numpy as np


class TDNN(nn.Module):
    def __init__(self, params):
        super(TDNN, self).__init__()

        self.params = params

        self.kernels = [Parameter(t.rand([out_dim, self.params.char_embed_size, kW])) for kW, out_dim in params.kernels]
        self.add_to_parameters(self.kernels, 'TDNN_kernel')

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]

        :return: tensor with shape [batch_size, max_seq_len, depth_sum]

        :descr: applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert input_size_len == 4, \
            'Wrong input rang, must be equal to 4, but found {}'.format(input_size_len)

        [batch_size, max_seq_len, _, _] = input_size

        x = x.view(-1, self.params.max_word_len, self.params.char_embed_size)
        x = x.transpose(1, 2)

        xs = [F.relu(F.conv1d(x, kernel)) for kernel in self.kernels]
        xs = [t.max(x, 2)[0] for x in xs]

        x = t.cat(xs, dimension=1)

        x = x.view(batch_size, max_seq_len, -1)

        return x

    def add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)
