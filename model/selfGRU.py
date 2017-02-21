import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class self_GRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        super(self_GRU, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(-0.3, 0.3)
