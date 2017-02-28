import torch.nn as nn


class self_LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(self_LSTM, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)