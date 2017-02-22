import torch.nn as nn


class self_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(self_Linear, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(-0.09, 0.09)
