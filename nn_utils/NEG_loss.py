import torch as t
from torch.nn import Parameter
import torch.nn as nn
from torch.autograd import Variable


class NEG_loss(nn.Module):

    def __init__(self, num_classes, embed_size):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. Embedding size
        """

        super(NEG_loss, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

    def forward(self, input_labes, out_labels, num_sampled):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size] of Long type
        :param num_sampled: An int. The number of sampled from noise examples

        :return: Loss estimation with shape of [batch_size]
            loss defined in Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality
            papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        """

        [batch_size] = input_labes.size()

        input = self.in_embed(input_labes)
        output = self.out_embed(out_labels)

        noise = Variable(-t.Tensor(batch_size, num_sampled, self.embed_size).uniform_(-1, 1))

        log_target = (input * output).sum(1).sigmoid().log().squeeze()  # [batch_size]

        ''' ∑[batch_size, num_sampled, embed_size] * [batch_size, embed_size] ->
            ∑[batch_size, num_sampled] -> [batch_size] '''
        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).squeeze().sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss
