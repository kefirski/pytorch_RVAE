import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD
from utils import *
import parameters as p
from NEG_loss import *
import numpy as np
import sys

assert len(sys.argv) == 4, \
    "Invalid parameters. len(arg) should be 1, got {}".format(len(sys.argv))

[num_iterations, batch_size, num_sample, use_cuda] = sys.argv

use_cuda = True if use_cuda == 1 else False

batch_loader = BatchLoader()
params = p.Parameters(batch_loader.max_word_len,
                      batch_loader.max_seq_len,
                      batch_loader.words_vocab_size,
                      batch_loader.chars_vocab_size)

neg = NEG_loss(params.word_vocab_size, params.word_embed_size)
if use_cuda:
    neg = neg.cuda()

# NEG_loss is defined over weight two embed matrixes with shape of [params.word_vocab_size, params.word_embed_size]
optimizer = SGD(neg.parameters(), 0.1)


for iteration in range(num_iterations):

    input_idx, target_idx = batch_loader.next_embedding_seq(batch_size)

    input = Variable(t.from_numpy(input_idx).long())
    target = Variable(t.from_numpy(target_idx).long())
    if use_cuda:
        input, target = input.cuda(), target.cuda()

    out = neg(input, target, num_sample).mean()

    optimizer.zero_grad()
    out.backward()
    optimizer.step()

    if iteration % 500 == 0:
        print('iteration = {}, loss = {}'.format(iteration, out))

word_embeddings = neg.in_embed.weight.numpy()
np.save('data/word_embeddings.npy', word_embeddings)

