import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD
from utils import *
import parameters as p
from nn_utils.NEG_loss import *
import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser(description='word2vec')
parser.add_argument('--num-iterations', type=int, default=1000000, metavar='NI',
                    help='num iterations (default: 1000000)')
parser.add_argument('--batch-size', type=int, default=10, metavar='BS',
                    help='batch size (default: 10)')
parser.add_argument('--num-sample', type=int, default=5, metavar='NS',
                    help='num sample (default: 5)')
args = parser.parse_args()


batch_loader = BatchLoader()
params = p.Parameters(batch_loader.max_word_len,
                      batch_loader.max_seq_len,
                      batch_loader.words_vocab_size,
                      batch_loader.chars_vocab_size)

use_cuda = t.cuda.is_available()

neg = NEG_loss(params.word_vocab_size, params.word_embed_size)
if use_cuda:
    neg = neg.cuda()

# NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
optimizer = SGD(neg.parameters(), 0.1)

for iteration in range(args.num_iterations):

    input_idx, target_idx = batch_loader.next_embedding_seq(args.batch_size)

    input = Variable(t.from_numpy(input_idx).long())
    target = Variable(t.from_numpy(target_idx).long())
    if use_cuda:
        input, target = input.cuda(), target.cuda()

    out = neg(input, target, args.num_sample).mean()

    optimizer.zero_grad()
    out.backward()
    optimizer.step()

    out = out.data.numpy()[0]
    if iteration % 500 == 0:
        print('iteration = {}, loss = {}'.format(iteration, out))

word_embeddings = neg.input_embeddings()
np.save('data/word_embeddings.npy', word_embeddings)
