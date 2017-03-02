import argparse
import os
import numpy as np
import torch as t
from torch.optim import Adam
from rvae import RVAE
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters

if __name__ == "__main__":

    if not os.path.exists('../data/word_embeddings.npy'):
        raise FileNotFoundError("word embdeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=90000, metavar='NI',
                        help='num iterations (default: 90000)')
    parser.add_argument('--batch-size', type=int, default=22, metavar='BS',
                        help='batch size (default: 22)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=True, metavar='UT',
                        help='load pretrained model (default: False)')

    args = parser.parse_args()

    batch_loader = BatchLoader('../')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_paramters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)

    for iteration in range(args.num_iterations):

        bce, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 5 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('-------------BCE--------------')
            print(bce.mean().data.cpu().numpy()[0])
            print('-------------KLD--------------')
            print(kld.mean().data.cpu().numpy()[0])
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample)
            print('------------------------------')

    t.save(rvae.state_dict(), 'trained_RVAE')
