import os
from utils import BatchLoader
from parameters import Parameters
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from RVAE import RVAE
from functional import handle_inputs, kld_coef
import numpy as np
import argparse

if __name__ == "__main__":

    if not os.path.exists('../data/word_embeddings.npy'):
        raise FileNotFoundError("word embdeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=40000, metavar='NI',
                        help='num iterations (default: 40000)')
    parser.add_argument('--batch-size', type=int, default=35, metavar='BS',
                        help='batch size (default: 35)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DR',
                        help='dropout (default: 0.2)')

    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_paramters(), args.learning_rate)

    for iteration in range(args.num_iterations):
        # TRAIN
        encoder_word_input, encoder_character_input, _, \
        decoder_input, target = batch_loader.next_batch(args.batch_size, 'train')
        [encoder_word_input, encoder_character_input, decoder_input, target] = [Variable(t.from_numpy(var)) for var in
                                                                                [encoder_word_input,
                                                                                 encoder_character_input,
                                                                                 decoder_input, target]]
        input = [encoder_word_input.long(), encoder_character_input.long(), decoder_input.long(), target.float()]
        input = [var.cuda() if args.use_cuda else var for var in input]
        [encoder_word_input, encoder_character_input, decoder_input, target] = input

        [batch_size, seq_len] = decoder_input.size()

        logits, _, kld = rvae(args.dropout, encoder_word_input, encoder_character_input, decoder_input, z=None)

        logits = logits.view(-1, parameters.word_vocab_size)
        prediction = F.softmax(logits)
        target = target.view(-1, parameters.word_vocab_size)

        # firsly NLL loss estimated, then summed over sequence to emit [batch_size] shaped BCE
        bce = (prediction.log() * target) \
            .view(batch_size, seq_len, parameters.word_vocab_size) \
            .sum(2).neg().sum(1).squeeze()

        loss = (bce + kld_coef(iteration) * kld).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print('i = {}, BCE = {}, KLD = {}, coef = {}'.format(iteration,
                                                                 bce.mean().data.cpu().numpy(),
                                                                 kld.mean().data.cpu().numpy(),
                                                                 kld_coef(iteration)))

        # VALIDATION
        if iteration % 20 == 0:
            encoder_word_input, encoder_character_input, _, \
            decoder_input, target = batch_loader.next_batch(1, 'validation')
            [encoder_word_input, encoder_character_input, decoder_input] = [Variable(t.from_numpy(var)) for var
                                                                            in
                                                                            [encoder_word_input,
                                                                             encoder_character_input,
                                                                             decoder_input]]
            input = [encoder_word_input.long(), encoder_character_input.long(), decoder_input.long()]
            input = [var.cuda() if args.use_cuda else var for var in input]
            [encoder_word_input, encoder_character_input, decoder_input] = input

            [batch_size, seq_len] = decoder_input.size()

            logits, _, _ = rvae(0., encoder_word_input, encoder_character_input, decoder_input, z=None)

            logits = logits.view(-1, parameters.word_vocab_size)
            prediction = F.softmax(logits)
            prediction = prediction.data.cpu().numpy()

            target = " ".join([batch_loader.sample_word_from_distribution(p) for p in target[0]])
            reconstruction = " ".join([batch_loader.sample_word_from_distribution(pred) for pred in prediction])

            print('\n')
            print('----------VALIDATION----------')
            print('--------reconstruction--------')
            print(reconstruction)
            print('-----------target-------------')
            print(target)
            print('------------------------------')

        # SAMPLE
        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            seed = Variable(t.from_numpy(seed).float())
            if args.use_cuda:
                seed = seed.cuda()

            decoder_input_np = batch_loader.fake_data(1)
            decoder_input = Variable(t.from_numpy(decoder_input_np).long())
            if args.use_cuda:
                decoder_input = decoder_input.cuda()

            result = ''

            for i in range(35):
                logits, _, _ = rvae(0., None, None, decoder_input, seed)

                logits = logits.view(-1, parameters.word_vocab_size)
                prediction = F.softmax(logits)

                words = [batch_loader.sample_word_from_distribution(p) for p in [prediction.data.cpu().numpy()[-1]]]

                result += ' ' + " ".join(words)

                words = [[batch_loader.word_to_idx[word] for word in words]]
                decoder_input_np = np.append(decoder_input_np, words, 1)
                decoder_input = Variable(t.from_numpy(decoder_input_np).long())
                if args.use_cuda:
                    decoder_input = decoder_input.cuda()

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(result)
            print('------------------------------')
