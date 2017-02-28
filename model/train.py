import argparse
import os
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from rvae import RVAE
from utils.batch_loader import BatchLoader
from utils.functional import kld_coef
from utils.parameters import Parameters

if __name__ == "__main__":

    if not os.path.exists('../data/word_embeddings.npy'):
        raise FileNotFoundError("word embdeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=40000, metavar='NI',
                        help='num iterations (default: 40000)')
    parser.add_argument('--batch-size', type=int, default=22, metavar='BS',
                        help='batch size (default: 22)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')

    args = parser.parse_args()

    batch_loader = BatchLoader('../')
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
        input = batch_loader.next_batch(args.batch_size, 'train')

        [encoder_word_input, encoder_character_input, decoder_word_input, _,  target] = \
            [Variable(t.from_numpy(var)) for var in input]

        input = [encoder_word_input.long(), encoder_character_input.long(), decoder_word_input.long(), target.float()]
        input = [var.cuda() if args.use_cuda else var for var in input]

        [encoder_word_input, encoder_character_input, decoder_word_input, target] = input

        [batch_size, seq_len] = decoder_word_input.size()

        logits, _, kld = rvae(args.dropout,
                              encoder_word_input, encoder_character_input,
                              decoder_word_input,
                              z=None)

        logits = logits.view(-1, parameters.word_vocab_size)
        prediction = F.softmax(logits)
        target = target.view(-1, parameters.word_vocab_size)

        bce = F.binary_cross_entropy(prediction, target, size_average=False)

        loss = (bce + kld_coef(iteration) * kld)/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            print(kld_coef(iteration))
            print('------------------------------')

        # SAMPLE
        if iteration % 20 == 0:

            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            seed = Variable(t.from_numpy(seed).float())
            if args.use_cuda:
                seed = seed.cuda()

            decoder_word_input_np, _ = batch_loader.go_input(1)

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

            if args.use_cuda:
                decoder_word_input = decoder_word_input.cuda()

            result = ''

            initial_state = None

            for i in range(50):
                logits, initial_state, _ = rvae(0., None, None,
                                                decoder_word_input,
                                                seed, initial_state)

                logits = logits.view(-1, parameters.word_vocab_size)
                prediction = F.softmax(logits)

                word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

                if word == batch_loader.end_token:
                    break

                result += ' ' + word

                decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
                decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

                decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

                if args.use_cuda:
                    decoder_word_input = decoder_word_input.cuda()

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(result)
            print('------------------------------')
