class Parameters:
    def __init__(self, max_word_len, max_seq_len, words_vocab_size, chars_vocab_size):

        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len + 1 # go or eos token
        self.words_vocab_size = words_vocab_size
        self.chars_vocab_size = chars_vocab_size

        self.word_embedding_size = 300
        self.character_embedding_size = 15

        self.encoder_rnn_size = [800, 950, 1100]
        self.encoder_num_rnn_layers = len(self.encoder_rnn_size)

        self.latent_variable_size = 1200

        self.decoder_rnn_size = [1200, 1350, 1400]
        self.decoder_num_rnn_layers = len(self.decoder_rnn_size)