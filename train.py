import os
from utils import BatchLoader

if __name__ == "__main__":

    if not os.path.exists('../data/word_embeddings.npy'):
        raise FileNotFoundError("word embdeddings file was't found")

    # batch_loader = BatchLoader()