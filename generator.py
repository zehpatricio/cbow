from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras import utils


class WordsGenerator(Sequence):

    def __init__(
        self, words, window_size, word2num, init=0, limit=20000, batch_size=64
    ):
        self.words = words
        self.window_size = window_size
        self.word2num = word2num
        self.init = init
        self.limit = limit
        self.batch_size = batch_size
        self.maxlen = window_size*2
        self.words_total = len(words)
        self.vocab_size =len(word2num)

    def __len__(self):
        return int(np.ceil((self.limit - self.init) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx + 1) * self.batch_size
        end = end if end <= self.limit else self.limit

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)

    def prepare_data(self, start, end):
        data_x = []
        data_y = []

        for index in range(start, end):
            if index >= end:
                break
            ctx_start = index - self.window_size
            ctx_end = index + self.window_size + 1
            ctx = [
                self.word2num[self.words[i]]
                for i in range(ctx_start, ctx_end)
                if 0 <= i < self.words_total and i != index
            ]
            ctx = pad_sequences([ctx], maxlen=self.maxlen)
            tgt = self.word2num[self.words[index]]

            data_x.append(array(ctx))
            data_y.append(array([self.to_categorical(tgt)]))

        data_y = utils.to_categorical(data_y, num_classes=self.vocab_size)
        return data_x, data_y
