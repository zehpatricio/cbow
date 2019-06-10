import load_words
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.backend as K
from keras import utils
import pandas as pd
import percentage
import pickle

def prepare_data(words, window_size, vocab_size):
    maxlen = window_size*2
    words_total = len(words)
    data_x = list()
    data_y = list()
    filename = u'prepared_data.pkl'
    try:
        with open(filename, u'rb') as arq:
            data_x, data_y = pickle.load(arq)
    except:
        for index, word in enumerate(words):
            percentage.progress(index, words_total)
            start = index - window_size
            end = index + window_size + 1
            
            contexts = [words[i] for i in range(start, end) if 0 <= i < words_total and i != index]
            contexts = ' '.join(contexts)
            contexts = [one_hot(contexts, vocab_size, filters=[])]
            labels = [one_hot(word, vocab_size, filters=[])]

            x = pad_sequences(contexts, maxlen=maxlen)
            y = np_utils.to_categorical(labels, vocab_size)
            data_x.append(x)
            data_y.append(y)

        with open(filename, u'wb') as arq:
            pickle.dump((data_x, data_y), arq)

    return data_x, data_y

if __name__ == '__main__':
    words = load_words.load_words()
    vocab = list(set(words))
    vocab_size = len(vocab)
    window_size = 2
    dim = 100

    print("CARREGANDO DADOS")
    x, y = prepare_data(words, window_size, vocab_size)
    print("DADOS CARREGADOS")

    slc = int(len(x)*0.9)
    train_docs = x[:slc]
    train_labels = y[:slc]

    test_docs = x[slc:]
    test_labels = y[slc:]

    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=dim, input_length=window_size*2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='adadelta')

    a = model.fit(train_docs, train_labels, epochs=1000, batch_size=64,callbacks=[parada])
    score = model.evaluate(test_docs, test_labels, batch_size=64)
    print(">>>>>>{}".format(score[1]*100))
    model.save('rede.h5')