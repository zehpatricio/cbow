import load_words
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
import keras.backend as K
from keras import utils
import pandas as pd
import percentage
import pickle
import generator

def load_data_from_file(filename, func, *args, **kwargs):
    try:
        with open(filename, u'rb') as arq:
            data = pickle.load(arq)
    except:
        data = func(*args, **kwargs)
        with open(filename, u'wb') as arq:
            pickle.dump(data, arq)
    return data

def load_word_dicts(vocab):
    word2num = {}
    num2word = {}
    for index, word in enumerate(vocab):
        word2num[word] = index
        num2word[index] = word
    return word2num, num2word

if __name__ == '__main__':
    words = load_words.load_words()
    vocab = list(set(words))
    vocab_size = len(vocab)
    window_size = 2
    dim = 100

    word2num, num2word = load_data_from_file(
        'word_dicts.pkl', load_word_dicts, vocab
    )

    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=dim, input_shape=(window_size*2,)))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    cbow.summary()

    parada = EarlyStopping(
        monitor='acc', min_delta=0.0004, patience=3,
        verbose=1, mode='auto', restore_best_weights=True
    )
    save_network = ModelCheckpoint(
        'network.{epoch:02d}.hdf5', save_best_only=True, monitor='acc'
    )

    gen = generator.WordsGenerator(words, window_size, word2num, limit=100000)
    a = cbow.fit_generator(
        gen, epochs=10, steps_per_epoch=100000, callbacks=[parada, save_network],
        use_multiprocessing=True, workers=6
    )
    # score = cbow.evaluate(test_docs, test_labels, batch_size=64)
    cbow.save('rede2.h5')
    # print(">>>>>>{}".format(score[1]*100))
