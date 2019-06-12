import load_words
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
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

def to_categorical(y, base_vector):
    vector = base_vector.copy()
    vector[y] = 1
    return vector

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

def prepare_data(words, window_size, word2num, init=0, limit=20000):
    maxlen = window_size*2
    words_total = len(words)
    base_vector = [0 for i in range(len(word2num))]
    
    for index in range(init, limit):
        if index >= limit:
            break
        start = index - window_size
        end = index + window_size + 1
        ctx = [word2num[words[i]] for i in range(start, end) if 0 <= i < words_total and i != index]
        ctx = pad_sequences([ctx], maxlen=maxlen)#.flatten()
        tgt = word2num[words[index]]

        x = array(ctx)
        y = array([to_categorical(tgt, base_vector)])
        yield x, y

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
    a = cbow.fit_generator(
        prepare_data(words, window_size, word2num), 
        epochs=100, steps_per_epoch=20000,callbacks=[parada]
    )
    # score = cbow.evaluate(test_docs, test_labels, batch_size=64)
    cbow.save('rede2.h5')
    # print(">>>>>>{}".format(score[1]*100))
