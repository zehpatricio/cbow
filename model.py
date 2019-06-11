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

def load_data_from_file(filename, func, *args, **kwargs):
    try:
        with open(filename, u'rb') as arq:
            data = pickle.load(arq)
    except:
        data = func(*args, **kwargs)
        with open(filename, u'wb') as arq:
            pickle.dump(data, arq)
    return data

def load_word_dicts(vocab, vocab_size):
    word2num = {}
    num2word = {}
    for word in vocab:
        num = one_hot(word, vocab_size, filters=[])
        word2num[word] = num[0]
        num2word[num[0]] = word
    return word2num, num2word

def prepare_data(words, window_size, word2num, filenumber=''):
    maxlen = window_size*2
    words_total = len(words)
    contexts = list()
    targets = list()
    
    for index, word in enumerate(words):
        percentage.progress(index, words_total)
        start = index - window_size
        end = index + window_size + 1
        context = [word2num[words[i]] for i in range(start, end) if 0 <= i < words_total and i != index]
        contexts.append(context)
        targets.append(word2num[word])
        
    data_x = [pad_sequences([ctx], maxlen=maxlen).flatten() for ctx in contexts]
    data_y = [np_utils.to_categorical(word) for word in targets]

    return data_x, data_y

def load_data(words, window_size, word2num):
    data_x = []
    data_y = []
    for num in range(1, 2):
        x, y = load_data_from_file(
            'prepared_data{}.pkl'.format(num),
            prepare_data, words, window_size, word2num, filenumber=num
        )
        data_x.extend(x)
        data_y.extend(y)
    
    return array(x), array(y)

if __name__ == '__main__':
    words = load_words.load_words()
    vocab = list(set(words))
    vocab_size = len(vocab)
    window_size = 2
    dim = 100

    word2num, num2word = load_data_from_file(
        'word_dicts.pkl', load_word_dicts, vocab, vocab_size
    )

    print("CARREGANDO DADOS")
    x, y = load_data(words, window_size, word2num)
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
    cbow.summary()
    parada = EarlyStopping(
        monitor='loss', min_delta=0.0004, patience=2, 
        verbose=1, mode='auto', restore_best_weights=True
    )
    a = cbow.fit(train_docs, train_labels, epochs=1000, batch_size=64,callbacks=[parada])
    score = cbow.evaluate(test_docs, test_labels, batch_size=64)
    print(">>>>>>{}".format(score[1]*100))
    cbow.save('rede.h5')