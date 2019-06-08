#!/usr/bin/python
# -*- coding: utf-8 -*-
import string
import pickle
import nltk
import glob
import tensorflow as tf
import percentage

import stopwords

def load_local_corpus(corpus_name):
    files = glob.glob(u'corpora/{}/*'.format(corpus_name))
    raw = []
    for filename in files:
        with open(filename) as arq:
            text = arq.read()
            punctuation = string.punctuation.replace('(', '\\(').replace(')', '\\)')
            for punt in punctuation:
                text = text.replace(punt, '')

            raw.append(text)

    raw = ' '.join(raw)
    return raw.split()

def load_words():
    filename = u'words.pkl'
    try:
        with open(filename, u'rb') as arq:
            words = pickle.load(arq)
    except:
        mac_morpho = nltk.corpus.floresta.words()
        floresta = nltk.corpus.floresta.words()
        cst_news = load_local_corpus(u'cstnews')
        opinus = load_local_corpus(u'opisumspt')

        all_words = (floresta+mac_morpho+cst_news+opinus)
        words = [w.lower() for w in all_words if w.lower() not in stopwords.STOPWORDS]
        with open(filename, u'wb') as arq:
            pickle.dump(words, arq)

    return words

def create_hot_one_vector(context_words, target_word, vocab):
    depth = len(vocab)
    indices = [vocab.index(w) for w in context_words]
    indices.append(vocab.index(target_word))
    return tf.one_hot(indices, depth)

def prepare_data(words, window, vocab):
    data = []
    total = len(words)-window
    for i in range(total):
        percentage.progress(i, total, status='Progresso')
        target = i+int(window/2)
        context_words = [words[j] for j in range(i, i+window) if j != target]
        one_hot = create_hot_one_vector(context_words, words[target], vocab)
        data.append(one_hot)
    return data

if __name__ == '__main__':
    words = load_words() # 278082
    vocab = list(set(words)) # 32522
    size = len(vocab)
    data = prepare_data(words[:1000], 5, vocab)
    # data2 = prepare_data(words[50000:100000], 5, vocab)
    # data = data + data2
    data_x = [dt[:4] for dt in data]
    data_y = [dt[4] for dt in data]

    import numpy
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, Lambda, Flatten
    from sklearn.model_selection import StratifiedKFold
    import keras.backend as K
    embed_size = 100

    cbow = Sequential()
    cbow.add(Embedding(input_dim=size, output_dim=64))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(64,)))
    cbow.add(Dense(size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # view model summary
    print(cbow.summary())

    for epoch in range(1, 6):
        loss = 0.
        i = 0
        for j in range(len(data_x)):
            i += 1

                # import pdb;pdb.set_trace()
            loss += cbow.train_on_batch(data_x[j], data_y[j])
            if i % 100000 == 0:
                print('Processed {} (context, word) pairs'.format(i))

        print('Epoch:', epoch, '\tLoss:', loss)
        print()

    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # for train, test in kfold.split(data_x, data_y):
    # X_train = data_x[train]
    # X_test = data_x[test]
    # y_train = data_y[train]
    # y_test = data_y[test]   


    # data_x = numpy.array(data_x)
    # data_y = numpy.array(data_y)

    # modelo = Sequential()
    # modelo.add(Embedding(size, 64, input_shape = (data_x.shape[1],1)))
    # modelo.add(Dense(units=64, activation='relu'))
    # modelo.add(Flatten())
    # modelo.add(Dense(units=995, activation = 'softmax'))
    # modelo.summary()
    # modelo.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    # # import pdb;pdb.set_trace()
    # historico = modelo.fit(data_x,data_y,epochs=1000, batch_size=64)
    # modelo.evaluate(X_test,y_test)
