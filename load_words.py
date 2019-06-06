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


words = load_words() # 278082
vocab = list(set(words)) # 32522
data = prepare_data(words, 5, vocab)
import pdb;pdb.set_trace()