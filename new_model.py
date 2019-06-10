import load_words
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras import utils
import pandas as pd

def prepare_data(words, window, vocab_size):
    context = []
    labels = []
    total = len(words)-window
    for i in range(total):
        # percentage.progress(i, total, status='Progresso')
        target = i+int(window/2)
        context_words = [words[j] for j in range(i, i+window) if j != target]
        context_words = ' '.join(context_words)
        
        context_words = one_hot(context_words, vocab_size, filters=[])
        target_word = one_hot(words[target], vocab_size, filters=[])
        
        context.append(context_words)
        labels.append(target_word)

    return context, labels

words = load_words.load_words()
vocab = list(set(words))
vocab_size = len(vocab)
x, y = prepare_data(words, 5, vocab_size)
docs = array(x)
max_length = 4

# sem utilidade
# lb = [i for i in range(vocab_size+1)]
# labels = utils.to_categorical(lb)

sl = int(len(x)*0.9)
train_docs = docs[:sl]
train_labels = array(y[:sl])

test_docs = docs[sl:]
test_labels = array(y[sl:])

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(max_length,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size+1, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

parada = EarlyStopping(
    monitor='loss', min_delta=0.0004, patience=2, 
    verbose=1, mode='auto', restore_best_weights=True
)
import pdb;pdb.set_trace()
a = model.fit(train_docs, train_labels, epochs=1000, batch_size=64,callbacks=[parada])
score = model.evaluate(test_docs, test_labels, batch_size=64)
print(">>>>>>{}".format(score[1]*100))
model.save('rede.h5')