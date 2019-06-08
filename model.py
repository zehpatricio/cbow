import load_words
import percentage
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras import utils

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
x, y = prepare_data(words[:100], 5, vocab_size)
labels = utils.to_categorical(array(y).flatten())
docs = array(x)
max_length = 4

train_labels = labels[:90]
test_labels = labels[90:]

train_docs = docs[:90]
test_docs = docs[90:]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(test_docs, test_labels, epochs=20, batch_size=128)
score = model.evaluate(test_docs, test_labels, batch_size=128)
print(">>>>>>"+score)