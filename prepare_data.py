import model
import load_words

words = load_words.load_words()
vocab = list(set(words))
vocab_size = len(vocab)
window_size = 2
sexto = int(len(words)/10)
num = 1
x, y = model.prepare_data(
    words[sexto*(num-1):], window_size, vocab_size, filenumber=num
)