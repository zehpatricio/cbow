import model
import load_words

words = load_words.load_words()
vocab = list(set(words))
vocab_size = len(vocab)
window_size = 2
sexto = int(len(words)/10)
import pdb;pdb.set_trace()
x, y = model.prepare_data(words[:sexto], window_size, vocab_size)