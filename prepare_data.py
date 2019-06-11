import model
import load_words

words = load_words.load_words()
vocab = list(set(words))
vocab_size = len(vocab)
window_size = 2
sexto = int(len(words)/10)
word2num, num2word = model.load_data_from_file(
    'word_dicts.pkl', model.load_word_dicts, vocab, vocab_size
)
num_words = len(words)
# num = 1
for num in range(5,6):
    start = sexto*(num-1)
    end = sexto*num
    x, y = model.load_data_from_file(
        'prepared_data{}.pkl'.format(num), 
        model.prepare_data, 
        words[start: (end if end < num_words else num_words)],
        window_size, word2num, filenumber=num
    )