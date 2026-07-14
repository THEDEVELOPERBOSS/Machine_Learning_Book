from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [ # Think of this like training data
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?' # all punctuation gets ignored except for '
    'I really enjoyed walking in the snow today'
]

tokenizer = Tokenizer(num_words= 100) # max number of words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(word_index)
# How to deal with out-of-vocabulary (OOV) token
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)

tokenizer = Tokenizer(num_words= 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
# Converts unpadded sequences into a padded set using the API.
# These get more complex as you go only one can be used at once.
# padded = pad_sequences(sequences) # pads at the beginning aka prepadding
# padded = pad_sequences(sequences, padding='post') # pads at the end
padded = pad_sequences(sequences, padding='post', maxlen=6) # Adds a max length of 6
padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post') # Cuts things off at the end
print(padded)

