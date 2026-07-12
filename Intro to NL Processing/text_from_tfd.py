import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

from tree import sequence
# More stopwords would be here in a deployed program
stopwords = ['a', 'about', 'above', 'yours', 'yourself', 'yourselves', 'is']

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    # adds spaces around these characters
    sentence = sentence.replace(',', " , ")
    sentence = sentence.replace('.', " . ")
    sentence = sentence.replace('-', " - ")
    sentence = sentence.replace('/', " / ")
    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + ' '
    imdb_sentences.append(filtered_sentence)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
# The stop words get dropped
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences) # Comes out like 'Today sunny day', 'Today rainy day', etc
# This prints it in the terminal
reverse_word_index = dict(
    [(value, key) for (key, value) in tokenizer.word_index.items()]
)
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])

print(decoded_review)