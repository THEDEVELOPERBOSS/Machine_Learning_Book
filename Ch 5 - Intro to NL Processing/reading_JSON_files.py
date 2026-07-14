import json
import string
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# More stopwords would be here in a deployed program
stopwords = ['a', 'about', 'above', 'yours', 'yourself', 'yourselves', 'is']

# Creates a table that removes punctuation from words
table = str.maketrans('', '', string.punctuation)

sentences = []
labels = []
urls = []

with open('sarcasm.json', encoding='utf-8') as f:

    # Each line is its own JSON object
    for line in f:
        item = json.loads(line)

        # Grab the headline and convert it to lowercase
        sentence = item['headline'].lower()

        # Add spaces around punctuation so it becomes easier to split into words
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace('-', ' - ')

        # Remove HTML if there is any
        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()

        words = sentence.split()

        filtered_sentence = ''

        # Remove punctuation and stopwords
        for word in words:
            word = word.translate(table)

            if word not in stopwords:
                filtered_sentence += word + ' '

        sentences.append(filtered_sentence)

        # 0 = not sarcastic, 1 = sarcastic
        labels.append(item['is_sarcastic'])

        urls.append(item['article_link'])
# Splits them into training and test sets
training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
# Tokenizes data and gets it ready for training
vocab_size = 20000
max_length = 10
trunc_type = 'post'
padding_type='post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(training_sequences, padding='post')
print(word_index)

