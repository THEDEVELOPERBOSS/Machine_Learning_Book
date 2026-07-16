import json
import string

import plt
from bs4 import BeautifulSoup
from keras.src.layers.activations import activation
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

import numpy as np
import tensorflow as tf
# Converts list of training and testing data and labels to numpy format
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
# Creates them with a specified max vocabulary size and out-of-vocabulary token
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
# Initialize the embeddding layer w/ vocab size and a specified num of embedding demensions
tf.keras.layers.Embedding(vocab_size, embedding_dim), # Ex: if embedding_dim is 16, every word in vocabulary will be assigned a 16-dimensional vector
# Averages dimensions of embeddings to produce fixed-length output vector
model = tf.keras.Sequential([ # Model architecture
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model.compile(loss='binary_crossentropy',
#              optimizer='adam',metrics=['accuracy']) # Adam is the optimizer, this loads it with default parameters

# This loads it with other parameters
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, # usually 0.001. Reduced by 90%
                                beta_1=0.9, # Beta 1 and 2 needs to be between 0 and 1. Typically close to 1
                                beta_2=0.999,
                                amsgrad=False
)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy']
)
wc=tokenizer.word_counts
print(wc)
from collections import OrderedDict
# Number of times the words appear in the dataset. Ignore stopwords when doing this
OrderedDict([('former', 75)]), ('versace', 1), ('store', 35), ('clerk', 8), ('sues', 12), ('secret', 68), ('black', 203), ('code', 16)])
# Sorts into descending order of word volume because its a OrderedDict
newlist = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))
print(newlist)
OrderedDict([('new', 1143), ('trump', 966), ('man', 940), ('not', 555), ('just', 430), ('will', 427), ('one', 406), ('year', 386)])
# Plots it so you can iterate through each item in list.
# X value = ordinal of where you are (1 for first item, 2 for second item, etc)
# Y value will then be the newlist item
xs=[],
ys=[],
curr_x=1

for item in newlist:
    xs.append(curr_x),
    curr_x=(curr_x+1,
    ys.append(newlist[item]),

plt.plot(xs,ys),
plt.axis([300,10000,0,100])# Zooms in on data by changing axis fo plot before calling plt.show. Looks at volume of words 300 to 10000 on x-axis with scale from 0 to 100 on y-axis
plt.show())
# New model architecture
model = tf.keras.Sequential([
    tf.keras.layers(2000, 7),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1 , activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])
# Now that the model has been trained and optimized to minimize overfitting. Run model and inspect results
# Create a new array of sentences
sentences = ['granny starting to fear spiders in the garden might be real',
             'game of thrones season finale showing this sunday night',
             'Tensorflow book will be a bestseller'
]
# Encodes using the same tokenizer for training. Important to use the same one
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# Makes them the same length the model expects(what it was trained in)
padded = pad_sequences(sequences,
                       maxlen=max_length,
                       padding=padding_type,
                       truncating=trunc_type
)
print(padded)
# Passes them to the model and get predicitons back
print(model.predict(padded))
# Results may look like this
# [[0.7194135 ] high score indicates high level of sarcasm
# [0.02041999] these 2 low scores indicate lower chance of sarcasm
# 0.13156283]]

