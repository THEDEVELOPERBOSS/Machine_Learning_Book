from keras import Sequential
from keras.src.layers import Bidirectional
from keras.src.layers.activations import activation
from keras.src.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# When loading the data instead of splitting each song line into a "sentence", we can create them on the fly from words in corpus
# here
window_size = 10
sentences = []

data = open('irish-lyrics.csv').read()
corpus = data.lower().split('\n')

# Turns the corpus into one long list of words
words = ' '.join(corpus).split()

# Creates a sliding window of words
range_size = len(words) - window_size + 1
for i in range(range_size):
    thissentence = ''
    for j in range(window_size):
        thissentence = thissentence + words[i + j]
        if j < window_size - 1:
            thissentence = thissentence + ' '
    sentences.append(thissentence)
# here

# Loads and tokenizes the data
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Turns each sliding window into a list of tokens
input_sequences = tokenizer.texts_to_sequences(sentences)

import numpy as np

# Uses prepadding to make them all the same shape
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

print(input_sequences[:5])
# Separates labels from input sequences
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
# Encode the labels
# Encodes into a set of Ys that you can then use to train
import tensorflow as tf
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
# Output layer w/ total number of words as parameter
# Each neuron in this layer will be the probability that the next word matches the word for that index value
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
model = Sequential()
model.add(Embedding(total_words, 8))
model.add(Bidirectional(LSTM(max_sequence_len-1, return_sequences='True')))
model.add(Bidirectional(LSTM(max_sequence_len-1)))
model.add(Dense(total_words, activation='softmax'))
# Compile model with categorical loss funciton
# Ex: Cateogorical cross entropy and an optimizer like Adam
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Since its a simple model with low data training for a while is possible
history = model.fit(xs, ys, epochs=1000, verbose=1)
# Generating Text
# Creates seed text: Initial Expression on which the network will base all the content it generates.
# It does this by predicting the next word
seed_text = "in the town of athy" # Starting with text its already seen
# Tokenize it using texts_to_sequences.
# This returns an array so even if only one value is present so take first value in the array
token_list = tokenizer.texts_to_sequences([seed_text])[0]
# Pad sequence to get into same shape as data used for training
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
# Predict the next word for this token list by calling.predict on token list
# Returns probabilities for each word in corpus, pass results to np.argmax to get the most likely one
predicted = np.argmax(model.predict(token_list), axis=-1)
print(predicted)
# looks up corresponding word to predicted
for word, index in tokenizer.word_index.items():
    if index == predicted:
        print(word)
        break
# Makes it so text generated from the model gets fed back into the seed text
seed_text = "sweet jeremy saw dublin"
next_words=10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
    output_word = ''

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text = seed_text + ' ' + output_word

print(seed_text)