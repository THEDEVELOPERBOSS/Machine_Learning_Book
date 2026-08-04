import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Ch 6 sarcasm detector pg 133 in Ch 7
# A small example corpus so the tokenizer can build a vocabulary.
sentences = [
    "I love machine learning",
    "Recurrent neural networks are useful",
    "TensorFlow helps build deep learning models",
    "This is a short example",
]
labels = np.array([1, 1, 1, 0], dtype=np.float32)

# Build tokenizer and derive the vocabulary size from it.
# This is the part that makes vocab_size and embedding_dim work.
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# The vocabulary size is based on the tokenizer's word index.
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16

# Convert the text into padded sequences so the model can process them.
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding="post", maxlen=8)

# Create a simple recurrent model using the derived values.
# This is the model architecture for the sarcasm detector example.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=8),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Loss function and classifier
# This is the compile step for the model.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train for one short epoch so the script proves the setup works.
# This is the fitting step for the example.
model.fit(padded_sequences, labels, epochs=1, verbose=0)

print("vocab_size:", vocab_size)
print("embedding_dim:", embedding_dim)
model.summary()