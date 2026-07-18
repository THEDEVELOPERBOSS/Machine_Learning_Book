import tensorflow as tf

# Ch 6 sarcasm detector pg 133 in Ch 7
model = tf.keras.Sequential([
    tf.keras.Layers.Embedding(vocab_size, embedding_dim),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim)), # Updated it to use a bidirectional LSTM(Long short term memory)
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# loss function and classifier
adam = tf.keras.optimizers.Adam(learning_rate=0.00001,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False
)
model.compile(loss='binarycrossentropy',
              optimizer=adam,
              metrics=['accuracy']
)

# Ch 6 sarcasm detector pg 135 in Ch 7
# Updated to have stacked LSTMS
model = tf.keras.Sequential([
    tf.keras.Layers.Embedding(vocab_size, embedding_dim),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim,
        return_sequences=True)),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# loss function and classifier
# Reduce learning rate to reduce overfitting
adam = tf.keras.optimizers.Adam(learning_rate=0.000008,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False
)
model.compile(loss='binarycrossentropy',
              optimizer=adam,
              metrics=['accuracy']
)

# Creates a dictionary for the GloVe when in google colab
glove_embeddings = dict()
f = open('tmp/glove/glove.twitter.27B.25d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()
# Searches for all 'frog' instances in model
glove_embeddings['frog']
# Creates new matrix using tokenizer to get word index
# Uses GloVe set embeddings
{'<OOV>': 1, 'new': 2, ... 'not': 5, 'just': 6, 'will': 7}
# Creates above matrix
embeddings_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
# Amends embedding layer to use pretrained embeddings by setting weights parameter
# and specify that it shouldn't the layer to be trained by trainable=False
# Ch 6 sarcasm detector pg 135 in Ch 7
model = tf.keras.Sequential([
    tf.keras.Layers.Embedding(vocab_size, embedding_dim,
                            weights=[embedding_matrix], trainable=False)
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim,
        return_sequences=True)),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# loss function and classifier
# Reduce learning rate to reduce overfitting
adam = tf.keras.optimizers.Adam(learning_rate=0.000008,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False
)
model.compile(loss='binarycrossentropy',
              optimizer=adam,
              metrics=['accuracy']
)
# Shows how large your vocab should be
# Sorts data out by creating list of Xs and Ys
# X = word index, Y = 1 if the word is in embeddings, 0 if it isn't
xs=[]
ys=[]
cumulative_x=[]
cumulative_y=[]
total_y=0
for word, index in tokenizer.word_index.items():
    xs.append(index)
    cumulative_x.append(index)
    if glove_embeddings.get(word) is not None:
        total_y = total_y + 1
        ys.append(1)
    else:
        ys.append(0)
    cumulative_y.append(total_y / index)
# Plots Xs against Ys
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,2))
ax.spines['top'].set_visible(False)

plt.margins(x=0, y=None, tight=True)
#plt.axis([1300, 14000, 0, 1])
plt.fill(ys)
# Plots cumulative_x vs cumulative_y
import matplotlib as plt
plt.plot(cumulative_x, cumulative_y)
plt.axis([0, 25000, .915, .985])