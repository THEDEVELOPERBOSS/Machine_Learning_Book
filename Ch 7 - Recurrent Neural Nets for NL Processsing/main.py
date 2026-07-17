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

# Ch 6 sarcasm detector pg 137 in Ch 7
# Updated to use dropout
model = tf.keras.Sequential([
    tf.keras.Layers.Embedding(vocab_size, embedding_dim),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim,
        return_sequences=True, dropout=0.2)),
    tf.keras.Biderictional(tf.keras.layers.LSTM(embedding_dim,
        dropout=0.2)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# loss function and classifier
adam = tf.keras.optimizers.Adam(learning_rate=0.000008,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False
)
model.compile(loss='binarycrossentropy',
              optimizer=adam,
              metrics=['accuracy']
)