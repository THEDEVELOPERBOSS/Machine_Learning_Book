import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.distribute.multi_process_runner import multiprocessing

data = tfds.load('horses_or_humans',
                 split='train',
                 as_supervised=True) # split ex: train, validation, test. Use it to specify what part of dataset you want to use

train_batches = data.shuffle(100).batch(10)
 # Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='Adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
# Load phase start
history = model.fit(train_batches, epochs=10)

# Parallelizing ETL to Improve Training Performance

# Creates a preprocessing function that is applied to every image in the dataset
def preprocess(image, label):
    # Resize all images to the size expected by the neural network
    image = tf.image.resize(image, (300, 300))

    # Convert pixel values from integers (0-255) to floats (0.0-1.0)
    image = tf.cast(image, tf.float32) / 255.0

    # Return the processed image along with its label
    return image, label

# Creates the optimized input pipeline
train_dataset = (
    data
    # Applies the preprocessing function to every image.
    # AUTOTUNE lets TensorFlow decide how many CPU threads to use.
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Stores the processed dataset in memory after the first pass.
    # This prevents preprocessing from being repeated every epoch.
    .cache()

    # Randomly shuffles the dataset to help prevent overfitting.
    # Larger buffer sizes generally produce better randomization.
    .shuffle(1024)

    # Groups images into batches of 32 before sending them to the GPU/CPU.
    .batch(32)

    # Prepares the next batch while the current one is training.
    # This helps keep the GPU/CPU busy and improves training performance.
    .prefetch(tf.data.AUTOTUNE)
)

# Train the model like before
model.fit(train_dataset, epochs=10, verbose=1)