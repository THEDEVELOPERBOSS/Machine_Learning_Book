import tensorflow as tf # Chapter 4
import tensorflow_datasets as tfds
# import tensorflow addons as tfa

rotation = tf.keras.layers.RandomRotation(
    factor=40 / 360,
    fill_mode="nearest"
)

data = tfds.load('horses_or_humans', split='train', as_supervised=True)

def augmentimages(image, label):
    image = tf.image.resize(image, (300, 300))
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.image.random_flip_left_right(image)
    image = rotation(image, training=True)
    return image, label

# Map to data to create a new dataset called train
train = data.map(augmentimages)
# Creates batches
train_batches = train.shuffle(100).batch(32)
# Example of splitting Dogs vs Cats dataset
train_data = tfds.load('cats_vs_dogs', split='train[:80%]',
                       as_supervised=True)

validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]' ,
                                    as_supervised=True)

test_data = tfds.load('cats_vs_dogs', split='train[-10%]',
                      as_supervised=True)
# How to check that you split it correclty. Have to iterate through whole set and count 1 by 1
print("Checking split has been done correctly:")
train_length = [i for i,_ in enumerate(train_data)][-1] + 1
print(train_length)
# ^ very slow process so only use while debugging