import tensorflow as tf
import tensorflow_datasets as tfds
mnist_data = tfds.load('fashion_mnist')
for item in mnist_data:
    print(item)
# load splits into a dataset containg the actual data
mnist_train = tfds.load(name="fashion_mnist",
                        split="train"
)
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))
# shows what the data looks like
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
# Inspects a value in the dataset
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])
# Shows data about the dataset
mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
print(info)
