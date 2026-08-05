import tensorflow as tf
# tf.data conatains APIs useful for manipulating data
# Use those to create a basic dataset containing 0-9 emulating a time series
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True) # shift = 1 causes each window be shifted one spot from previous one. 1st window has the 5 items at 0, next window 5 items beginning at 1, etc
# drop remainder equalling True makes it so once it gets to point close to end of dataset where the windows would be smaller than desired size of 5 they will be dropped
# Splits data into windows of 5 items
dataset = dataset.flat_map(lambda window: window.batch(5)) # Beginnings of windowed dataset. 
dataset = dataset.map(lambda window: (window[:-1], window[-1:])) # Makes it so there are n values defining a feature and subsequent value giving a label
# ^ Adds a lambda function that splits each window into everything before the last value, and then the last value. Results in an x and a y dataset
dataset = dataset.shuffle(buffer_size=10) # Shuffles it and with a batch size of 2 
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print("x = ", x.numpy(), "y = ", y.numpy())
for window in dataset:
    print(window.numpy())