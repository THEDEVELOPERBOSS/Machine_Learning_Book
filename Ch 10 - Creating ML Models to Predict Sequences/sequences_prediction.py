import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
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
for x, y in dataset:
    print(x.numpy(), y.numpy())
# From chapter 9 until adding in the noise
# Create a time series that has a trend, seasonality and noise
def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    # arbitrary pattern change as wanted
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
def seasonality(time, period, amplitude=1, phase=0):
    # Repeats the same pattern at each period
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype='float32')
baseline = 10
series = trend(time, .05)
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# Update with noise
series += noise(time, noise_level, seed=42)

# Turns above series into windowed dataset with a standalone function
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window:
            (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# Gets a training-ready dataset
# Splits series into training and validation sets, specify details ex: size of window, batch size, shuffle buffer size
split_time = 100
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# Since we are using tf.data.Dataset it can easily be passed to model.fit as a single parameter
# Inspects what the data looks like
dataset = windowed_dataset(series, window_size, 1, shuffle_buffer_size)
for feature, label in dataset.take(1):
    print(feature)
    print(label)
# Creating a neural network model. Because we have the data in tf.data.Dataset it is very easy w/ tf.keras
# A simple DNN(Deep Neural Network)
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([ # Very simple model w/ 2 dense layers. 1st accepts input shape of window_size before an output layer that will contain predicted value
    tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
# Model is compiled with a loss function and optimizer. Loss function specifed as mse(mean square error). Commonly used in regression problems
# Optimizer is sgd(stohastic gradient descent). Takes parameters for lr(learning rate) and momentum and these tweak how optimizer learns
# Every dataset is different so its good to have control
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
# Training becomes calling model.fit passing it your dataset and specifying the number of epochs
model.fit(dataset, epochs=100, verbose=1)
# As your data is in a list called in a series to predict value pass model values from time t to time t+window_size
# Then gives you predicted value for next time step
# Predicts value at time step 1,020 take values from steps 1,000 to 1,019(PAY ATTENTION TO HOW SERIES IS SPECIFED BELOW)
print(series[1000:1020])
# Gets value at setp 1,020
print(series[1020])
# Gets prediction for data point, pass series into model.predict. To keep input shape consistent use np.newaxis
print(model.predict(series[1000:1020][np.newaxis]))
# More generic version
start_point = 1000 
print(series[start_point:start_point+window_size])
print(series[start_point+window_size])
print(model.predict(series[start_point:start_point+window_size][np.newaxis])) 
# ^ All of this is asssuming a small window size of 20 data points 
# To change window size reformat dataset by calling windowed_dataset function and retrain model
# This shows you the overall results for the model with a loop
forecast = [] # Create a new array 
for time in range(len(series) - window_size):
    forecast.append( # Calls predict method and store results in forecast array. Not possible for first n elements of data where n is the window_size because you would not have enough data to make a prediction
        model.predict(series[time:time + window_size][np.newaxis]) 
    )

