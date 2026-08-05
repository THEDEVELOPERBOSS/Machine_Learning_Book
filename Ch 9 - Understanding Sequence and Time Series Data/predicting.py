import numpy as np
import matplotlib 
# Set backend before importing pyplot to avoid import errors in some environments
try:
    matplotlib.use("TkAgg")
except Exception:
    # fallback to a non-interactive backend if TkAgg isn't available
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Less accurate way
# Predicts series from a split time period onwards
# Period you want to split is in variable split_time
split_time = len(time) - 365
x_valid = series[split_time:]
naive_forecast = series[split_time - 1:-1]

# MSE takes differnce between predicted value and actual value at time (t), square it(to remove negatives), finds average
mse = np.mean(np.square(x_valid - naive_forecast))
mae = np.mean(np.abs(x_valid - naive_forecast))
print(mse)
# MAE calculate differnce of predicted and actual at time (t) take its absolute value(to remove negatives), finds average 
print(mae)
# More accurate way 
# Takes a group of 30 values instead of t - 1, averages out, and sets that to predicted value at time
def moving_average_forecast(series, window_size):
    # Forecasts the mean of the last few values.
    # If window_size=1, then this is equivalent to naive forecast
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

time_valid = time[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

# Smooths out trends and seasonality using differencing
# Subtracts value at t - 365 from the value at t. Results in a flatter diagram
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
# Calculates moving average of those values and add back in past values
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

# This will make it show the graph. Doesn't show this in the book
plot_series(time, series)
plt.show()
