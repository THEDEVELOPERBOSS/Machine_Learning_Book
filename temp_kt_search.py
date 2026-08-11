import tensorflow as tf
import numpy as np

# patch Keras Tuner before importing RandomSearch

def disable_tensorboard_dir(self, callbacks, trial, execution=0):
    return

try:
    from keras_tuner.src.engine.tuner import Tuner as _KTTuner
    _KTTuner._configure_tensorboard_dir = disable_tensorboard_dir
except Exception as e:
    print('patch import error', e)

from keras_tuner import RandomSearch
print('RandomSearch MRO', [(c.__module__, c.__name__) for c in RandomSearch.__mro__])
print('class patch applied', RandomSearch.__mro__[1]._configure_tensorboard_dir is disable_tensorboard_dir)

window_size = 5
x = np.arange(100, dtype=np.float32)
y = x * 0.1

# simple dataset
sequence = tf.data.Dataset.from_tensor_slices(x)
sequence = sequence.window(window_size+1, shift=1, drop_remainder=True)
sequence = sequence.flat_map(lambda window: window.batch(window_size+1))
sequence = sequence.map(lambda window: (window[:-1], window[-1]))
sequence = sequence.batch(2).prefetch(1)


def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size,)),
        tf.keras.layers.Dense(units=hp.Int('units', 4, 8, 2), activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


try:
    tuner = RandomSearch(build_model, objective='loss', max_trials=1, executions_per_trial=1, directory='tmp_my_dir', project_name='tmp')
    tuner._configure_tensorboard_dir = disable_tensorboard_dir
    tuner.search(sequence, epochs=1, verbose=0)
    print('search completed')
except Exception as e:
    import traceback; traceback.print_exc()
