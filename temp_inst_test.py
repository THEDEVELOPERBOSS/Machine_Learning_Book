import tensorflow as tf
from keras_tuner import RandomSearch
from keras_tuner.src.engine.tuner import Tuner as KT

window_size=5

def build_model(hp):
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size,)),
        tf.keras.layers.Dense(hp.Int('units',4,8,2), activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

class DummyHP:
    def Int(self, name, min_value, max_value, step):
        return min_value

print('Class method before:', KT._configure_tensorboard_dir)

tuner = RandomSearch(build_model, objective='loss', max_trials=1, directory='temp_dir', project_name='temp')
print('Instance method before same as class?', tuner._configure_tensorboard_dir == KT._configure_tensorboard_dir)
print('Instance method before repr:', tuner._configure_tensorboard_dir)

patch = lambda self, callbacks, trial, execution=0: None

tuner._configure_tensorboard_dir = patch
print('Instance method after same as patch?', tuner._configure_tensorboard_dir == patch)
print('Class method unaffected?', KT._configure_tensorboard_dir == patch)
print('Instance attr exists in dict?', '_configure_tensorboard_dir' in tuner.__dict__)
print('Call instance method:', tuner._configure_tensorboard_dir)
