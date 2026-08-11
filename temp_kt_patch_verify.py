import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner.src.engine.tuner import Tuner as _KTTuner

def disable(self, callbacks, trial, execution=0):
    return

_KTTuner._configure_tensorboard_dir = disable
print('Tuner class module:', _KTTuner.__module__)
print('RandomSearch MRO:', [(c.__module__, c.__name__) for c in RandomSearch.__mro__])
print('patched class method:', RandomSearch.__mro__[1]._configure_tensorboard_dir is disable)
print('patched class method repr:', RandomSearch.__mro__[1]._configure_tensorboard_dir)
print('tuner class is Tuner:', RandomSearch.__mro__[1] is _KTTuner)
