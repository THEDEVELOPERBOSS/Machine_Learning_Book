import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner.src.engine.tuner import Tuner as _KTTuner

print('RandomSearch class before patch:', RandomSearch)
print('RandomSearch MRO before patch:', [(c.__module__, c.__name__) for c in RandomSearch.__mro__])
print('Original method:', RandomSearch.__mro__[1]._configure_tensorboard_dir)

def patch(self, callbacks, trial, execution=0):
    return

_KTTuner._configure_tensorboard_dir = patch
print('Patched method on class:', RandomSearch.__mro__[1]._configure_tensorboard_dir is patch)

# Instantiate tuner and patch instance explicitly as well
from keras_tuner.src.engine.tuners.randomsearch import RandomSearch as RSClass
# Actually RandomSearch imported above is same class
try:
    tuner = RandomSearch(lambda hp: None, objective='loss', max_trials=1, directory='my_dir', project_name='test')
    tuner._configure_tensorboard_dir = patch
    print('instance patched direct:', tuner._configure_tensorboard_dir is patch)
except Exception as e:
    print('instantiate error', e)
