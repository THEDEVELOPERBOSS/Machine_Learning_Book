Categorical classification (2+ classes):
- Generators must use class_mode='categorical'
- Model must end with Dense(num_classes, softmax)
- Loss must be categorical_crossentropy All three must match or training will fail with shape errors.

IMPORTANT:
Training and validation generators MUST use the same target_size.
If the model input is (150,150,3), both generators must use target_size=(150,150).
Mismatched sizes cause Dense layer shape errors during validation.

use this if modes become issue again
what mode it's running in
print("Train class mode:", train_generator.class_mode)
print("Validation class mode:", validation_generator.class_mode)
what it should be doing
print(validation_generator.class_indices)
print(validation_generator.samples)
what it is actually doing for validation
print("Validation class indices:", validation_generator.class_indices)
print("Validation samples:", validation_generator.samples)
what it is actually doing for training generator
print("Train class indices:", train_generator.class_indices)
print("Train samples:", train_generator.samples)
