import os
import zipfile
from keras.src.layers.activations import activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf


local_zip = '../Training and Validation/ROCK_PAPER_SCISSORS_STUFF/rps.zip'

extract_folder = r'../Training and Validation/ROCK_PAPER_SCISSORS_STUFF/rps'
os.makedirs(extract_folder, exist_ok=True)

with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
TRAINING_DIR = os.path.join(extract_folder, "train")
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=True
)
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

validation_dir = os.path.join(extract_folder, 'validation')

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    class_mode='categorical'
)
# change 150 x 150 as needed to match input data
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image:
    # 150x150 with 3 bytes of color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
# Make sure it uses a categorical lose function. Binary cross entropy will not work with more than 2 classes
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])
# Training
history = model.fit(train_generator, epochs=25,
                    validation_data = validation_generator, verbose = 1)

# Categorical classification (2+ classes):
# - Generators must use class_mode='categorical'
# - Model must end with Dense(num_classes, softmax)
# - Loss must be categorical_crossentropy
# All three must match or training will fail with shape errors.

# IMPORTANT:
# Training and validation generators MUST use the same target_size.
# If the model input is (150,150,3), both generators must use target_size=(150,150).
# Mismatched sizes cause Dense layer shape errors during validation.

# use this if modes become issue again
# what mode it's running in
# print("Train class mode:", train_generator.class_mode)
# print("Validation class mode:", validation_generator.class_mode)
# what it should be doing
# print(validation_generator.class_indices)
# print(validation_generator.samples)
# what it is actually doing for validation
# print("Validation class indices:", validation_generator.class_indices)
# print("Validation samples:", validation_generator.samples)
# what it is actually doing for training generator
# print("Train class indices:", train_generator.class_indices)
# print("Train samples:", train_generator.samples)
