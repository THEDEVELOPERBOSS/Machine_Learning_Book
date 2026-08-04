import os
import zipfile
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

script_dir = Path(__file__).resolve().parent

# This is the dataset zip file I downloaded and extracted for Rock Paper Scissors
local_zip = (
    script_dir.parent.parent
    / "Training and Validation"
    / "ROCK_PAPER_SCISSORS_STUFF"
    / "archive.zip"
)

extract_folder = (
    script_dir.parent.parent
    / "Training and Validation"
    / "ROCK_PAPER_SCISSORS_STUFF"
)

os.makedirs(extract_folder, exist_ok=True)

# Extract the dataset if it has not already been extracted
dataset_dir = extract_folder / "Rock-Paper-Scissors"

if not dataset_dir.exists():
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

# These are the folders inside the extracted Rock-Paper-Scissors dataset
TRAINING_DIR = dataset_dir / "train"
validation_dir = dataset_dir / "validation"

training_datagen = ImageDataGenerator(
    rescale=1./255,
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

    tf.keras.layers.Dense(3, activation='softmax'),

    # This should help fight overfitting in the model
    tf.keras.layers.Dropout(0.2),
])

# Make sure it uses a categorical loss function.
# Binary cross entropy will not work with more than 2 classes
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    verbose=1
)