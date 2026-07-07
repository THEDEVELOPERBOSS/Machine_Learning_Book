import urllib.request
import zipfile
import tensorflow as tf
from keras.src.layers.activations import activation
from tensorflow.keras.optimizers import RMSprop
# try to make a facial recognition version of this. Maybe a real time version
file_name = '../Training and Validation/KERAS_IMAGE_DATA_GENERATOR_STUFF/horse-or-human.zip'
training_dir = '../Training and Validation/KERAS_IMAGE_DATA_GENERATOR_STUFF/horse-or-human/training'

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='categorical'
# Binary = two classes, encoded as a single number
# Categorical = two or more classes, encoded as a vector
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu' ,
                input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=1 # change this for train epochs number
)

validation_file_name = '../Training and Validation/KERAS_IMAGE_DATA_GENERATOR_STUFF/horse-or-human.zip'
validation_dir = '../Training and Validation/KERAS_IMAGE_DATA_GENERATOR_STUFF/horse-or-human/validation'

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='categorical'
)

history = model.fit(
    train_generator,
    epochs=1, # change this for validation epochs number
    validation_data=validation_generator
)
import numpy as np
import os
from keras.preprocessing import image
# specify image folder
img_folder = r"C:\Users\Mark Dahl\Pictures\image_recognitiontest"

for fn in os.listdir(img_folder):
    path = os.path.join(img_folder, fn)
    # predicting images
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])

    predicted_class = np.argmax(classes)

    if predicted_class == 0:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
# Image augmentation
train_datagen = ImageDataGenerator( # tweak these as needed and find optimal amount
    rescale=1./255,
    rotation_range=40, # rotating each image randomly up to 40 degrees left or right
    width_shift_range=0.2, # translating the image up to 20% horizontally
    height_shift_range=0.2, # translating the image up to 20% vertically
    shear_range=0.2, # shearing the image by up to 20%
    zoom_range=0.2, # zooming the image by up to 20%
    horizontal_flip=True, # Randomly flipping the image horizontally or vertically
    fill_mode='nearest' # Filling in any missing pixels after a move or shear with nearest neighbors
    )
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

weights_file = r"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
    include_top=False,
    weights=None)
pre_trained_model.load_weights(weights_file)

pre_trained_model.summary() # to see architecture

# freezes network from retraining
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output.shape)
last_output = last_layer.output

# add dense layers
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# add a final softmax layer for classification(sigmoid if using binary. 2 would be 1 and all categorical would be binary)
x = layers.Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='cateogrical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# https://laurencemoroney.com/datasets.html