import tensorflow as tf # Chapter 1
data = tf.keras.datasets.fashion_mnist # this is the way to access the data. This allows you to not have to download all the images and seperate them into training and test sets and so on.

(training_images, training_labels), (test_images, test_labels) = data.load_data() # this is how to return the training sets. This is how you make the data accessible into the program I think

training_images = training_images / 255.0 # this is 2 lines are how you do an operation accross the whole array
test_images = test_images / 255.0

model = tf.keras.Sequential([ # defining the nerual network
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu), #  RELU = rectified linear unit.It doesn't matter what you put for 128 it just changes the number of 'nerouns'. More nerouns means slower run times
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', # specify the loss function and the optimizer
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50) # train the network by fitting the training images to the training labels over the 5 epochs

classifcations = model.predict(test_images)
print(classifcations[0])
print(test_labels[0])