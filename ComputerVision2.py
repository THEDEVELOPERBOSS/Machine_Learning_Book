import tensorflow as tf
# this version of the program will keep running until a desired accuracy percentage is met
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95): # 95 is the desired percentage currently but that can be changed. 50 is just the max amount of trials it will run. 
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()



training_images=training_images/255.0
test_images=test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50,  # it trains for 50 epochs 
          callbacks=[callbacks])