import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist     #28 x 28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
""""normalize or scaling is useful and the values of the image pixels (0-255) will narrow to (0-1), it helps to train the model faster"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    #First layer (Input layer)
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
"""1st parameter how many uints in the layer(128), relu = rectified linear unit, it is commonly used for activation
    activation helps the neurons to fire (sort of like how neurons fire in human brain)
    after input layer, 2 hidden layers"""
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))       #softmax is used for probability distribution

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
"""loss is the degree of error, NN always try to minimize loss (not optimize the accuracy)
    adam is commonly used for optimizer, if one is familiar with gradient descent,one can use stochastic gradient descent
    metrics is a thing that u want to calculate, in this case we are measuring accuracy"""

model.fit(x_train, y_train, epochs = 3)     #epochs is how many times one wants to train

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#cmap = plt.cm.binary plots in binary otherwise plot will have some blurry colors
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save_weights('epic_num_reader.h5')   #save works for network graph, one can use save_weights(model.save_weights('epic_num_reader.model'))
# new_model = tf.keras.models.load_weights('epic_num_reader.h5')
# predictions = new_model.predict([x_test])      #x_test is in square brackets bcoz predict values always takes a list
# print(predictions)  #this will print an array of numbers, which are probability distributions
# """One can use numpy or tensor arc max tf.arcmax, but working with a tensor is difficult, instead we can use numpy"""
# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0])
# plt.show()