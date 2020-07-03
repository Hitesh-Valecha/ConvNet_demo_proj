import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = x/255.0

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))    #Dense works with 1D data not 2d for which we're using flatten to convert 2d to 1d

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ['accuracy'])  #rmsprop will perform gradient descent
model.fit(x, y, batch_size = 32, epochs=2)
# model.fit_generator(training_data, samples_per_epoch = 2048, nb_epoch = 30, validation_data = validation_data, nb_val_samples = 832)
# model.save_weights('models/simple_CNN.h5')