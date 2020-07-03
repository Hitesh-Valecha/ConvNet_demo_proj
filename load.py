import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "./PetImages/"
CATEGORIES = ["Dog","Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  #path to cats, dogs directory
    
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img_array, cmap = "gray")
        # plt.show()
        break
    break

IMG_SIZE = 150
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap = 'gray')
# plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  #path to cats, dogs directory
        class_num = CATEGORIES.index(category)  #to assign 0 & 1 to dogs and cats not necessary in that order

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:       #[:10] is printing only 10 values otherwise it'll print for every img
    print(sample[1])

x = []      #feature set
y = []      #labels

for features, labels in training_data:
    x.append(features)
    y.append(labels)

"""We made a list but we can't pass a list to a NN, sometimes we can pass labels directly as list but never features,
    so we convert x in a numpy array"""

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
"""-1 is how many features do we have, it's any number.
    we can do x 1 0, x 1 1
    or directly x -1 1
    4th parameter 1 because it's in grayscale
    for color features to feed to NN, 4th para should be 3 (RGB)"""

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# pickle_in  = open("x.pickle", "rb")
# x = pickle.load(pickle_in)
# x[1]      #x[1] is the feature & y[1] would be it's label