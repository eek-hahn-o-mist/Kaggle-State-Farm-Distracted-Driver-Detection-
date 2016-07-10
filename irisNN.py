import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

# import csv
csv = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
iris = np.genfromtxt(csv, delimiter = ',', dtype = None)

# Munge data and create binary classes
iris = np.delete(iris, 0, 0)  # delete header row
iris[iris[:,4] == 'setosa', 4] = 1  # change setosa to class 1
iris[(iris[:,4] == 'versicolor') | (iris[:,4] == 'virginica'), 4] = 0  # others to class 0

# split into data and label classes
data = iris[:,0:4]
data = data.astype(float)

labels = iris[:,4]
labels = np.array([labels])
labels = labels.T  # must be transposed to match Kera's format
labels = labels.astype(int)

# develop NN model
model = Sequential()
model.add(Dense(1, input_dim = 4))
model.add(Activation('sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fit NN
model.fit(data, labels, nb_epoch = 50, batch_size = 50)