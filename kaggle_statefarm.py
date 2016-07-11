# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:46:58 2016

@author: yuexinmao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:37:00 2016
@author: yumao
"""

import cv2
import numpy as np
import os
import sys 
import csv
import glob
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

#path = "\Users\yuexinmao\Desktop\StateFarm"
path = "/Users/yuexinmao/Desktop/StateFarm"

#path = ""

def load_driver(path):
    driver_file = os.path.join(path, 'driver_imgs_list.csv')
    # a dictionary with key as the image names and valuse as [drive, class]
    dr = dict()
     
    f= open(driver_file, 'r')
    reader = csv.reader(f)
    for line in reader:
         dr[line[2]] = line[0:2]  ## add each line to the dic with key: image name, value: [driver, class]. e.g. 'img_21235.jpg': ['p016', 'c2']
    f.close()
    return dr

def load_train(path, img_rows, img_cols, train_size): # train_size: number of samples selected in each class folder,  #img_rows: resize row index  #img_cols: resize column index 
    x_train = [];  y_train = [];   driver_id = []
    
    driver_data = load_driver(path)
    
    for i in range(0,10):
        path_c = os.path.join(path, 'train', 'c' + str(i), '*.jpg')
        print(path_c)
        files = glob.glob(path_c)
        #print(files)
        #print(i)
        for index, img in enumerate(files):
            
            if index >= train_size: ## loop # of train_size pics then break
                break
            
            flbase = os.path.basename(img)
            driver_id.append(driver_data[flbase][0]) ## a list that save the drive ID information
            
            img = cv2.imread(img, 0)
            img_resize = cv2.resize(img, (img_cols, img_rows))
            
            x_train.append(img_resize)
            y_train.append(i)
            
            #print x_train
    return x_train, y_train, driver_id


def load_test(path, img_rows, img_cols):
    x_test = [];  y_test_id = [];    
 
    path_c = os.path.join(path, 'test', '*.jpg')
    print(path_c)
    files = glob.glob(path_c)
    for img in files:
        flbase = os.path.basename(img)
 
        img = cv2.imread(img, 0)
        img_resize = cv2.resize(img, (img_cols, img_rows))
            
        x_test.append(img_resize)
        y_test_id.append(flbase)
    return x_test, y_test_id
    

def cache_data_train (path, img_rows, img_cols, train_size):
    
    x_train, y_train, driver_id = load_train(path, img_rows, img_cols, train_size)
    #print(x_train)
    path_t = os.path.join(path, 'Train_r' + str(img_rows) + '_c' + str(img_rows) + '_tsize' + str(train_size)+'.dat')
    file = open(path_t, 'wb')
    pickle.dump((x_train,y_train,driver_id), file)
    file.close()
    
def cache_data_test (path, img_rows, img_cols):
    
    x_test, y_test_id = load_test(path, img_rows, img_cols)
    path_t = os.path.join(path, 'Test_r' + str(img_rows) + '_c' + str(img_rows) +'.dat')
    file = open(path_t, 'wb')
    pickle.dump((x_test, x_test), file)
    file.close()

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data



def model_create_v1(num_input):
    
    model = Sequential()
    model.add(Dense(1, input_dim = num_input))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_create_v2(img_rows, img_cols):
    nb_classes = 10
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

#x_train, y_train, driver_id = load_train(path, 20, 20, 10)
#x_test, y_test_id = load_test(path, 20, 20)

#cache_data_train(path, img_rows=20, img_cols=20, train_size=10)
#cache_data_test (path, img_rows=20, img_cols=20)



## Test CNN 2D
def test_cnn2():
    img_rows = 20; img_cols = 20; train_size =10
    train_data, train_target, driver_id = load_train(path, img_rows = 20, img_cols = 20, train_size =10)
    
    # change the list to the numpy array
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    
    
    # change the array to the matrix 
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    # normalization 
    train_data /= 255
    
    
    
    ## Test CNN 2D
    model = model_create_v2(img_rows, img_cols)
    out = model.fit(train_data, train_target, nb_epoch = 50, batch_size = 50)
    return out

## Test NN
def test_nn():
    img_rows = 20; img_cols = 20; train_size =10
    train_data, train_target, driver_id = load_train(path, img_rows = 20, img_cols = 20, train_size =10)
    
    train_data = [item.flatten() for item in train_data]
    
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.astype('float32')
    # normalization 
    train_data /= 255
    
    # number of input is img_rows * img_cols
    num_input = img_rows * img_cols
    model = model_create_v1(num_input)
    out = model.fit(train_data, train_target, nb_epoch = 50, batch_size = 50)


#test_cnn2()
test_nn()

 
