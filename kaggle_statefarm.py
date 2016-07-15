# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:46:58 2016

@author: yuexinmao
"""


import cv2
import numpy as np
import os
#import sys 
import csv
import glob
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split



#path = "\Users\yuexinmao\Desktop\StateFarm"
path = "/Users/yuexinmao/Desktop/StateFarm"
uniq_driver =['p002',
 'p012',
 'p014',
 'p015',
 'p016',
 'p021',
 'p022',
 'p024',
 'p026',
 'p035',
 'p039',
 'p041',
 'p042',
 'p045',
 'p047',
 'p049',
 'p050',
 'p051',
 'p052',
 'p056',
 'p061',
 'p064',
 'p066',
 'p072',
 'p075',
 'p081']
 
 
 

def load_driver(path):
    
    driver_file = os.path.join(path, 'driver_imgs_list.csv')
    
    dr = dict()  # a dictionary with key as the image names and valuse as [drive, class]
    
    #uniq_driver =[];  # record the unique drive ID 
     
    f= open(driver_file, 'r')
    reader = csv.reader(f)
    for line in reader:
         dr[line[2]] = line[0:2]  # add each line to the dic with key: image name, value: [driver, class]. e.g. 'img_21235.jpg': ['p016', 'c2']
         #if line[0]!= 'subject' and line[0] not in uniq_driver:
         #    uniq_driver.append(line[0]) 
    f.close()
    return dr

def load_train(path, img_rows, img_cols, train_size):  # train_size: number of samples selected in each class folder,  #img_rows: resize row index  #img_cols: resize column index 
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
    
#def split_validation_set(train, target, test_size):
#    random_state = 51
#    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
#    return X_train, X_test, y_train, y_test

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



def train_data_load_normalize(path,img_rows, img_cols, train_size):
    
    train_data, train_target, driver_id = load_train(path, img_rows, img_cols, train_size)
    
    # change the list to the numpy array
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    
    # change the array to the matrix 
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    # normalization 
    train_data /= 255
    
    return train_data, train_target, driver_id
    


#def test_data_load_normalize


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index




# test_cnn


# train_size: number of samples selected in each class folder,  
# img_rows: resize row index  
# img_cols: resize column index
 
img_rows = 20; img_cols = 20; train_size =10

train_data, train_target, driver_id = train_data_load_normalize(path, img_rows, img_cols, train_size)

# There are 26 unique drives, we use data of 25 drivers for training and data for the rest 1 drive for validation  

temp_index = 0  # put into a for loop if want to test all the 26 drivers
driverlist_validation = [uniq_driver[temp_index]]

driverlist_train = uniq_driver  
driverlist_train.remove(driverlist_validation[0])


X_train, Y_train, train_ind = copy_selected_drivers(train_data, train_target, driver_id, driverlist_train)
X_valid, Y_valid, test_ind = copy_selected_drivers(train_data, train_target, driver_id, driverlist_validation)


model1 = model_create_v2(img_rows, img_cols)
out2 = model1.fit(X_train, Y_train, nb_epoch = 50, batch_size = 50)


# does not work for NN (model_create_v1) for now 
#num_input = img_rows * img_cols
#model2 = model_create_v1(num_input)
#out2 = model2.fit(train_data, train_target, nb_epoch = 50, batch_size = 50)
 
 
