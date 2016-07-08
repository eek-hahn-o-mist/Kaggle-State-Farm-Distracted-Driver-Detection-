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

path = "C:\\Users\yumao\Desktop\StateFarm"


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





#x_train, y_train, driver_id = load_train(path, 20, 20, 10)
#x_test, y_test_id = load_test(path, 20, 20)

cache_data_train(path, img_rows=20, img_cols=20, train_size=10)
#cache_data_test (path, img_rows=20, img_cols=20)

