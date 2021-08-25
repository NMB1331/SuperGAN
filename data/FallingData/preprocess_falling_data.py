"""
This file takes in the raw data from the Simulated Falls and Daily Living Activities Dataset, and
processes it into a numpy array and saves it to a .h5 file, to be used to train a SuperGAN to 
simulate falling data.

Author: Nathaniel M. Burley
"""

import numpy as np
import pandas as pd
import h5py
from keras.utils import to_categorical
from sklearn import preprocessing
import os

# LISTS DECLARED- ONE FOR FILES WITH A FALL, ONE FOR FILES WITH WALKING
fall_file_list = []
walk_file_list = []
walk_label_list = []
fall_label_list = []

# LOOP THROUGH SUBFOLDERS TO CREATE A LIST OF PATHS TO ACTUAL DATA FILES
for dirpath, dirnames, filenames in os.walk("RawFallingData"):
    for filename in [f for f in filenames if f.endswith(".txt")]:
        if ("801" in os.path.join(dirpath, filename)):
            walk_file_list.append(os.path.join(dirpath, filename))
            walk_label_list.append(0)
        if ("901" in os.path.join(dirpath, filename)):
            fall_file_list.append(os.path.join(dirpath, filename))
            fall_label_list.append(1)



# READ THE RELEVANT DATAFILES INTO A NUMPY ARRAY
# Read in falling data
x_fall = np.zeros(1)
for i in range(0, len(fall_file_list)):
    x = np.loadtxt(fall_file_list[0], dtype=np.float32, comments='/', skiprows=6, usecols=range(9,12))
    x = np.reshape(x, (1,x.shape[0],x.shape[1]))
    if i==0:
        x_fall = x
    else:
        x_fall = np.concatenate((x_fall,x), axis=0)
    #print(x_fall.shape)

# Read in walking data
x_walk = np.zeros(1)
for i in range(0, len(walk_file_list)):
    x = np.loadtxt(walk_file_list[0], dtype=np.float32, comments='/', skiprows=6, usecols=range(9,12))
    x = np.reshape(x, (1,x.shape[0],x.shape[1]))
    x = x[:,0:x_fall.shape[1],:]
    if i==0:
        x_walk = x
    else:
        x_walk = np.concatenate((x_walk,x), axis=0)
    #print(x_walk.shape)

# Create labels
fall_labels = np.asarray(fall_label_list[0:x_fall.shape[0]])
walk_labels = np.asarray(walk_label_list[0:x_walk.shape[0]])
#print(fall_labels.shape)



# FINALLY, WE SAVE THE DATA TO AN H5 FILE
print("Fall shape: {}".format(x_fall.shape))
print("Walk shape: {}".format(x_walk.shape))
print("Fall label shape: {}".format(fall_labels.shape))
print("Walk label shape: {}".format(walk_labels.shape))
X = np.concatenate((x_fall, x_walk))
Y = np.concatenate((fall_labels, walk_labels))
h5f = h5py.File('fall_data_segmented.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('y', data=Y)
h5f.close()

