'''
Contains functions necessary for processing the .txt input file and loading the appropriate data
'''

import numpy as np
import h5py

def parse_input_file(fname):
    f = open(fname, "r")
    text = f.readlines()
    text = [x.strip() for x in text]
    fpath_data = text[0]
    fpath_classifier = text[1]
    class_label = int(text[2])
    try:
        save_directory = text[3]
    except:
        save_directory = False
    try:
        SFD_loss = text[4]
    except:
        SFD_loss = 1
    try:
        outfile = text[5]
    except:
        outfile = "GAN_train_results"
    return fpath_data, fpath_classifier, class_label, save_directory, SFD_loss, outfile


def choose_data_for_given_label(X, y, y_onehot, chosen_label):
    indices_toKeep = np.where(y==chosen_label)
    X = X[indices_toKeep]
    y = y[indices_toKeep]
    y_onehot = y_onehot[indices_toKeep]
    return X,y,y_onehot

def load_data_full(fpath_data):
    hf = h5py.File(fpath_data, 'r')
    X = np.array(hf.get("X"))
    y = np.array(hf.get("y"))
    y_onehot = np.array(hf.get("y_onehot"))
    hf.close()
    return X,y,y_onehot

def load_data_for_selected_class(fpath_data, class_label):
    hf = h5py.File(fpath_data, 'r')
    X = np.array(hf.get("X"))
    y = np.array(hf.get("y"))
    y_onehot = np.array(hf.get("y_onehot"))
    hf.close()
    X,y,y_onehot = choose_data_for_given_label(X,y,y_onehot,class_label)
    return X,y,y_onehot