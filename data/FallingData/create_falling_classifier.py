"""
Trains the classifier that will be used in the SuperGAN to generate Falling Data. Uses the architecture described in 
Synthetic Sensor Data Generation for Health Applications: A Supervised Deep Learning Approach, by Norgaard et. al
Authors: Nathaniel M. Burley, Skyler Norgaard
Dataset: UC Irvine Machine Learning Repository- 
"""

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, Lambda, Conv1D, MaxPooling1D
from keras.optimizers import rmsprop
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import sys, os, h5py

import saving_module as save
import input_module


# ########################################## HELPER FUNCTIONS/DATA PREPROCESSING #####################################


def saveClassifier(model,epoch):
    model_json = model.to_json()  # convert model to JSON object
    model_file_name = "FALL_classifier_sub" + str(epoch) + "epochs.h5"
    weights_name = "FALL_classifier_weights_" + str(epoch) + "epochs.h5"
    print("Filename to save: " + weights_name)

    with open(model_file_name, "w+") as json_file:
        json_file.write(model_json)

    model.save_weights(weights_name)

def saveClassifierSubSplit(model,epoch, sub):
    model_json = model.to_json()  # convert model to JSON object
    file_name = "FALL_hold_out" + str(sub) + "_" + str(epoch) + "epochs.json"
    weights_name = "FALL_weights_hold_out" + str(sub) + "_" + str(epoch) + "epochs.h5"
    inner_folder = "sub" + str(sub)
    fpath_model = "FALL_model_" + "sub" +str(sub) + "_" + str(epoch) + "epochs.json"
    fpath_weights = str(os.getcwd()) + "/FALL_weights_" + "sub" + str(sub) + "_" + str(epoch) + "epochs.h5"
    print("\nWeights filename: " + fpath_model)
    print("Model filename: " + fpath_weights)

    with open(fpath_model, "w+") as json_file:
        json_file.write(model_json)

    model.save_weights(fpath_weights)

def writeClassifierResultsSubSplit(epoch, f1_score_train, f1_score_test, f1score_target, sub):
    path = os.path.join(os.getcwd(), "pretrained_model_subsplit")
    filename = "f1_FALL_results_sub" + str(sub) + ".csv"

    full_fname = os.path.join(path, filename)
    to_write = str(epoch) + "," + str(f1_score_train) + "," + str(f1_score_test) + "," + str(f1score_target) + "\n"
    with open(filename, "w+") as f:
        f.write(to_write)

# LOAD DATA FROM H5 FILE
sub = 4
fpath_data = sys.argv[1]
hf = h5py.File(fpath_data, 'r')
X_source = np.array(hf.get('X'))
y_source = np.array(hf.get('y'))
print("X source shape: {}".format(X_source.shape))
print("Y source shape: {}".format(y_source.shape))

# VARIABLES REGARDING DATA SHAPE, TRAINING
num_seqs = X_source.shape[0]
seq_length = X_source.shape[1]
num_channels = X_source.shape[2]
input_shape = (seq_length, num_channels)
#num_classes = y_source.shape[1]
epochs = 30

# SPLIT INTO TRAINING AND VALIDATION SETS
X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.2, random_state=90)
print(y_train)


# ######################################### NETWORK ARCHITECTURE/TRAINING ##########################################################


# FUNCTION THAT CREATES THE CLASSIFIER FOR THE FALLING DATA (CLASSIFIER ARCHITECTURE)
def createFALLClassifier(input_shape):
    classifier = Sequential(name="classifier")
    classifier.add(Conv1D(filters=128, kernel_size=(5), padding="same", input_shape=input_shape))
    classifier.add(MaxPooling1D(pool_size=(2)))
    classifier.add(Conv1D(filters=96, use_bias="true", kernel_size=(5), padding="same", activation="relu"))
    classifier.add(Conv1D(filters=64, kernel_size=(5), padding="same", activation="relu"))
    classifier.add(Conv1D(filters=48, kernel_size=(5), padding="same", activation="relu"))
    classifier.add(MaxPooling1D(pool_size=(2)))
    classifier.add(Conv1D(filters=32, kernel_size=(5), padding="same", activation='relu'))
    classifier.add(MaxPooling1D(pool_size=(2)))
    #classifer.add(Reshape(15, num_channels * 32))
    classifier.add(Dropout(0.5))
    classifier.add(LSTM(128, return_sequences=True,activation="tanh"))
    classifier.add(Dropout(0.5))
    classifier.add(LSTM(128, activation="tanh"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation="sigmoid"))

    # TRAINING PARAMETERS
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.summary()
    return classifier


if __name__ == "__main__":
    model = createFALLClassifier(input_shape)
    accuracy_counter = 0
    for i in range(1, epochs+1):
        print("EPOCH {}".format(i))
        # Trains the model
        results = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size = 32, shuffle=True)

        # Stores evaluation metrics, to be saved for later (if you so wish, which I don't)
        y_pred_train = model.predict_classes(X_train, 100)
        f1score_train = f1_score(y_train, y_pred_train, average="weighted")
        y_pred_test = model.predict_classes(X_test, 100)
        f1score_test = f1_score(y_test, y_pred_test, average="weighted")
        print("F1Score train: {}     F1Score test: {}\n".format(f1score_train, f1score_test))
        y_pred_target = model.predict_classes(X_test, 100)
        f1score_target = f1_score(y_test, y_pred_target, average="weighted")

        # Increments counter, we want to tally how many times the accuracy is above 97%
        if results.history['acc'][0] > 0.97: accuracy_counter += 1

        # Stops training once accuracy is above 97% 3 times
        if (accuracy_counter > 2):
            print("Saving classifier....")
            model_file_name = "FALL_classifier_sub" + str(i) + "epochs."
            weights_name = "FALL_classifier_weights_" + str(i) + "epochs"
            model.save(model_file_name)
            model.save_weights(weights_name)
            print("Classifier: {}\nWeights: {}".format(model_file_name, weights_name))
            break
            #saveClassifierSubSplit(model, i, sub)  # save the weights
            #writeClassifierResultsSubSplit(i, f1score_train, f1score_test, f1score_target, sub) #save the training and test f1