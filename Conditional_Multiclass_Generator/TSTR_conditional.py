"""
This script is used for creating a classifier that is trained on synthetic data, and evaluated on 
real data. This will aid in the evaluation of the synthetic data produced by the GANs.
 
Use: python train_synthetic_test_real.py /path_to_generator_folders(plural)/ path_to_real_data.h5
Author: Nathaniel M. Burley 
"""

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, Lambda, Conv1D, MaxPooling1D
from keras.optimizers import rmsprop
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import sys, os, h5py, glob, sklearn
import conditional_training_module as train


# LOAD IN GENERATORS, REAL DATA
sub = 4
generator_paths = sys.argv[1:(len(sys.argv)-2)] if len(sys.argv) > 3 else sys.argv[1]
fpath_real_data = sys.argv[len(sys.argv)-1]
print("Generator paths: {}\nReal data path: {}".format(generator_paths, fpath_real_data))


######################################## DATASETS BUILT ############################################

# VARIABLES REGARDING DATA GENERATION
num_samples = 432
num_labels = 9
latent_dim = 10
num_timesteps = 125

# GET LATEST WEIGHT FILES FROM EVERY GENERATOR FOLDER (OR THE WEIGHTS FROM A SINGLE GENERATOR)
generator_class_weights = []
if len(sys.argv) <= 3:
    generator_class_weights.append(generator_paths)
else:
    for f in generator_paths:
        f += '/'
        list_of_files = [f + i for i in os.listdir(f)]
        list_of_files.sort()
        latest_file = list_of_files[len(list_of_files)-1]
        generator_class_weights.append(latest_file)
        print("Latest file in {}: {}".format(f, latest_file))


# READ REAL DATA IN FOR EVALUATION
hf_real = h5py.File(fpath_real_data, 'r')
X_real = np.array(hf_real.get('X'))
Y_real = np.array(hf_real.get('y'))
Y_real_onehot = np.array(hf_real.get('y_onehot'))
print("X real shape: {}".format(X_real.shape))
print("Y real shape: {}".format(Y_real.shape))
print("Y real onehot shape: {}".format(Y_real_onehot))


# CREATE GENERATED (TRAINING) DATASET
# We load the models one by one, and if this is the first iteration, make the generated data equal  
# to X, labels equal to Y. Else, we join this iteration with previous ones
X_synthetic = np.empty_like(X_real)
Y_synthetic = np.empty_like(Y_real)
Y_synthetic_onehot = np.empty_like(Y_real_onehot)
if len(sys.argv) <= 3:
    generator = load_model(generator_class_weights[0])
    X_synthetic, Y_synthetic = train.generate_synthetic_data_all_classes(num_samples, generator, 0, \
    num_labels, latent_dim, num_timesteps)
else:
    for i in range(0, len(generator_class_weights)):
        generator = load_model(generator_class_weights[i])
        X_synthetic = train.generate_synthetic_data(num_samples, generator, 10, 125) if i == 0 \
            else np.concatenate((X_synthetic, train.generate_synthetic_data(num_samples, generator, 10, 125)), axis=0)
        Y_synthetic = np.zeros(num_samples) if i==0 else np.concatenate((Y_synthetic, (i * np.ones(num_samples))), axis=0)
# Make synthetic onehot y (for 9 classes)
Y_synthetic_onehot = to_categorical(Y_synthetic)
print("X synthetic shape: {}".format(X_synthetic.shape))
print("Y synthetic shape: {}".format(Y_synthetic.shape))
print("Y synthetic onehot shape: {}".format(Y_synthetic_onehot))

# SAVE THE GENERATED DATASET
h5f = h5py.File('conditionalGAN_generated_data.h5', 'w')
h5f.create_dataset('X', data=X_synthetic)
h5f.create_dataset('y', data=Y_synthetic)
h5f.create_dataset('y_onehot', data=Y_synthetic_onehot)
h5f.close()
print("Data saved successfully to conditionalGAN_generated_data.h5")


################################### CLASSIFIER ARCHITECTURE/TRAINING ###############################

# VARIABLES REGARDING DATA SHAPE, TRAINING
num_seqs = X_real.shape[0]
seq_length = X_real.shape[1]
num_channels = X_real.shape[2]
input_shape = (seq_length, num_channels)
num_classes = Y_real_onehot.shape[1]
epochs = 100

# FUNCTION THAT CREATES THE CLASSIFIER FOR THE GYROSCOPE DATA (CLASSIFIER ARCHITECTURE)
def createTSTRGyroClassifier(input_shape):
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
    classifier.add(Dense(num_classes, activation="softmax"))

    # TRAINING PARAMETERS
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    return classifier


# MAIN FUNCTION- CLASSIFIER TRAINED ON SYNTHETIC, TESTED ON REAL
if __name__ == "__main__":
    model = createTSTRGyroClassifier(input_shape)
    # for i in range(1, epochs+1):
    #     print("EPOCH {}".format(i))
    # Trains the model
    results = model.fit(X_synthetic, Y_synthetic_onehot, validation_data=(X_real, Y_real_onehot),\
        epochs=100, batch_size = 32, shuffle=True)
    # Stores evaluation metrics, to be saved for later (if you so wish, which I don't)
    y_pred_train = model.predict_classes(X_real, 100)
    f1score_train = f1_score(Y_real, y_pred_train, average="weighted")
    y_pred_test = model.predict_classes(X_real, 100)
    f1score_test = f1_score(Y_real, y_pred_test, average="weighted")
    y_pred_target = model.predict_classes(X_real, 100)
    f1score_target = f1_score(Y_real, y_pred_target, average="weighted")
    print("F1Score train: {}\nF1Score test: {}\nF1Score target: {}\n\n\n".format(f1score_train, f1score_test, f1score_target))
