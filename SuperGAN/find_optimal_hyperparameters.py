# Only 2 lines will be added
# Rest of the flow and code remains the same as default keras
# import plaidml.keras
# plaidml.keras.install_backend()

# Imports 
import numpy as np
import pandas as pd
import tensorflow as tf
import training_module as train
import input_module as input_mod
import saving_module as save
from keras.models import Model, load_model, Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, Lambda, Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from random import shuffle
import models, sys, h5py, keras, os, statistics
import matplotlib.pyplot as plt

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Global variables/constants defined
CLASS0_DATAFILE = "/Users/nmb1331/Downloads/heartbeat/ptbdb_normal.csv"
CLASS1_DATAFILE = "/Users/nmb1331/Downloads/heartbeat/ptbdb_abnormal.csv"

# Global variables for training
NUM_CLASSES = 2
NUM_CLASSIFIER_EPOCHS = 12 
NUM_TSTR_CLASSIFIER_EPOCHS = 12 # Seems like 60 is the best
NUM_GAN_EPOCHS = 40 # 40 is optimal, but also try 60, or maybe go back to 25
CLASSIFIER_TRAIN_RATIO = 0.8
NUM_SYNTHETIC_SAMPLES = 5000 # Per class
NUM_STATS = 9 # Number of statistics computed for loss function. Do NOT change

# PARAMETERS RELATED TO TRAINING
latent_dim = 10 #length of random input fed to generator
batch_size = 30 #num instances generated for G/D training
test_size = 100 #num instances generated for validating data
real_synthetic_ratio = 5 #num synthetic instances per real instance for computing RTS metric
synthetic_synthetic_ratio = 10 #num synthetic instances to compare for computing STS metric
disc_lr = .01 #learning rate of discriminator
accuracy_threshold = 0.95 #threshold to stop generator training

# WEIGHTS FOR DIFFERENT TERMS IN THE LOSS FUNCTION
D_loss_weight = 1
C_loss_weight = 1
SFD_loss_weight = 1



# Read in the data for class 0 (normal heartbeats); make datasets for this class (used in GAN)
hr_class0_raw = pd.read_csv(CLASS0_DATAFILE)
hr_class0_raw.columns = [i for i in range(1, len(hr_class0_raw.columns)+1)]
X_0 = hr_class0_raw
hr_class0_raw = hr_class0_raw.drop(labels=len(hr_class0_raw.columns), axis=1)

# Read in the data for class 1 (abnormal heartbeat); make datasets for this class (used in GAN)
hr_class1_raw = pd.read_csv(CLASS1_DATAFILE)
hr_class1_raw.columns = [i for i in range(1, len(hr_class1_raw.columns)+1)]
X_1 = hr_class1_raw
hr_class1_raw = hr_class1_raw.drop(labels=len(hr_class1_raw.columns), axis=1)

# Read in the data for class 0 (normal heartbeat); make datasets for this class (used in GAN)
Y_0_raw = X_0[(len(X_0.columns))].values.T
Y_0 = keras.utils.np_utils.to_categorical(Y_0_raw)
X_0 = X_0.drop(labels=len(X_0.columns), axis=1)
X_0 = X_0.values
X_0 = X_0.reshape((X_0.shape[0], X_0.shape[1], 1))
print("X_0 shape: {}".format(X_0.shape))
print("Y_0 shape: {}".format(Y_0.shape))

# Read in the data for class 1 (abnormal heartbeat); make datasets for this class (used in GAN)
Y_1_raw = X_1[(len(X_1.columns))].values.T
Y_1 = keras.utils.np_utils.to_categorical(Y_1_raw)
X_1 = X_1.drop(labels=len(X_1.columns), axis=1)
X_1 = X_1.values
X_1 = X_1.reshape((X_1.shape[0], X_1.shape[1], 1))
print("X_1 shape: {}".format(X_1.shape))
print("Y_1 shape: {}".format(Y_1.shape))

# Build an X for the WHOLE dataset
X = np.concatenate((X_0, X_1))
Y = np.concatenate((Y_0_raw, Y_1_raw))
Y = keras.utils.np_utils.to_categorical(Y)
print("X shape: {}".format(X.shape))
print("Y shape: {}".format(Y.shape))

# Build training and testing sets (used in classifier)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, \
    train_size=CLASSIFIER_TRAIN_RATIO, shuffle=True)
print("Train X shape: {}".format(train_X.shape))
print("Train Y shape: {}".format(train_Y.shape))
print("Test X shape: {}".format(test_X.shape))
print("Test Y shape: {}".format(test_Y.shape))



# Parameters for training shape
num_seqs = X.shape[0]
seq_length = X.shape[1]
num_channels = X.shape[2]
input_shape = (seq_length, num_channels)

# Function that creates our classifier
def createClassifier(input_shape):
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
    classifier.add(Dense(NUM_CLASSES, activation="softmax"))

    # Training parameters
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.summary()
    return classifier

# Classifier trained here!
model = createClassifier(input_shape)
results = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),\
        epochs=NUM_CLASSIFIER_EPOCHS, batch_size = 64, shuffle=True, verbose=1)



#######################################################################################################################
########################################### HYPER PARAMETER OPTIMIZATION ##############################################

# HYPERPARAM_NAME = "DISC_LR"
# hyperparam_vals = [0.005 * i for i in range (1, 21)]
# hyperparam_tstr_acc_avg = []
# hyperparam_tstr_acc_max = []
# rts_max_vals = []
# rts_avg_vals = []
# sfd_min_vals = []
# sfd_avg_vals = []

# for val in hyperparam_vals:

#     disc_lr = val
#     print("\n\n\nTRAINING WITH {} SET TO {}\n".format(HYPERPARAM_NAME, val))
#     rts_vals = []
#     sfd_vals = []


# CREATE GENERATOR AND DISCRIMINATOR
C = model # Classifier we trained earlier (To show, #print(C.summary()) )
C.name = "C"
# Generator/discriminator for class 0
G_0 = models.create_G(seq_length, num_channels, latent_dim)
D_0 = models.create_D(seq_length, num_channels)
D_to_freeze_0 = D_0
D_model_0 = models.compile_discriminator_model(D_0, disc_lr)
# Generator/discriminator for class 0
G_1 = models.create_G(seq_length, num_channels, latent_dim)
D_1 = models.create_D(seq_length, num_channels)
D_to_freeze_1 = D_1
D_model_1 = models.compile_discriminator_model(D_1, disc_lr)

# CREATE STATISTICAL FEATURE NETWORK AND COMPUTE FEATURE VECTOR FOR REAL DATA (used in loss function)
feature_net_0 = models.create_statistical_feature_net(seq_length, num_channels)
S_X_train_0 = np.repeat(np.reshape(np.mean(feature_net_0.predict(X, batch_size), axis=0), \
                                (1, num_channels*NUM_STATS)), batch_size, axis=0)  
S_X_test_0 = np.repeat(np.reshape(np.mean(feature_net_0.predict(X, batch_size), axis=0), \
                                (1, num_channels*NUM_STATS)), test_size, axis=0)
# DO THE SAME FOR CLASS ONE
feature_net_1 = models.create_statistical_feature_net(seq_length, num_channels)
S_X_train_1 = np.repeat(np.reshape(np.mean(feature_net_1.predict(X, batch_size), axis=0), \
                                (1, num_channels*NUM_STATS)), batch_size, axis=0)  
S_X_test_1 = np.repeat(np.reshape(np.mean(feature_net_1.predict(X, batch_size), axis=0), \
                                (1, num_channels*NUM_STATS)), test_size, axis=0)

# SET CLASS LABEL TO ZERO
class_label = 0

#CREATE FULL ARCHITECTURE WHERE OUPUT OF GENERATOR IS FED TO DISCRIMINATOR AND CLASSIFIER
for layer in D_to_freeze_0.layers:
    layer.trainable = False
GCD = Model(inputs=G_0.input, outputs=[D_to_freeze_0(G_0.output), C(G_0.output), feature_net_0(G_0.output)])
GCD.compile(loss={"D":"binary_crossentropy","C":"binary_crossentropy","SFN": train.euc_dist_loss},\
            optimizer="adam", metrics={"D":"accuracy",'C':"accuracy"},\
            loss_weights = {"D": D_loss_weight, "C": C_loss_weight,"SFN": SFD_loss_weight})


GC_acc=0
epoch=1

while GC_acc<accuracy_threshold or epoch <= NUM_GAN_EPOCHS:
    print("Epoch: " + str(epoch))

    #TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
    D_loss_vec = train.train_D(batch_size, X_0, G_0, D_model_0, latent_dim)
    GCD_loss_vec = train.train_G(batch_size, X_0, class_label, S_X_train_0, NUM_CLASSES, GCD, latent_dim)
    D_acc = D_loss_vec[1] #accuracy for discriminator during its "turn" for training
    GD_acc = GCD_loss_vec[4] #accuracy for generator in tricking discriminator
    print("D Acc: " + str(D_acc))
    print("G Acc in tricking D: " + str(GD_acc))

    #GENERATE SYNTHETIC DATA AND FEED TO CLASSIFIER TO DETERMINE ACCURACY
    synthetic_data = train.generate_synthetic_data(test_size, G_0, latent_dim, seq_length)
    pred = C.predict_classes(synthetic_data,test_size,verbose=0)
    true = [class_label]*test_size
    GC_acc = accuracy_score(true, pred)
    print("C acc for synthetic data: " + str(GC_acc))

    #COMPUTE RTS AND STS METRICS
    mean_RTS_sim, mean_STS_sim, _ = train.compute_similarity_metrics(synthetic_data, X_0, test_size,\
    real_synthetic_ratio, synthetic_synthetic_ratio)
    print("RTS similarity: " + str(mean_RTS_sim))
    print("STS similarity: " + str(mean_STS_sim))

    #COMPUTE STATISTICAL FEATURE DISTANCE
    synthetic_features = feature_net_0.predict(synthetic_data, test_size, verbose=0)
    SFD = train.compute_SFD(synthetic_features, S_X_test_0)
    print("SFD: " + str(SFD) + "\n")

    epoch+=1
    one_segment_real = np.reshape(X_0[np.random.randint(0, X_0.shape[0], 1)], (seq_length, num_channels))
    # sfd_vals.append(float(SFD))
    # rts_vals.append(float(mean_RTS_sim))



# SET CLASS LABEL TO ONE
class_label = 1

#CREATE FULL ARCHITECTURE WHERE OUPUT OF GENERATOR IS FED TO DISCRIMINATOR AND CLASSIFIER
for layer in D_to_freeze_1.layers:
    layer.trainable = False
GCD = Model(inputs=G_1.input, outputs=[D_to_freeze_1(G_1.output), C(G_1.output), feature_net_1(G_1.output)])
GCD.compile(loss={"D":"binary_crossentropy","C":"binary_crossentropy","SFN": train.euc_dist_loss},\
            optimizer="adam", metrics={"D":"accuracy",'C':"accuracy"},\
            loss_weights = {"D": D_loss_weight, "C": C_loss_weight,"SFN": SFD_loss_weight})


GC_acc=0
epoch=1

while GC_acc<accuracy_threshold or epoch <= NUM_GAN_EPOCHS:
    #print("Epoch: " + str(epoch))

    #TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
    D_loss_vec = train.train_D(batch_size, X_1, G_1, D_model_1, latent_dim)
    GCD_loss_vec = train.train_G(batch_size, X_1, class_label, S_X_train_1, NUM_CLASSES, GCD, latent_dim)
    D_acc = D_loss_vec[1] #accuracy for discriminator during its "turn" for training
    GD_acc = GCD_loss_vec[4] #accuracy for generator in tricking discriminator
    print("D Acc: " + str(D_acc))
    print("G Acc in tricking D: " + str(GD_acc))

    #GENERATE SYNTHETIC DATA AND FEED TO CLASSIFIER TO DETERMINE ACCURACY
    synthetic_data = train.generate_synthetic_data(test_size, G_1, latent_dim, seq_length)
    pred = C.predict_classes(synthetic_data,test_size,verbose=0)
    true = [class_label]*test_size
    GC_acc = accuracy_score(true, pred)
    print("C acc for synthetic data: " + str(GC_acc))

    #COMPUTE RTS AND STS METRICS
    mean_RTS_sim, mean_STS_sim, _ = train.compute_similarity_metrics(synthetic_data, X_1, test_size,\
    real_synthetic_ratio, synthetic_synthetic_ratio)
    print("RTS similarity: " + str(mean_RTS_sim))
    print("STS similarity: " + str(mean_STS_sim))

    #COMPUTE STATISTICAL FEATURE DISTANCE
    synthetic_features = feature_net_1.predict(synthetic_data, test_size, verbose=0)
    SFD = train.compute_SFD(synthetic_features, S_X_test_1)
    print("SFD: " + str(SFD) + "\n")

    epoch+=1
    one_segment_real = np.reshape(X_1[np.random.randint(0, X_1.shape[0], 1)], (seq_length, num_channels))
    # sfd_vals.append(float(SFD))
    # rts_vals.append(float(mean_RTS_sim))



#######################################################################################################################
########################################### HYPER PARAMETER OPTIMIZATION ##############################################

hyperparam_vals = [1000 * i for i in range(1, 11)]
HYPERPARAM_NAME = "NUM_SYNTHETIC_SAMPLES"
hyperparam_tstr_acc_avg = []
hyperparam_tstr_acc_max = []

for val in hyperparam_vals:
    NUM_SYNTHETIC_SAMPLES= val


    # Generate data for class 0
    X_synthetic_0 = train.generate_synthetic_data(NUM_SYNTHETIC_SAMPLES, G_0, latent_dim, seq_length)
    Y_synthetic_0 = np.zeros(NUM_SYNTHETIC_SAMPLES)
        
    # Generate data for class 1
    X_synthetic_1 = train.generate_synthetic_data(NUM_SYNTHETIC_SAMPLES, G_1, latent_dim, seq_length)
    Y_synthetic_1 = np.ones(NUM_SYNTHETIC_SAMPLES)

    # Build the synthetic dataset (combine generated data from both classes)
    X_synthetic = np.concatenate((X_synthetic_0, X_synthetic_1), axis=0)
    print(X_synthetic.shape)
    Y_synthetic = keras.utils.np_utils.to_categorical\
        (np.concatenate((Y_synthetic_0, Y_synthetic_1), axis=0))
    print(Y_synthetic.shape)

    # Classifier trained here!
    tstr_model = createClassifier(input_shape)
    tstr_results = tstr_model.fit(X_synthetic, Y_synthetic, validation_data=(test_X, test_Y),\
            epochs=NUM_TSTR_CLASSIFIER_EPOCHS, batch_size = 64, shuffle=True)

    # Store the results
    try:
        hyperparam_tstr_acc_max.append(max(tstr_results.history['val_acc']))
    except:
        hyperparam_tstr_acc_max.append(None)
    try:
        hyperparam_tstr_acc_avg.append(statistics.mean(tstr_results.history['val_acc']))
    except:
        hyperparam_tstr_acc_avg.append(None)

    # try:
    #     rts_avg_vals.append(statistics.mean(rts_vals))
    # except:
    #     rts_avg_vals.append(None)

    # try:
    #     rts_max_vals.append(max(rts_vals))
    # except:
    #     rts_max_vals.append(None)

    # try:
    #     sfd_avg_vals.append(statistics.mean(sfd_vals))
    # except:
    #     sfd_avg_vals.append(None)
    # try:
    #     sfd_min_vals.append(min(sfd_vals))
    # except:
    #     sfd_min_vals.append(None)
    


# Plot the parameter's maximum TSTR value
plt.scatter(hyperparam_vals, hyperparam_tstr_acc_max, label="Max TSTR Accuracy")
plt.scatter(hyperparam_vals, hyperparam_tstr_acc_avg, label="Avg. TSTR Accuracy")
plt.plot(hyperparam_vals, hyperparam_tstr_acc_max)
plt.plot(hyperparam_vals, hyperparam_tstr_acc_avg)
plt.xlabel('{} Value'.format(HYPERPARAM_NAME))
plt.ylabel('TSTR ACCURACY')
plt.title('MAX, AVG TSTR ACCURACY FOR VARYING VALUES OF {}'.format(HYPERPARAM_NAME))
plt.gca().legend(loc="upper right")
tstr_filename = "/Users/nmb1331/gans_deep_learning/IMWUT_GAN/code/NateBurley_Research_GANs/SuperGAN/images/TSTR_STATS-"\
     + HYPERPARAM_NAME + "-{}_GAN_EPOCHS.png".format(str(NUM_GAN_EPOCHS))
plt.savefig(tstr_filename)
plt.close()

# Plot the mean and min SFD value
# plt.scatter(hyperparam_vals, sfd_avg_vals, label="SFD Average")
# plt.scatter(hyperparam_vals, sfd_min_vals, label="SFD Minimum")
# plt.plot(hyperparam_vals, sfd_avg_vals)
# plt.plot(hyperparam_vals, sfd_min_vals)
# plt.xlabel('{} Value'.format(HYPERPARAM_NAME))
# plt.ylabel('SFD Value')
# plt.title('SFD VALUES FOR VARYING VALUES OF {}'.format(HYPERPARAM_NAME))
# plt.gca().legend(loc="upper right")
# tstr_filename = "/Users/nmb1331/gans_deep_learning/IMWUT_GAN/code/NateBurley_Research_GANs/SuperGAN/images/SFD_STATS-"\
#      + HYPERPARAM_NAME + "-{}_GAN_EPOCHS.png".format(str(NUM_GAN_EPOCHS))
# plt.savefig(tstr_filename)
# plt.close()

# # Plot the mean and max RTS value
# plt.scatter(hyperparam_vals, rts_avg_vals, label="RTS Average")
# plt.scatter(hyperparam_vals, rts_max_vals, label="RTS Maximum")
# plt.plot(hyperparam_vals, rts_avg_vals)
# plt.plot(hyperparam_vals, rts_max_vals)
# plt.xlabel('{} Value'.format(HYPERPARAM_NAME))
# plt.ylabel('RTS Value')
# plt.title('RTS VALUES FOR VARYING VALUES OF {}'.format(HYPERPARAM_NAME))
# plt.gca().legend(loc="upper right")
# tstr_filename = "/Users/nmb1331/gans_deep_learning/IMWUT_GAN/code/NateBurley_Research_GANs/SuperGAN/images/RTS_STATS-"\
#      + HYPERPARAM_NAME + "-{}_GAN_EPOCHS.png".format(str(NUM_GAN_EPOCHS))
# plt.savefig(tstr_filename)
# plt.close()