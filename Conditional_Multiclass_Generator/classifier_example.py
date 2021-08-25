from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, LSTM, Permute, Dropout, Lambda, LeakyReLU, Activation, Input, Average
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling1D

def create_deepconv(seq_length, num_channels, num_classes):
    INPUT_SHAPE = (seq_length, num_channels)
    FINAL_SEQ_LENGTH = 15
    model = Sequential(name = "classifier")
    model.add(Conv1D(filters=128, kernel_size=(5), padding="same", input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D(pool_size=(2)))  # need to change last dimension to 10*num filters w/ this layer
    model.add(Conv1D(filters=96, use_bias=True, kernel_size=(5), padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=(5), padding="same", activation="relu"))
    model.add(Conv1D(filters=48, kernel_size=(5), padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=(2)))  # need to change last dimension to 10*num filters w/ this layer
    model.add(Conv1D(filters=32, kernel_size=(5), padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=(2)))  # need to change last dimension to 10*num filters w/ this layer
    #model.add(Reshape((FINAL_SEQ_LENGTH, num_channels * 32)))  # CHANGED FROM 21 SINCE I SWITHCED CONVOLUTIONAL SIZE
    model.add(Dropout(.5))
    model.add(LSTM(128, return_sequences=True, activation="tanh"))
    model.add(Dropout(.5))
    model.add(LSTM(128, activation="tanh"))
    model.add(Dropout(.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model