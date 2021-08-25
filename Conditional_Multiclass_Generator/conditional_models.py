'''
This file contains the code for creating the auxiliary classifier recurrent architecture. 
In this case, labels are fed to the network as a one-hot vector.
'''

import numpy as np
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, LSTM, Permute, Dropout, Lambda, LeakyReLU, Activation, Input
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop
import os
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

'''
Issues that need to be figured out in both cases!!!!!

'''


'''
Below are the models in which the input and labels are concatenated and fed in as one vector
'''

def create_cond_D(seq_length, num_channels, num_labels):
    D_input_shape = (seq_length, num_channels + num_labels)
    D_in = Input(shape=D_input_shape)
    D = Dropout(.5)(D_in)
    D = LSTM(100, activation="tanh")(D)
    D = Dense(1, activation="sigmoid")(D)
    D = Model(inputs=D_in, outputs=D, name="D")
    return D

def create_cond_G(seq_length, num_channels, num_labels, latent_dim):
    G_input_shape = (seq_length, latent_dim + num_labels)

    #PORTION OF MODEL THAT ACTUAL GENERATES SYNTHETIC DATA
    G_in = Input(shape=G_input_shape)
    G = Dropout(.5)(G_in)
    G = LSTM(128, return_sequences=True, activation="tanh")(G)
    G = Dropout(.5)(G)
    G = Dense(num_channels, activation="tanh")(G)

    #PORTION OF MODEL RESPONSIBLE FOR OUTPUTTING GENERATOR DATA CONCATENATED W/ CLASS LABELS
    G_split = Lambda(lambda x: x[:, :, latent_dim:(latent_dim + num_labels)])(G_in) #select just labels
    G_concat = concatenate([G,G_split],axis=2) #reattach labels to each time step in the generated data

    #RETURN VERSION OF SYNTHETIC DATA W/ AND W/O LABELS ATTACHED
    #THIS ALLOWS FOR DATA TO BE ANALYZED BUT FOR DISCRIMINATOR TO RECEIVE CONDITIONAL INPUT
    G = Model(inputs=G_in, outputs=[G, G_concat]) 
    return G

'''
Below are the models for which the input and labels are fed to separate branches of the
network. The hope is that the network will be able to learn a representation of the different
classes which will then allow for the generated data to be different for different classes.
'''

def create_cond_D_multibranch(seg_length, num_channels, num_labels):
    dropout = 0.5
    input_lstm_shape = (seg_length, num_channels)
    input_lstm = Input(shape=input_lstm_shape)
    lstm = LSTM(125, return_sequences=True, activation="tanh", name="d1")(input_lstm)
    lstm = Dropout(dropout)(lstm)
    lstm = LSTM(125, activation="tanh", name="d2")(lstm)
    lstm = Dropout(dropout)(lstm)

    # This is the dense part of the network that takes the label vectors
    input_dense_shape = num_labels
    input_dense = Input(shape=(input_dense_shape,))
    dense = Dense(125, activation="sigmoid", name="d3")(input_dense)

    # WE THEN NEED TO COMBINE THESE BRANCHES AND DEFINE THE LAYERS FOR BOTH TASKS
    merged = concatenate([lstm, dense])
    real_fake_output = Dense(1, name="d4", activation="sigmoid")(merged)
    #class_label_output = Dense(num_labels, name="d5", activation="softmax")(merged)

    # also outputs the label input so they can be passed to the discriminator in sequential fashion
    model = Model(inputs=[input_lstm, input_dense], outputs=[real_fake_output], name="D")

    return model

def create_cond_G_multibranch(seg_length, num_channels, num_labels, latent_dim):

    #inputting both noise and labels separately since they're fed to different dense layers before being fed to the lstm that generates the data
    input_noise = Input(shape=(latent_dim,))
    input_labels = Input(shape=(num_labels,))
    #input_labels = Lambda(lambda x: x)(input_labels) #had to make this extra layer that does nothing in order to feed input in and directly out of network properly

    dense_noise = Dense(200, activation="tanh", name="g1")(input_noise)
    dense_noise = Dropout(.5)(dense_noise)
    dense_labels = Dense(1000, activation="tanh", name="g2")(input_labels)
    dense_labels = Dropout(.5)(dense_labels)
    merge = concatenate([dense_noise, dense_labels])
    generator = Dense(seg_length*3, name="g3", activation="tanh")(merge)
    generator = Dropout(.5)(generator)
    generator = Reshape((seg_length,3))(generator)
    generator = LSTM(3, return_sequences=True, activation="tanh", name="g4")(generator)

    model = Model(inputs=[input_noise,input_labels], outputs=[generator, input_labels], name="G")
    return model
