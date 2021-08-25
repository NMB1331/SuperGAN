'''
Functions for training generator and assessing data. In particular, contains functions for
training generator and discriminator, generating synthetic data, and computing the similarity metrics

Author: Skyler Norgaard
'''

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K

#FUNCTION FOR COMPUTING EUCLIDIAN DISTANCE (SFD) LOSS WITHIN KERAS OPTIMIZATION FUNCTION
def euc_dist_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

#FUNCTON FOR COMPUTING AVERAGE STATISTICAL FEATURE DURING TRAINING
def compute_SFD(real_features, synthetic_features):
    distance_vector = np.sqrt(np.sum(np.square(real_features - synthetic_features), axis=1))
    SFD = np.mean(distance_vector)
    return SFD

#FUNCTION FOR GENERATING RANDOM INPUT BY SAMPLING FROM NORMAL DISTRIBUTION (INPUT VARIES AT EACH TIMESTEP)
def generate_input_noise(batch_size, latent_dim, time_steps):
    return np.reshape(np.array(np.random.normal(0, 1, latent_dim * time_steps * batch_size)),(batch_size, time_steps, latent_dim))

#FUNCTION FOR GENERATING A SYNTHETIC DATA SET
def generate_synthetic_data(size, generator, latent_dim, time_steps):
    noise = generate_input_noise(size, latent_dim, time_steps)
    synthetic_data = generator.predict(noise)
    return synthetic_data

#FUNCTION FOR TRAINING GENERATOR FROM DBOTH DISCRIMINATOR AND CLASSIFIER OUTPUT
def train_G(batch_size, X, class_label, actual_features, num_labels, model, latent_dim):
    noise = generate_input_noise(batch_size, latent_dim, X.shape[1])
    real_synthetic_labels = np.ones([batch_size, 1]) #labels related to whether data is real or synthetic
    class_labels = to_categorical([class_label]*batch_size, num_classes=num_labels) #labels related to the class of the data
    loss = model.train_on_batch(noise, [real_synthetic_labels,class_labels, actual_features])
    return loss

#FUNCTION FOR TRAINING DISCRIMINATOR (FROM GENERATOR INPUT)
def train_D(batch_size, X, generator, discriminator_model, latent_dim):
    #GENERATE SYNTHETIC DATA
    noise = generate_input_noise(batch_size, latent_dim, X.shape[1])
    synthetic_data = generator.predict(noise)

    #SELECT A RANDOM BATCH OF REAL DATA
    indices_toKeep = np.random.choice(X.shape[0], batch_size, replace=False)
    real_data = X[indices_toKeep]

    #MAKE FULL INPUT AND LABELS FOR FEEDING INTO NETWORK
    full_input = np.concatenate((real_data, synthetic_data))
    real_synthetic_label = np.ones([2 * batch_size, 1])
    real_synthetic_label[batch_size:, :] = 0

    #TRAIN D AND RETURN LOSS
    loss = discriminator_model.train_on_batch(full_input, real_synthetic_label)
    return loss

#FUNCTION TO COMPUTE MEAN RTS AND STS SIMILARITY WHICH IS USED TO HELP MONITOR GENERATOR TRAINING
def compute_similarity_metrics(X_synthetic, X_real, batch_size, real_synthetic_ratio, synthetic_synthetic_ratio=10, real_real_ratio=10):

	#NECESSARY FEATURES REGARDING DATA SHAPE
	#synthetic_synthetic_ratio = X_synthetic.shape[0] - 1
	num_segs = len(X_real)
	seq_length = X_real.shape[1]
	num_channels = X_real.shape[2]
	synth_num_segs = len(X_synthetic)
	synth_seq_length = X_synthetic.shape[1]
	synth_num_channels = X_synthetic.shape[2]
	RTS_sims = []
	STS_sims = []
	RTR_sims = []

	#RESHAPE DATA INTO 2 DIMENSIONS
	X_real = X_real.reshape(num_segs,seq_length*num_channels)
	X_synthetic = X_synthetic.reshape(synth_num_segs,synth_seq_length*synth_num_channels)

    #FOR EACH FAKE SEGMENT, CALCULATE ITS SIMILARITY TO A USER DEFINED NUMBER OF REAL SEGMENTS
	for i in range(batch_size):
		indices_toCompare = np.random.choice(num_segs,real_synthetic_ratio,replace=False)
		for j in indices_toCompare:
			sim = cosine_similarity(X_real[j].reshape(1,-1), X_synthetic[i].reshape(1,-1))
			sim = sim[0,0]
			RTS_sims += [sim]


	#ALSO COMPUTE SIMILARITY BETWEEN USER DEFINED NUMBER OF SYNTHETIC SEGMENTS TO TEST FOR GENERATOR COLLAPSE
	chosen_synthetic_value = X_synthetic[np.random.choice(len(X_synthetic),1)] #get one random fake sample
	X_synthetic = np.delete(X_synthetic, chosen_synthetic_value, axis=0) #remove this sample so we dont compare to itself
	synthetic_toCompare = X_synthetic[np.random.choice(len(X_synthetic), synthetic_synthetic_ratio)]

	for other_synthetic in synthetic_toCompare:
		sim2 = cosine_similarity(chosen_synthetic_value, other_synthetic.reshape(1,-1))
		sim2 = sim2[0,0]
		STS_sims += [sim2]

	#COMPUTE SIMILARITY BETWEEN USER DEFINED NUMBER OF REAL SEGMENTS (RTR)
	chosen_real_value = X_real[np.random.choice(len(X_real),1)] #get one random fake sample
	X_real = np.delete(X_real, chosen_real_value, axis=0) #remove this sample so we dont compare to itself
	real_toCompare = X_real[np.random.choice(len(X_real), real_real_ratio)]

	for other_real in real_toCompare:
		sim3 = cosine_similarity(chosen_real_value, other_real.reshape(1,-1))
		sim3 = sim3[0,0]
		RTR_sims += [sim3]

	RTS_sims = np.array(RTS_sims)
	STS_sims = np.array(STS_sims)
	RTR_sims = np.array(RTR_sims)
	mean_RTS_sim = np.mean(RTS_sims)
	mean_STS_sim = np.mean(STS_sims)
	mean_RTR_sim = np.mean(RTR_sims)

	return mean_RTS_sim, mean_STS_sim, mean_RTR_sim