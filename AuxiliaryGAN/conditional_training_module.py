'''
Contains the functions for training the conditional models
'''

import numpy as np
import input_module as input_mod
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import backend as K


#FUNCTION FOR GENERATING RANDOM INPUT BY SAMPLING FROM NORMAL DISTRIBUTION (INPUT VARIES AT EACH TIMESTEP)
def generate_input_noise(batch_size, latent_dim, time_steps):
    return np.reshape(np.array(np.random.normal(0, 1, latent_dim * time_steps * batch_size)),(batch_size, time_steps, latent_dim))

'''
SHOULD MAKE THESE MORE EFFICIENT
'''

#FUNCTION FOR GENERATING SYNTHETIC DATA FOR ONE CLASS
def generate_synthetic_data_one_class(size, generator, class_label, num_labels, latent_dim, time_steps):
    noise = generate_input_noise(size, latent_dim, time_steps)
    class_labels_input = np.reshape(to_categorical([class_label]*(size*time_steps), num_classes=num_labels),(size, time_steps, num_labels))
    G_input = np.concatenate((noise,class_labels_input),axis=2)
    synthetic_data = generator.predict(G_input)[0] #have to choose index 0 bc this is version w/o labels attached
    return synthetic_data

#FUNCTION FOR GENERATING SYNTHETIC DATA FOR ALL CLASSES
def generate_synthetic_data_all_classes(size, generator, class_label, num_labels, latent_dim, time_steps):
    start=1
    for label in range(num_labels):
        synthetic_data_one_class = generate_synthetic_data_one_class(size, generator, class_label, num_labels, latent_dim, time_steps)

        if start==1:
            synthetic_data = synthetic_data_one_class
        else:
            synthetic_data = np.concatenate((synthetic_data, synthetic_data_one_class))

    return synthetic_data

#CREATES TIME-SERIES VECTOR OF LABELS THAT IS CONCATENATED TO THE INPUT OF BOTH G AND D
def create_input_label_vector_all_classes(instances_per_class, time_steps, num_labels):
    start=1
    for class_label in range(num_labels):
        class_labels_input_one_class = np.reshape(to_categorical([class_label]*(instances_per_class*time_steps), num_classes=num_labels),(instances_per_class, time_steps, num_labels))

        if start == 1:
            class_labels_input = class_labels_input_one_class
            start=0
        else:
            class_labels_input = np.concatenate((class_labels_input, class_labels_input_one_class))

    return class_labels_input

#CREATES VECTOR OF LABELS THAT IS USED FOR TRAINING THE GENERATOR BASED ON FEEDBACK FROM DISCRIMINATOR
def create_target_label_vector_all_classes(instances_per_class, time_steps, num_labels):
    start=1
    for class_label in range(num_labels):
        class_labels_target_onehot_one_class = to_categorical([class_label]*instances_per_class, num_classes=num_labels) #labels related to the class of the data, used as target
        class_labels_target_one_class = [class_label]*instances_per_class

        if start == 1:
            class_labels_target_onehot = class_labels_target_onehot_one_class
            class_labels_target = class_labels_target_one_class
            start=0
        else:
            class_labels_target_onehot = np.concatenate((class_labels_target_onehot, class_labels_target_onehot_one_class))
            class_labels_target += class_labels_target_one_class

    return class_labels_target_onehot, class_labels_target

#SELECTS A RANDOM BATCH OF REAL DATA (ENSURES THAT ALL CLASSES ARE REPRESENTED EVENLY)
def select_random_batch_of_real_data(X, y, instances_per_class, num_labels):
    start = 1
    for label in range(num_labels):
        indices_for_given_class = np.where(y==label)
        X_c = X[indices_for_given_class]
        indices_toKeep = np.random.choice(X_c.shape[0], instances_per_class, replace=False)
        real_data_one_class = X_c[indices_toKeep]

        if start == 1:
            real_data = real_data_one_class
            start=0
        else:
            real_data = np.concatenate((real_data, real_data_one_class))

    return real_data

#FUNCTION COMPUTES AVERAGE STATISTICS FOR EACH CLASS AND CONCATENATES RESULT INTO APPROPRIATELY
#LENGTHED VECTOR FOR TRAINING AND TESTING
def compute_statistical_vector(X, y, feature_net, num_channels, num_classes, instances_per_class_train, instances_per_class_test):
    start=1
    for label in range(num_classes):
        #SELECT DATA FOR GIVEN CLASS AND CALCULATE AVERAGE STATISTICS
        indices_toKeep = np.where(y==label)
        selected_X = X #[indices_toKeep]
        f_net_pred = np.array(feature_net.predict(selected_X))
        print("Feature net prediction shape: {}".format(f_net_pred.shape))
        feat_net_pred = np.reshape(f_net_pred, (f_net_pred.shape[0], f_net_pred.shape[1], 1))
        S_X_train_one_class = np.repeat(np.reshape(np.mean(feat_net_pred, axis=0), (1, 9)), instances_per_class_train, axis=0)  
        S_X_test_one_class = np.repeat(np.reshape(np.mean(feature_net.predict(selected_X), axis=0), (1, 9)), instances_per_class_test, axis=0)

        if start==1:
            S_X_train = S_X_train_one_class
            S_X_test = S_X_test_one_class
            start=0
        else:
            S_X_train = np.concatenate((S_X_train, S_X_train_one_class))
            S_X_test = np.concatenate((S_X_test, S_X_test_one_class))

    return S_X_train, S_X_test

#TRAINS CONDITIONAL GENERATOR WHERE CLASS LABELS ARE CONCATENATED TO INPUT VECTOR
def train_cond_G(instances_per_class, X, class_labels_input, class_labels_target, actual_features, num_labels, model, seq_length, latent_dim):
    noise = generate_input_noise(instances_per_class * num_labels, latent_dim, X.shape[1])
    real_synthetic_labels = np.ones([instances_per_class * num_labels, 1]) #labels related to whether data is real or synthetic
    #class_labels_input = create_input_label_vector_all_classes(instances_per_class, seq_length, num_labels)
    class_labels_target = create_target_label_vector_all_classes(instances_per_class, seq_length, num_labels)[0]
    G_input = np.concatenate((noise,class_labels_input),axis=2)
    loss = model.train_on_batch(G_input, [real_synthetic_labels,class_labels_target,actual_features])
    return loss

#FUNCTION FOR TRAINING CONDITIONAL DISCRIMINATOR (FROM GENERATOR INPUT)
def train_cond_D(instances_per_class, X, y, class_labels_input, num_labels, generator, discriminator_model, seq_length, latent_dim):
    #GENERATE SYNTHETIC DATA
    noise = generate_input_noise(instances_per_class * num_labels, latent_dim, X.shape[1])
    #class_labels_input = create_input_label_vector_all_classes(instances_per_class, seq_length, num_labels)
    G_input = np.concatenate((noise,class_labels_input),axis=2)
    synthetic_data = generator.predict(G_input)[1]

    #SELECT A RANDOM BATCH OF REAL DATA
    real_data = select_random_batch_of_real_data(X,y,instances_per_class, num_labels)
    real_data = np.concatenate((real_data,class_labels_input),axis=2)

    #MAKE FULL INPUT AND LABELS FOR FEEDING INTO NETWORK
    full_input = np.concatenate((real_data, synthetic_data))
    real_synthetic_label = np.ones([2 * instances_per_class * num_labels, 1])
    real_synthetic_label[(instances_per_class * num_labels):, :] = 0

    #TRAIN D AND RETURN LOSS
    loss = discriminator_model.train_on_batch(full_input, real_synthetic_label)
    return loss

'''
These are for training the auxiliary classifier version of the GAN. Again, the difference
is that the discriminator makes both predictions rather than including a separate model for
the classification task.
'''

#TRAINS CONDITIONAL GENERATOR WHERE CLASS LABELS ARE CONCATENATED TO INPUT VECTOR
def train_AC_G(instances_per_class, X, class_labels_input, class_labels_target, actual_features, num_labels, model, seq_length, latent_dim):
    noise = generate_input_noise(instances_per_class * num_labels, latent_dim, X.shape[1])
    real_synthetic_labels = np.ones([instances_per_class * num_labels, 1]) #labels related to whether data is real or synthetic
    #class_labels_input = create_input_label_vector_all_classes(instances_per_class, seq_length, num_labels)
    #class_labels_target = create_target_label_vector_all_classes(instances_per_class, seq_length, num_labels)[0]
    G_input = np.concatenate((noise,class_labels_input),axis=2)
    loss = model.train_on_batch(G_input, [real_synthetic_labels,class_labels_target,actual_features])
    return loss

#FUNCTION FOR TRAINING CONDITIONAL DISCRIMINATOR (FROM GENERATOR INPUT)
def train_AC_D(instances_per_class, X, y, class_labels_input, class_labels_target, num_labels, generator, discriminator_model, seq_length, latent_dim):
    #GENERATE SYNTHETIC DATA
    noise = generate_input_noise(instances_per_class * num_labels, latent_dim, X.shape[1])
    #class_labels_input = create_input_label_vector_all_classes(instances_per_class, seq_length, num_labels)
    G_input = np.concatenate((noise,class_labels_input),axis=2)
    synthetic_data = generator.predict(G_input)[1]

    #SELECT A RANDOM BATCH OF REAL DATA
    real_data = select_random_batch_of_real_data(X,y,instances_per_class, num_labels)
#     class_labels_input = np.array(class_labels_input)
#     class_labels_input = np.reshape(class_labels_input, (class_labels_input[0],class_labels_input.shape[2]))
    print("Class labels input: {}".format(class_labels_input))
    print("Real data shape: {}".format(real_data.shape))
    real_data = np.concatenate((real_data,class_labels_input), axis=2) #originally axis=2

    #MAKE FULL INPUT AND LABELS FOR FEEDING INTO NETWORK
    full_input = np.concatenate((real_data, synthetic_data))
    real_synthetic_label = np.ones([2 * instances_per_class * num_labels, 1])
    real_synthetic_label[(instances_per_class * num_labels):, :] = 0
    full_class_label_vector = np.concatenate((class_labels_target, class_labels_target))

    loss = discriminator_model.train_on_batch(full_input, [real_synthetic_label, full_class_label_vector])
    return loss