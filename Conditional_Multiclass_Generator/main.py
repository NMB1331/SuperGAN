'''
Main file where generator training and metric calculations take place.

Author: Skyler Norgaard
'''
import numpy as np
import models
import training_module as train
import input_module as input
import saving_module as save
import sys
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score

#LOAD FILE WITH NECESSARY FILEPATHS FOR GENERATION AS WELL AS FILTERED DATA FOR GIVEN CLASS LABEL
input_file = sys.argv[1]
fpath_data, fpath_classifier, class_label, model_save_directory = input.parse_input_file(input_file)
X,y,y_onehot = input.load_data_for_selected_class(fpath_data, class_label)
write_train_results = False

#VARIABLES REGARDING DATA SHAPE
num_seqs = X.shape[0]
seq_length = X.shape[1]
num_channels = X.shape[2]
input_shape = (seq_length, num_channels)
num_classes = y_onehot.shape[1]

#PARAMETERS RELATED TO TRAINING
latent_dim = 10 #length of random input fed to generator
epochs = 100 #num training epochs
batch_size = 25 #num instances generated for G/D training
test_size = 100 #num instances generated for validating data
real_synthetic_ratio = 5 #num synthetic instances per real instance for computing RTS metric
synthetic_synthetic_ratio = 10 #num synthetic instances to compare for computing STS metric
disc_lr = .01 #learning rate of discriminator
accuracy_threshold = .8 #threshold to stop generator training

#WEIGHTS FOR DIFFERENT TERMS IN THE LOSS FUNCTION
D_loss_weight = 1
C_loss_weight = 1
SFD_loss_weight = 0 # Change filenames if this is 1

#LOAD THE PRE-TRAINED CLASSIFIER
C = load_model(fpath_classifier)
C.name = "C"

#CREATE GENERATOR AND DISCRIMINATOR
G = models.create_G(seq_length, num_channels, latent_dim)
D = models.create_D(seq_length, num_channels)
D_to_freeze = D
D_model = models.compile_discriminator_model(D, disc_lr)

#CREATE STATISTICAL FEATURE NETWORK AND COMPUTE FEATURE VECTOR FOR REAL DATA (used in loss function)
feature_net = models.create_statistical_feature_net(seq_length, num_channels)
S_X_train = np.repeat(np.reshape(np.mean(feature_net.predict(X, batch_size), axis=0), (1, num_channels*9)), batch_size, axis=0)  
S_X_test = np.repeat(np.reshape(np.mean(feature_net.predict(X, batch_size), axis=0), (1, num_channels*9)), test_size, axis=0)  # made a vector with the averages to check against the generated data


#CREATE FULL ARCHITECTURE WHERE OUPUT OF GENERATOR IS FED TO DISCRIMINATOR AND CLASSIFIER
for layer in D_to_freeze.layers:
    layer.trainable = False
GCD = Model(inputs=G.input, outputs=[D_to_freeze(G.output), C(G.output),feature_net(G.output)])
GCD.compile(loss={"D":"binary_crossentropy","C":"categorical_crossentropy","SFN": train.euc_dist_loss}, 
			optimizer="adam", metrics={"D":"accuracy",'C':"accuracy"},
			loss_weights = {"D": D_loss_weight, "C": C_loss_weight,"SFN": SFD_loss_weight})


GC_acc=0
epoch=1
max_RTS = 0
max_RTS_epoch = 0
min_STS = 1000
min_STS_epoch = 0
while GC_acc<accuracy_threshold:
    print("Epoch: " + str(epoch))

    #TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
    D_loss_vec = train.train_D(batch_size, X, G, D_model, latent_dim)
    GCD_loss_vec = train.train_G(batch_size, X, class_label, S_X_train, num_classes, GCD, latent_dim)
    D_acc = D_loss_vec[1] #accuracy for discriminator during its "turn" for training
    GD_acc = GCD_loss_vec[4] #accuracy for generator in tricking discriminator
    print("D Acc: " + str(D_acc))
    print("G Acc in tricking D: " + str(GD_acc))

    #GENERATE SYNTHETIC DATA AND FEED TO CLASSIFIER TO DETERMINE ACCURACY
    synthetic_data = train.generate_synthetic_data(test_size, G, latent_dim, seq_length)
    pred = C.predict_classes(synthetic_data,test_size,verbose=0)
    true = [class_label]*test_size
    GC_acc = accuracy_score(true, pred)
    print("C acc for synthetic data: " + str(GC_acc))

    #COMPUTE RTS AND STS METRICS
    mean_RTS_sim, mean_STS_sim = train.compute_similarity_metrics(synthetic_data, X, test_size,real_synthetic_ratio, synthetic_synthetic_ratio)
    print("RTS similarity: " + str(mean_RTS_sim))
    print("STS similarity: " + str(mean_STS_sim))

    #COMPUTE STATISTICAL FEATURE DISTANCE
    synthetic_features = feature_net.predict(synthetic_data, test_size, verbose=0)
    SFD = train.compute_SFD(synthetic_features, S_X_test)
    print("SFD: " + str(SFD))

    #IF DESIRED, SAVE GENERTOR MODEL / WRITE TRAINING RESULTS
    if model_save_directory!=False:
        save.save_G(G, epoch, class_label, model_save_directory)
    if write_train_results == True:
        save.write_results(epoch, class_label, D_acc, GD_acc, GC_acc, mean_RTS_sim, mean_STS_sim)

    #RE-EVALUATE OUR TALLIES OF EPOCHS WITH LOWEST STS, HIGHEST RTS
    if (mean_RTS_sim > max_RTS):
        max_RTS_epoch = epoch
    if (mean_STS_sim < min_RTS):
        min_STS_epoch = epoch

    epoch+=1
    one_segment_real = np.reshape(X[np.random.randint(0, X.shape[0], 1)], (seq_length, num_channels))

# PRINT THE EPOCHS WITH THE HIGHEST RTS AND LOWEST STS (HIGH SIMILARITY TO REAL, HIGH VARIANCE)
print("\n\nMAX RTS EPOCH: {}    MIN STS EPOCH: {}".format(max_RTS_epoch, min_STS_epoch))