'''
Version of main file where generator is conditioned over class labels (rather
than splitting data initially and then training a generator for each label separately)

IN THIS CASE, WE ARENT LOADING A PRE-TRAINED CLASSIFIER TO HELP WITH TRAINING, WE'RE IMPORTING ONE AND TRAINING IT
AS WE GO. ITS KIND OF LIKE A HYBRIC BETWEEN THE AUXILIARY AND OURS

Author: Skyler Norgaard
'''
import numpy as np
import models
import conditional_models
import training_module as train
import conditional_training_module as cond_train
import input_module as input_mod
import saving_module as save
import sys
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score
import classifier_example

#LOAD FILE WITH NECESSARY FILEPATHS FOR GENERATION AS WELL AS FULL DATA
input_file = sys.argv[1]
fpath_data, fpath_classifier, class_label, model_save_directory, SFD_loss, outfile = input_mod.parse_input_file(input_file)
X,y,y_onehot = input_mod.load_data_full(fpath_data)
write_train_results = True

#VARIABLES REGARDING DATA SHAPE
num_seqs = X.shape[0]
seq_length = X.shape[1]
num_channels = X.shape[2]
input_shape = (seq_length, num_channels)
num_classes = y_onehot.shape[1]

#PARAMETERS RELATED TO TRAINING
latent_dim = 10 #length of random input fed to generator
epochs = 100 #num training epochs
instances_per_class_train = 10 #number of instances to generate per class for training
instances_per_class_test = 25
batch_size = instances_per_class_train * num_classes #total num instances generated for G/D training
test_size = instances_per_class_test * num_classes #num instances generated for validating data
real_synthetic_ratio = 5 #num synthetic instances per real instance for computing RTS metric
synthetic_synthetic_ratio = 10 #num synthetic instances to compare for computing STS metric
disc_lr = .01 #learning rate of discriminator
accuracy_threshold = .8 #threshold to stop generator training

#CREATE VECTOR OF LABELS USED FOR TRAINING C AND CONDITIONING G AND D
class_labels_target_train = cond_train.create_target_label_vector_all_classes(instances_per_class_train, seq_length, num_classes)[1]
class_labels_target_test = cond_train.create_target_label_vector_all_classes(instances_per_class_test, seq_length, num_classes)[1]
class_labels_input_train = cond_train.create_input_label_vector_all_classes(instances_per_class_train, seq_length, num_classes)


#WEIGHTS FOR DIFFERENT TERMS IN THE LOSS FUNCTION
D_loss_weight = 1
C_loss_weight = 1
SFD_loss_weight = float(SFD_loss)

#LOAD THE CLASSIFIER
C = classifier_example.create_deepconv(seq_length, num_channels, num_classes)
C.name = "C"
'''
C = load_model(fpath_classifier)
C.name = "C"
'''


#CREATE GENERATOR AND DISCRIMINATOR
G = conditional_models.create_cond_G(seq_length, num_channels, num_classes, latent_dim)
D = conditional_models.create_cond_D(seq_length, num_channels, num_classes)
D_to_freeze = D
D_model = models.compile_discriminator_model(D, disc_lr)

#CREATE STATISTICAL FEATURE NETWORK AND COMPUTE FEATURE VECTOR FOR REAL DATA (used in loss function)
feature_net = models.create_statistical_feature_net(seq_length, num_channels)
S_X_train, S_X_test = cond_train.compute_statistical_vector(X,y,feature_net,num_channels,num_classes,instances_per_class_train,instances_per_class_test)

#CREATE FULL ARCHITECTURE WHERE OUPUT OF GENERATOR IS FED TO DISCRIMINATOR AND CLASSIFIER
for layer in D_to_freeze.layers:
    layer.trainable = False
GCD = Model(inputs=G.input, outputs=[D_to_freeze(G.output[1]), C(G.output[0]),feature_net(G.output[0])])
GCD.compile(loss={"D":"binary_crossentropy","C":"categorical_crossentropy","SFN": train.euc_dist_loss}, 
			optimizer="adam", metrics={"D":"accuracy",'C':"accuracy"},
			loss_weights = {"D": D_loss_weight, "C": C_loss_weight,"SFN": SFD_loss_weight})


GC_acc=0
epoch=1
max_RTS = 0
max_RTS_epoch = 0
min_STS = 1000
min_STS_epoch = 0
min_SFD = 10000
min_SFD_epoch = 0

while (GC_acc<accuracy_threshold) and (epoch <= 100):
    print("Epoch: " + str(epoch))

    #TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
    D_loss_vec = cond_train.train_cond_D(instances_per_class_train, X, y, class_labels_input_train, num_classes, G, D_model, seq_length, latent_dim)
    GCD_loss_vec = cond_train.train_cond_G(instances_per_class_train, X, class_labels_input_train, class_labels_target_train, S_X_train, num_classes, GCD, seq_length, latent_dim)
    D_acc = D_loss_vec[1] #accuracy for discriminator during its "turn" for training
    GD_acc = GCD_loss_vec[4] #accuracy for generator in tricking discriminator
    print("D Acc: " + str(D_acc))
    print("G Acc in tricking D: " + str(GD_acc))

    #GENERATE SYNTHETIC DATA AND FEED TO CLASSIFIER TO DETERMINE ACCURACY
    synthetic_data = cond_train.generate_synthetic_data_all_classes(test_size, G, class_label, num_classes, latent_dim, seq_length)
    pred = C.predict_classes(synthetic_data,test_size,verbose=0)
    #true = [class_label]*test_size
    GC_acc = accuracy_score(class_labels_target_test, pred)
    print("C acc for synthetic data: " + str(GC_acc))

    #COMPUTE RTS AND STS METRICS
    mean_RTS_sim, mean_STS_sim, _ = train.compute_similarity_metrics(synthetic_data, X, test_size,real_synthetic_ratio, synthetic_synthetic_ratio)
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
        save.write_results(outfile, epoch, class_label, D_acc, GD_acc, GC_acc, mean_RTS_sim, mean_STS_sim)

    #RE-EVALUATE OUR TALLIES OF EPOCHS WITH LOWEST STS, HIGHEST RTS
    if (mean_RTS_sim > max_RTS) and epoch >= 25:
        max_RTS_epoch = epoch
        max_RTS = mean_RTS_sim
    if (mean_STS_sim < min_STS) and epoch >= 25:
        min_STS_epoch = epoch
        min_STS = mean_STS_sim
    if (SFD < min_SFD) and epoch >= 25:
        min_SFD_epoch = epoch
        min_SFD = SFD

    epoch+=1
    one_segment_real = np.reshape(X[np.random.randint(0, X.shape[0], 1)], (seq_length, num_channels))

# PRINT THE EPOCHS WITH THE HIGHEST RTS AND LOWEST STS (HIGH SIMILARITY TO REAL, HIGH VARIANCE)
print("\n\nMAX RTS EPOCH: {} value: {}\nMIN STS EPOCH: {} value: {}\nMIN SFD EPOCH: {} value: {}". \
    format(max_RTS_epoch, max_RTS, min_STS_epoch, min_STS, min_SFD_epoch, min_SFD))