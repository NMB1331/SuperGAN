# SuperGAN

## Summary
This repository contains code for WSU's SuperGAN, a generative adversarial neural network architecture for the creation of synthetic time series data. 

## Purpose
Many real-world applications for deep learning require sensitive data (such as work in the medical field) or have sparse datasets (deep space sensor data from NASA) that prohibit the larger research community from contributing. This model allows for the creation of synthetic time series data that mimics the structure and patterns of real data, allowing synthetic datasets to be created and disseminated.

## Data
We used three datasets in our experimental evaluation: the publicly available Daily and Sports Activities dataset, the PTB Diagnostic ECG Dataset, and the Epileptic Seizure Recognition Dataset.

The original Epileptic Seizure Recognition Data Set from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. 500 individuals participated and gave one brain sample for 23.6 seconds, which was divided into 23 sample readings and split into separate one second chunks, and thus each row is now a recording of brain activity for one second. The original dataset consisted of five activity labels being performed during the brain scan, four of normal activities and one where the participant experienced an epileptic seizure. To simplify the task as a proof-of-concept for our approach, we transformed the data into a classification problem with the goal of generating data a classifier could be trained on to determine if someone was having a seizure or not. 

The ECG Heartbeat Categorization Dataset consists of data from the PTB Diagnostic ECG Database and the the MIT-BIH Arrhythmia Dataset, to acquire a sufficient number of samples for training deep neural networks. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals are preprocessed and segmented, with each segment corresponding to a heartbeat. The data is split into two categories, one for normal and another for abnormal heartbeats. We used this dataset with the goal of generating synthetic data a classifier could be trained on to determine if someone had a healthy or abnormal heartbeat.

The Daily Sports and Activities dataset features eight subjects (four male, four female) who perform 19 different activities for five minutes each. The five minute segments are then divided into five second windows. For our architecture comparison experiments, we divide the data by sensor type and create separate datasets for the accelerometer and gyroscope data. This helps us to get a better sense of how our method performs on varying types of data. We then utilize the data from the accelerometer / gyroscope sensor that was placed on the subjects' right leg and select nine sports activities for our analysis. In particular, the nine sports activities are:
1) Walking on a treadmill with a speed of 4 km/h in flat position
2) Walking on a treadmill with a speed of 4 km/h in 15 degree inclined position
3) Running on a treadmill with a speed of 8 km/h
4) Exercising on a stepper
5) Exercising on a cross trainer
6) Cycling on an exercise bike in horizontal position
7) Cycling on an exercise bike in vertical position
8) Rowing
9) Jumping

For preprocessing, we perform L2 normalization on the data, which results in all values falling within the range (-1,1), which is the range of the hyperbolic tangent function and thus is the range of our SuperGAN generator.

## Results
We conducted experiments to find the optimal hyperparameters, and found that TSTR (Train on Synthetic, Test on Real) accuracies tended to get better when generated data appeared to have a low Statistical Feature Distances during GAN training (SFDs, which is the Euclidean distance between real and synthetic samples), low values for the cosine similarity between synthetic samples and themselves, (STS, which indicates that the generator isn’t just producing the same sample over and over) and low values for the real to synthetic similarity (RTS, a measurement of cosine similarity between real and synthetic data points as a means of evaluating how similar they are to each other). These experiments provide an exciting opportunity for future work, as they suggest that more adjustment, testing, and optimization of hyperparameter values can yield even better results. To date, we have found that it is best to train the pre-trained classifier for the GAN training for 12 epochs, and the actual GAN for about 40 epochs. Further, results tended to improve when weighting the classifier’s contribution to the loss function during GAN training to 0.75, the discriminator’s input to the loss function to 1.25, and leaving the weight of the statistical feature distance  at 1. (For more information, see the section on GAN training.) 

For our experiment with the Epileptic Seizure Recognition Data Set, we used the above optimized hyperparameters to pre-train a classifier with an accuracy of 97.5% when classifying between samples with a seizure and samples without a seizure. We then trained the GAN for 40 epochs, and generated 2000 synthetic samples per class. Finally, we trained a classifier on this data, and validated it on real. We achieved a peak validation accuracy of 70%. By comparison, following the same procedure, the Auxiliary GAN method (see other section) achieved a maximum accuracy of only 57%.

For our experiment with the ECG Heartbeat Categorization Dataset, we followed the same procedure as above, and achieved a maximum TSTR validation accuracy of 73%. 
