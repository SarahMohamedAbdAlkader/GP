# This is the Python code for the MATLAB implementation under the name of "EEG_modified.m"
# For the Graduation project "EEG Based Computer Aided Diagnosis for Children's Brain Disorders"
# Abdelrahman Ramzy, Bassel Samer, Mohamed Essam & Sarah Abdelkader
# Under the supervision of Dr. Sherif EL-Gohary
# SBME 2019


# References Used
# [1] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
# [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis, entropy
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn import tree
from sklearn import svm
import time

# Data-sets from Bonn University for Epilepsy
# Each set has 4096 rows and 100 columns
# Each set is converted to a numpy array to be easily dealt with
F = pd.read_csv('F:/SBE.4/GP/Codes/Latest Code/Datasets/F.csv')
Raw_F = np.transpose(np.array(F).astype(np.double))
N = pd.read_csv('F:/SBE.4/GP/Codes/Latest Code/Datasets/N.csv')
Raw_N = np.transpose(np.array(N).astype(np.double))
O = pd.read_csv('F:/SBE.4/GP/Codes/Latest Code/Datasets/O.csv')
Raw_O = np.transpose(np.array(O).astype(np.double))
S = pd.read_csv('F:/SBE.4/GP/Codes/Latest Code/Datasets/S.csv')
Raw_S = np.transpose(np.array(S).astype(np.double))
Z = pd.read_csv('F:/SBE.4/GP/Codes/Latest Code/Datasets/Z.csv')
Raw_Z = np.transpose(np.array(Z).astype(np.double))

# Variables Used
fs = 173.61  # Hz
f = 40  # Hz

# # Filtering the signals
# # Filter parameters
# cutoff = 40
# nyquistFrequency = 0.5*fs
# filterOrder = cutoff
# filterCutoff = cutoff/nyquistFrequency
# # Filter coefficients
# # References [1][2]
# b, a = butter(filterOrder, filterCutoff, 'low', analog=False)
# The signal is already filtered as stated in the paper
Filtered_F = Raw_F
Filtered_N = Raw_N 
Filtered_O = Raw_O 
Filtered_S = Raw_S 
Filtered_Z = Raw_Z 

# Signal Grouping
Ictal = np.transpose(np.array(S))
Normal = np.transpose(np.concatenate((Z, O), axis=1))
InterIctal = np.transpose(np.concatenate((F, N), axis=1))

# Dividing the data into a 70:30 ratio
trainingSizeNormal = int(np.shape(Normal)[0] * 0.7)
trainingSizeIctal = int(np.shape(Ictal)[0] * 0.7)
trainingSizeInterIctal = int(np.shape(InterIctal)[0] * 0.7)

testingSizeNormal = int(np.shape(Normal)[0] * 0.3)
testingSizeIctal = int(np.shape(Ictal)[0] * 0.3)
testingSizeInterIctal = int(np.shape(InterIctal)[0] * 0.3)


# Creating the labels
Normal_Label_Training = np.zeros((trainingSizeNormal, 1), dtype=int)
Ictal_Label_Training = -1*np.ones((trainingSizeIctal, 1), dtype=int)
InterIctal_Label_Training = 2*np.ones((trainingSizeInterIctal, 1), dtype=int)
Normal_Label_Testing = np.zeros((testingSizeNormal, 1), dtype=int)
Ictal_Label_Testing = -1*np.ones((testingSizeIctal, 1), dtype=int)
InterIctal_Label_Testing = 2*np.ones((testingSizeInterIctal, 1), dtype=int)
trainingLabels = np.concatenate((Normal_Label_Training, Ictal_Label_Training, InterIctal_Label_Training), axis=0)
testingLabels = np.concatenate((Normal_Label_Testing, Ictal_Label_Testing, InterIctal_Label_Testing), axis=0)


Normal_Training = Normal[:trainingSizeNormal, :]
Normal_Testing = Normal[trainingSizeNormal:, :]
Ictal_Training = Ictal[:trainingSizeIctal, :]
Ictal_Testing = Ictal[trainingSizeIctal:, :]
InterIctal_Training = InterIctal[:trainingSizeInterIctal, :]
InterIctal_Testing = InterIctal[trainingSizeInterIctal:, :]


# Applying the wavelet transform to divide the set into its components equivalent to Delta, Theta, Alpha, Beta & Gamma
# The refrences used for this part are the following
# [1] http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# [2] "Feature Extraction ofEpilepsy EEG  using Discrete Wavelet Transform" from Minia University By: AsmaaHamad et al.
# [3] https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
Normal_DWT_Training = pywt.wavedec(Normal_Training, 'db1', level=4)
Normal_DWT_Testing = pywt.wavedec(Normal_Testing, 'db1', level=4)
Ictal_DWT_Training = pywt.wavedec(Ictal_Training, 'db1', level=4)
Ictal_DWT_Testing = pywt.wavedec(Ictal_Testing, 'db1', level=4)
InterIctal_DWT_Training = pywt.wavedec(InterIctal_Training, 'db1', level=4)
InterIctal_DWT_Testing = pywt.wavedec(InterIctal_Testing, 'db1', level=4)

# Features Extraction 
# [1] Mean
Normal_Delta_Training_Mean = np.reshape(np.mean(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_Mean = np.reshape(np.mean(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_Mean = np.reshape(np.mean(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_Mean = np.reshape(np.mean(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_Mean = np.reshape(np.mean(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_Mean = np.reshape(np.mean(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_Mean = np.reshape(np.mean(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Mean = np.reshape(np.mean(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_Mean = np.reshape(np.mean(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Mean = np.reshape(np.mean(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Mean = np.reshape(np.mean(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Mean = np.reshape(np.mean(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Mean = np.reshape(np.mean(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Mean = np.reshape(np.mean(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Mean = np.reshape(np.mean(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Mean = np.reshape(np.mean(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_Mean = np.reshape(np.mean(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_Mean = np.reshape(np.mean(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_Mean = np.reshape(np.mean(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_Mean = np.reshape(np.mean(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_Mean = np.reshape(np.mean(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_Mean = np.reshape(np.mean(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Mean = np.reshape(np.mean(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_Mean = np.reshape(np.mean(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Mean = np.reshape(np.mean(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Mean = np.reshape(np.mean(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Mean = np.reshape(np.mean(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Mean = np.reshape(np.mean(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Mean = np.reshape(np.mean(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Mean = np.reshape(np.mean(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))



# [2] Standard Deviation
Normal_Delta_Training_STD = np.reshape(np.std(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_STD = np.reshape(np.std(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_STD = np.reshape(np.std(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_STD = np.reshape(np.std(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_STD = np.reshape(np.std(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_STD = np.reshape(np.std(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_STD = np.reshape(np.std(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_STD = np.reshape(np.std(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_STD = np.reshape(np.std(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_STD = np.reshape(np.std(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_STD = np.reshape(np.std(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_STD = np.reshape(np.std(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_STD = np.reshape(np.std(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_STD = np.reshape(np.std(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_STD = np.reshape(np.std(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_STD = np.reshape(np.std(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_STD = np.reshape(np.std(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_STD = np.reshape(np.std(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_STD = np.reshape(np.std(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_STD = np.reshape(np.std(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_STD = np.reshape(np.std(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_STD = np.reshape(np.std(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_STD = np.reshape(np.std(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_STD = np.reshape(np.std(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_STD = np.reshape(np.std(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_STD = np.reshape(np.std(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_STD = np.reshape(np.std(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_STD = np.reshape(np.std(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_STD = np.reshape(np.std(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_STD = np.reshape(np.std(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [3] Variance 
Normal_Delta_Training_VAR = np.reshape(np.var(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_VAR = np.reshape(np.var(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_VAR = np.reshape(np.var(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_VAR = np.reshape(np.var(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_VAR = np.reshape(np.var(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_VAR = np.reshape(np.var(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_VAR = np.reshape(np.var(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_VAR = np.reshape(np.var(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_VAR = np.reshape(np.var(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_VAR = np.reshape(np.var(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_VAR = np.reshape(np.var(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_VAR = np.reshape(np.var(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_VAR = np.reshape(np.var(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_VAR = np.reshape(np.var(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_VAR = np.reshape(np.var(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_VAR = np.reshape(np.var(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_VAR = np.reshape(np.var(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_VAR = np.reshape(np.var(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_VAR = np.reshape(np.var(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_VAR = np.reshape(np.var(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_VAR = np.reshape(np.var(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_VAR = np.reshape(np.var(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_VAR = np.reshape(np.var(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_VAR = np.reshape(np.var(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_VAR = np.reshape(np.var(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_VAR = np.reshape(np.var(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_VAR = np.reshape(np.var(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_VAR = np.reshape(np.var(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_VAR = np.reshape(np.var(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_VAR = np.reshape(np.var(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [4] Median 
Normal_Delta_Training_Median = np.reshape(np.median(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_Median = np.reshape(np.median(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_Median = np.reshape(np.median(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_Median = np.reshape(np.median(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_Median = np.reshape(np.median(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_Median = np.reshape(np.median(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_Median = np.reshape(np.median(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Median = np.reshape(np.median(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_Median = np.reshape(np.median(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Median = np.reshape(np.median(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Median = np.reshape(np.median(InterIctal_DWT_Training[0], axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Median = np.reshape(np.median(InterIctal_DWT_Training[1], axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Median = np.reshape(np.median(InterIctal_DWT_Training[2], axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Median = np.reshape(np.median(InterIctal_DWT_Training[3], axis=1),
                                             (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Median = np.reshape(np.median(InterIctal_DWT_Training[4], axis=1),
                                              (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Median = np.reshape(np.median(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_Median = np.reshape(np.median(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_Median = np.reshape(np.median(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_Median = np.reshape(np.median(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_Median = np.reshape(np.median(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_Median = np.reshape(np.median(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_Median = np.reshape(np.median(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Median = np.reshape(np.median(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_Median = np.reshape(np.median(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Median = np.reshape(np.median(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Median = np.reshape(np.median(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Median = np.reshape(np.median(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Median = np.reshape(np.median(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Median = np.reshape(np.median(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Median = np.reshape(np.median(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [5] Kurtosis
Normal_Delta_Training_kurtosis = np.reshape(kurtosis(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_kurtosis = np.reshape(kurtosis(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_kurtosis = np.reshape(kurtosis(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_kurtosis = np.reshape(kurtosis(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_kurtosis = np.reshape(kurtosis(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_kurtosis = np.reshape(kurtosis(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_kurtosis = np.reshape(kurtosis(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_kurtosis = np.reshape(kurtosis(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_kurtosis = np.reshape(kurtosis(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_kurtosis = np.reshape(kurtosis(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Training[0], axis=1),
                                                (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Training[1], axis=1),
                                                (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Training[2], axis=1),
                                                (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Training[3], axis=1),
                                               (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Training[4], axis=1),
                                                (trainingSizeInterIctal, 1))

Normal_Delta_Testing_kurtosis = np.reshape(kurtosis(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_kurtosis = np.reshape(kurtosis(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_kurtosis = np.reshape(kurtosis(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_kurtosis = np.reshape(kurtosis(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_kurtosis = np.reshape(kurtosis(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_kurtosis = np.reshape(kurtosis(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_kurtosis = np.reshape(kurtosis(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_kurtosis = np.reshape(kurtosis(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_kurtosis = np.reshape(kurtosis(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_kurtosis = np.reshape(kurtosis(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_kurtosis = np.reshape(kurtosis(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [6] Skewness 
Normal_Delta_Training_skew = np.reshape(skew(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_skew = np.reshape(skew(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_skew = np.reshape(skew(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_skew = np.reshape(skew(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_skew = np.reshape(skew(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_skew = np.reshape(skew(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_skew = np.reshape(skew(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_skew = np.reshape(skew(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_skew = np.reshape(skew(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_skew = np.reshape(skew(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_skew = np.reshape(skew(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_skew = np.reshape(skew(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_skew = np.reshape(skew(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_skew = np.reshape(skew(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_skew = np.reshape(skew(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_skew = np.reshape(skew(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_skew = np.reshape(skew(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_skew = np.reshape(skew(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_skew = np.reshape(skew(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_skew = np.reshape(skew(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_skew = np.reshape(skew(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_skew = np.reshape(skew(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_skew = np.reshape(skew(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_skew = np.reshape(skew(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_skew = np.reshape(skew(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_skew = np.reshape(skew(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_skew = np.reshape(skew(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_skew = np.reshape(skew(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_skew = np.reshape(skew(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_skew = np.reshape(skew(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [7] Energy 
Normal_Delta_Training_Energy = np.reshape(np.sum(np.square(Normal_DWT_Training[0]), axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_Energy = np.reshape(np.sum(np.square(Normal_DWT_Training[1]), axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_Energy = np.reshape(np.sum(np.square(Normal_DWT_Training[2]), axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_Energy = np.reshape(np.sum(np.square(Normal_DWT_Training[3]), axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_Energy = np.reshape(np.sum(np.square(Normal_DWT_Training[4]), axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Training[0]), axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Training[1]), axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Training[2]), axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Training[3]), axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Training[4]), axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Training[0]), axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Training[1]), axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Training[2]), axis=1),
                                              (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Training[3]), axis=1),
                                             (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Training[4]), axis=1),
                                              (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Energy = np.reshape(np.sum(np.square(Normal_DWT_Testing[0]), axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_Energy = np.reshape(np.sum(np.square(Normal_DWT_Testing[1]), axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_Energy = np.reshape(np.sum(np.square(Normal_DWT_Testing[2]), axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_Energy = np.reshape(np.sum(np.square(Normal_DWT_Testing[3]), axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_Energy = np.reshape(np.sum(np.square(Normal_DWT_Testing[4]), axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Testing[0]), axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Testing[1]), axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Testing[2]), axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Testing[3]), axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Energy = np.reshape(np.sum(np.square(Ictal_DWT_Testing[4]), axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Testing[0]), axis=1),
                                             (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Testing[1]), axis=1),
                                             (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Testing[2]), axis=1),
                                             (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Testing[3]), axis=1),
                                            (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Energy = np.reshape(np.sum(np.square(InterIctal_DWT_Testing[4]), axis=1),
                                             (testingSizeInterIctal, 1))


# [8] Minimum coefficient
Normal_Delta_Training_Min = np.reshape(np.min(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_Min = np.reshape(np.min(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_Min = np.reshape(np.min(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_Min = np.reshape(np.min(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_Min = np.reshape(np.min(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_Min = np.reshape(np.min(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_Min = np.reshape(np.min(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Min = np.reshape(np.min(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_Min = np.reshape(np.min(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Min = np.reshape(np.min(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Min = np.reshape(np.min(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Min = np.reshape(np.min(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Min = np.reshape(np.min(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Min = np.reshape(np.min(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Min = np.reshape(np.min(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Min = np.reshape(np.min(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_Min = np.reshape(np.min(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_Min = np.reshape(np.min(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_Min = np.reshape(np.min(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_Min = np.reshape(np.min(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_Min = np.reshape(np.min(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_Min = np.reshape(np.min(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Min = np.reshape(np.min(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_Min = np.reshape(np.min(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Min = np.reshape(np.min(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Min = np.reshape(np.min(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Min = np.reshape(np.min(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Min = np.reshape(np.min(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Min = np.reshape(np.min(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Min = np.reshape(np.min(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [9] Maximum coefficient
Normal_Delta_Training_Max = np.reshape(np.max(Normal_DWT_Training[0], axis=1), (trainingSizeNormal, 1))
Normal_Theta_Training_Max = np.reshape(np.max(Normal_DWT_Training[1], axis=1), (trainingSizeNormal, 1))
Normal_Alpha_Training_Max = np.reshape(np.max(Normal_DWT_Training[2], axis=1), (trainingSizeNormal, 1))
Normal_Beta_Training_Max = np.reshape(np.max(Normal_DWT_Training[3], axis=1), (trainingSizeNormal, 1))
Normal_Gamma_Training_Max = np.reshape(np.max(Normal_DWT_Training[4], axis=1), (trainingSizeNormal, 1))

Ictal_Delta_Training_Max = np.reshape(np.max(Ictal_DWT_Training[0], axis=1), (trainingSizeIctal, 1))
Ictal_Theta_Training_Max = np.reshape(np.max(Ictal_DWT_Training[1], axis=1), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Max = np.reshape(np.max(Ictal_DWT_Training[2], axis=1), (trainingSizeIctal, 1))
Ictal_Beta_Training_Max = np.reshape(np.max(Ictal_DWT_Training[3], axis=1), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Max = np.reshape(np.max(Ictal_DWT_Training[4], axis=1), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Max = np.reshape(np.max(InterIctal_DWT_Training[0], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Max = np.reshape(np.max(InterIctal_DWT_Training[1], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Max = np.reshape(np.max(InterIctal_DWT_Training[2], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Max = np.reshape(np.max(InterIctal_DWT_Training[3], axis=1), (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Max = np.reshape(np.max(InterIctal_DWT_Training[4], axis=1), (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Max = np.reshape(np.max(Normal_DWT_Testing[0], axis=1), (testingSizeNormal, 1))
Normal_Theta_Testing_Max = np.reshape(np.max(Normal_DWT_Testing[1], axis=1), (testingSizeNormal, 1))
Normal_Alpha_Testing_Max = np.reshape(np.max(Normal_DWT_Testing[2], axis=1), (testingSizeNormal, 1))
Normal_Beta_Testing_Max = np.reshape(np.max(Normal_DWT_Testing[3], axis=1), (testingSizeNormal, 1))
Normal_Gamma_Testing_Max = np.reshape(np.max(Normal_DWT_Testing[4], axis=1), (testingSizeNormal, 1))

Ictal_Delta_Testing_Max = np.reshape(np.max(Ictal_DWT_Testing[0], axis=1), (testingSizeIctal, 1))
Ictal_Theta_Testing_Max = np.reshape(np.max(Ictal_DWT_Testing[1], axis=1), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Max = np.reshape(np.max(Ictal_DWT_Testing[2], axis=1), (testingSizeIctal, 1))
Ictal_Beta_Testing_Max = np.reshape(np.max(Ictal_DWT_Testing[3], axis=1), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Max = np.reshape(np.max(Ictal_DWT_Testing[4], axis=1), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Max = np.reshape(np.max(InterIctal_DWT_Testing[0], axis=1), (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Max = np.reshape(np.max(InterIctal_DWT_Testing[1], axis=1), (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Max = np.reshape(np.max(InterIctal_DWT_Testing[2], axis=1), (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Max = np.reshape(np.max(InterIctal_DWT_Testing[3], axis=1), (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Max = np.reshape(np.max(InterIctal_DWT_Testing[4], axis=1), (testingSizeInterIctal, 1))


# [10] Entropy 
Normal_Delta_Training_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Training[0])), (trainingSizeNormal, 1))
Normal_Theta_Training_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Training[1])), (trainingSizeNormal, 1))
Normal_Alpha_Training_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Training[2])), (trainingSizeNormal, 1))
Normal_Beta_Training_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Training[3])), (trainingSizeNormal, 1))
Normal_Gamma_Training_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Training[4])), (trainingSizeNormal, 1))

Ictal_Delta_Training_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Training[0])), (trainingSizeIctal, 1))
Ictal_Theta_Training_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Training[1])), (trainingSizeIctal, 1))
Ictal_Alpha_Training_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Training[2])), (trainingSizeIctal, 1))
Ictal_Beta_Training_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Training[3])), (trainingSizeIctal, 1))
Ictal_Gamma_Training_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Training[4])), (trainingSizeIctal, 1))

InterIctal_Delta_Training_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Training[0])),
                                               (trainingSizeInterIctal, 1))
InterIctal_Theta_Training_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Training[1])),
                                               (trainingSizeInterIctal, 1))
InterIctal_Alpha_Training_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Training[2])),
                                               (trainingSizeInterIctal, 1))
InterIctal_Beta_Training_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Training[3])),
                                              (trainingSizeInterIctal, 1))
InterIctal_Gamma_Training_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Training[4])),
                                               (trainingSizeInterIctal, 1))

Normal_Delta_Testing_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Testing[0])), (testingSizeNormal, 1))
Normal_Theta_Testing_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Testing[1])), (testingSizeNormal, 1))
Normal_Alpha_Testing_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Testing[2])), (testingSizeNormal, 1))
Normal_Beta_Testing_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Testing[3])), (testingSizeNormal, 1))
Normal_Gamma_Testing_Entropy = np.reshape(entropy(np.transpose(Normal_DWT_Testing[4])), (testingSizeNormal, 1))

Ictal_Delta_Testing_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Testing[0])), (testingSizeIctal, 1))
Ictal_Theta_Testing_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Testing[1])), (testingSizeIctal, 1))
Ictal_Alpha_Testing_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Testing[2])), (testingSizeIctal, 1))
Ictal_Beta_Testing_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Testing[3])), (testingSizeIctal, 1))
Ictal_Gamma_Testing_Entropy = np.reshape(entropy(np.transpose(Ictal_DWT_Testing[4])), (testingSizeIctal, 1))

InterIctal_Delta_Testing_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Testing[0])),
                                              (testingSizeInterIctal, 1))
InterIctal_Theta_Testing_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Testing[1])),
                                              (testingSizeInterIctal, 1))
InterIctal_Alpha_Testing_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Testing[2])),
                                              (testingSizeInterIctal, 1))
InterIctal_Beta_Testing_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Testing[3])),
                                             (testingSizeInterIctal, 1))
InterIctal_Gamma_Testing_Entropy = np.reshape(entropy(np.transpose(InterIctal_DWT_Testing[4])),
                                              (testingSizeInterIctal, 1))


# Collecting the features together 
Normal_Delta_Training_Features = np.concatenate((Normal_Delta_Training_Mean, Normal_Delta_Training_STD,
                                                 Normal_Delta_Training_VAR, Normal_Delta_Training_Median,
                                                 Normal_Delta_Training_kurtosis, Normal_Delta_Training_skew,
                                                 Normal_Delta_Training_Energy, Normal_Delta_Training_Min,
                                                 Normal_Delta_Training_Max), axis=1)
Normal_Theta_Training_Features = np.concatenate((Normal_Theta_Training_Mean, Normal_Theta_Training_STD,
                                                 Normal_Theta_Training_VAR, Normal_Theta_Training_Median,
                                                 Normal_Theta_Training_kurtosis, Normal_Theta_Training_skew,
                                                 Normal_Theta_Training_Energy, Normal_Theta_Training_Min,
                                                 Normal_Theta_Training_Max), axis=1)
Normal_Alpha_Training_Features = np.concatenate((Normal_Alpha_Training_Mean, Normal_Alpha_Training_STD,
                                                 Normal_Alpha_Training_VAR, Normal_Alpha_Training_Median,
                                                 Normal_Alpha_Training_kurtosis, Normal_Alpha_Training_skew,
                                                 Normal_Alpha_Training_Energy, Normal_Alpha_Training_Min,
                                                 Normal_Alpha_Training_Max), axis=1)
Normal_Beta_Training_Features = np.concatenate((Normal_Beta_Training_Mean, Normal_Beta_Training_STD,
                                                Normal_Beta_Training_VAR, Normal_Beta_Training_Median,
                                                Normal_Beta_Training_kurtosis, Normal_Beta_Training_skew,
                                                Normal_Beta_Training_Energy, Normal_Beta_Training_Min,
                                                Normal_Beta_Training_Max), axis=1)
Normal_Gamma_Training_Features = np.concatenate((Normal_Gamma_Training_Mean, Normal_Gamma_Training_STD,
                                                 Normal_Gamma_Training_VAR, Normal_Gamma_Training_Median,
                                                 Normal_Gamma_Training_kurtosis, Normal_Gamma_Training_skew,
                                                 Normal_Gamma_Training_Energy, Normal_Gamma_Training_Min,
                                                 Normal_Gamma_Training_Max), axis=1)

Normal_Delta_Testing_Features = np.concatenate((Normal_Delta_Testing_Mean, Normal_Delta_Testing_STD,
                                                Normal_Delta_Testing_VAR, Normal_Delta_Testing_Median,
                                                Normal_Delta_Testing_kurtosis, Normal_Delta_Testing_skew,
                                                Normal_Delta_Testing_Energy, Normal_Delta_Testing_Min,
                                                Normal_Delta_Testing_Max), axis=1)
Normal_Theta_Testing_Features = np.concatenate((Normal_Theta_Testing_Mean, Normal_Theta_Testing_STD,
                                                Normal_Theta_Testing_VAR, Normal_Theta_Testing_Median,
                                                Normal_Theta_Testing_kurtosis, Normal_Theta_Testing_skew,
                                                Normal_Theta_Testing_Energy, Normal_Theta_Testing_Min,
                                                Normal_Theta_Testing_Max), axis=1)
Normal_Alpha_Testing_Features = np.concatenate((Normal_Alpha_Testing_Mean, Normal_Alpha_Testing_STD,
                                                Normal_Alpha_Testing_VAR, Normal_Alpha_Testing_Median,
                                                Normal_Alpha_Testing_kurtosis, Normal_Alpha_Testing_skew,
                                                Normal_Alpha_Testing_Energy, Normal_Alpha_Testing_Min,
                                                Normal_Alpha_Testing_Max), axis=1)
Normal_Beta_Testing_Features = np.concatenate((Normal_Beta_Testing_Mean, Normal_Beta_Testing_STD,
                                               Normal_Beta_Testing_VAR, Normal_Beta_Testing_Median,
                                               Normal_Beta_Testing_kurtosis, Normal_Beta_Testing_skew,
                                               Normal_Beta_Testing_Energy, Normal_Beta_Testing_Min,
                                               Normal_Beta_Testing_Max), axis=1)
Normal_Gamma_Testing_Features = np.concatenate((Normal_Gamma_Testing_Mean, Normal_Gamma_Testing_STD,
                                                Normal_Gamma_Testing_VAR, Normal_Gamma_Testing_Median,
                                                Normal_Gamma_Testing_kurtosis, Normal_Gamma_Testing_skew,
                                                Normal_Gamma_Testing_Energy, Normal_Gamma_Testing_Min,
                                                Normal_Gamma_Testing_Max), axis=1)

Ictal_Delta_Training_Features = np.concatenate((Ictal_Delta_Training_Mean, Ictal_Delta_Training_STD,
                                                Ictal_Delta_Training_VAR, Ictal_Delta_Training_Median,
                                                Ictal_Delta_Training_kurtosis, Ictal_Delta_Training_skew,
                                                Ictal_Delta_Training_Energy, Ictal_Delta_Training_Min,
                                                Ictal_Delta_Training_Max), axis=1)
Ictal_Theta_Training_Features = np.concatenate((Ictal_Theta_Training_Mean, Ictal_Theta_Training_STD,
                                                Ictal_Theta_Training_VAR, Ictal_Theta_Training_Median,
                                                Ictal_Theta_Training_kurtosis, Ictal_Theta_Training_skew,
                                                Ictal_Theta_Training_Energy, Ictal_Theta_Training_Min,
                                                Ictal_Theta_Training_Max), axis=1)
Ictal_Alpha_Training_Features = np.concatenate((Ictal_Alpha_Training_Mean, Ictal_Alpha_Training_STD,
                                                Ictal_Alpha_Training_VAR, Ictal_Alpha_Training_Median,
                                                Ictal_Alpha_Training_kurtosis, Ictal_Alpha_Training_skew,
                                                Ictal_Alpha_Training_Energy, Ictal_Alpha_Training_Min,
                                                Ictal_Alpha_Training_Max), axis=1)
Ictal_Beta_Training_Features = np.concatenate((Ictal_Beta_Training_Mean, Ictal_Beta_Training_STD,
                                               Ictal_Beta_Training_VAR, Ictal_Beta_Training_Median,
                                               Ictal_Beta_Training_kurtosis, Ictal_Beta_Training_skew,
                                               Ictal_Beta_Training_Energy, Ictal_Beta_Training_Min,
                                               Ictal_Beta_Training_Max), axis=1)
Ictal_Gamma_Training_Features = np.concatenate((Ictal_Gamma_Training_Mean, Ictal_Gamma_Training_STD,
                                                Ictal_Gamma_Training_VAR, Ictal_Gamma_Training_Median,
                                                Ictal_Gamma_Training_kurtosis, Ictal_Gamma_Training_skew,
                                                Ictal_Gamma_Training_Energy, Ictal_Gamma_Training_Min,
                                                Ictal_Gamma_Training_Max), axis=1)

Ictal_Delta_Testing_Features = np.concatenate((Ictal_Delta_Testing_Mean, Ictal_Delta_Testing_STD,
                                               Ictal_Delta_Testing_VAR, Ictal_Delta_Testing_Median,
                                               Ictal_Delta_Testing_kurtosis, Ictal_Delta_Testing_skew,
                                               Ictal_Delta_Testing_Energy, Ictal_Delta_Testing_Min,
                                               Ictal_Delta_Testing_Max), axis=1)
Ictal_Theta_Testing_Features = np.concatenate((Ictal_Theta_Testing_Mean, Ictal_Theta_Testing_STD,
                                               Ictal_Theta_Testing_VAR, Ictal_Theta_Testing_Median,
                                               Ictal_Theta_Testing_kurtosis, Ictal_Theta_Testing_skew,
                                               Ictal_Theta_Testing_Energy, Ictal_Theta_Testing_Min,
                                               Ictal_Theta_Testing_Max), axis=1)
Ictal_Alpha_Testing_Features = np.concatenate((Ictal_Alpha_Testing_Mean, Ictal_Alpha_Testing_STD,
                                               Ictal_Alpha_Testing_VAR, Ictal_Alpha_Testing_Median,
                                               Ictal_Alpha_Testing_kurtosis, Ictal_Alpha_Testing_skew,
                                               Ictal_Alpha_Testing_Energy, Ictal_Alpha_Testing_Min,
                                               Ictal_Alpha_Testing_Max), axis=1)
Ictal_Beta_Testing_Features = np.concatenate((Ictal_Beta_Testing_Mean, Ictal_Beta_Testing_STD,
                                              Ictal_Beta_Testing_VAR, Ictal_Beta_Testing_Median,
                                              Ictal_Beta_Testing_kurtosis, Ictal_Beta_Testing_skew,
                                              Ictal_Beta_Testing_Energy, Ictal_Beta_Testing_Min,
                                              Ictal_Beta_Testing_Max), axis=1)
Ictal_Gamma_Testing_Features = np.concatenate((Ictal_Gamma_Testing_Mean, Ictal_Gamma_Testing_STD,
                                               Ictal_Gamma_Testing_VAR, Ictal_Gamma_Testing_Median,
                                               Ictal_Gamma_Testing_kurtosis, Ictal_Gamma_Testing_skew,
                                               Ictal_Gamma_Testing_Energy, Ictal_Gamma_Testing_Min,
                                               Ictal_Gamma_Testing_Max), axis=1)

InterIctal_Delta_Training_Features = np.concatenate((InterIctal_Delta_Training_Mean, InterIctal_Delta_Training_STD,
                                                     InterIctal_Delta_Training_VAR, InterIctal_Delta_Training_Median,
                                                     InterIctal_Delta_Training_kurtosis, InterIctal_Delta_Training_skew,
                                                     InterIctal_Delta_Training_Energy, InterIctal_Delta_Training_Min,
                                                     InterIctal_Delta_Training_Max), axis=1)
InterIctal_Theta_Training_Features = np.concatenate((InterIctal_Theta_Training_Mean, InterIctal_Theta_Training_STD,
                                                     InterIctal_Theta_Training_VAR, InterIctal_Theta_Training_Median,
                                                     InterIctal_Theta_Training_kurtosis, InterIctal_Theta_Training_skew,
                                                     InterIctal_Theta_Training_Energy, InterIctal_Theta_Training_Min,
                                                     InterIctal_Theta_Training_Max), axis=1)
InterIctal_Alpha_Training_Features = np.concatenate((InterIctal_Alpha_Training_Mean, InterIctal_Alpha_Training_STD,
                                                     InterIctal_Alpha_Training_VAR, InterIctal_Alpha_Training_Median,
                                                     InterIctal_Alpha_Training_kurtosis, InterIctal_Alpha_Training_skew,
                                                     InterIctal_Alpha_Training_Energy, InterIctal_Alpha_Training_Min,
                                                     InterIctal_Alpha_Training_Max), axis=1)
InterIctal_Beta_Training_Features = np.concatenate((InterIctal_Beta_Training_Mean, InterIctal_Beta_Training_STD,
                                                    InterIctal_Beta_Training_VAR, InterIctal_Beta_Training_Median,
                                                    InterIctal_Beta_Training_kurtosis, InterIctal_Beta_Training_skew,
                                                    InterIctal_Beta_Training_Energy, InterIctal_Beta_Training_Min,
                                                    InterIctal_Beta_Training_Max), axis=1)
InterIctal_Gamma_Training_Features = np.concatenate((InterIctal_Gamma_Training_Mean, InterIctal_Gamma_Training_STD,
                                                     InterIctal_Gamma_Training_VAR, InterIctal_Gamma_Training_Median,
                                                     InterIctal_Gamma_Training_kurtosis, InterIctal_Gamma_Training_skew,
                                                     InterIctal_Gamma_Training_Energy, InterIctal_Gamma_Training_Min,
                                                     InterIctal_Gamma_Training_Max), axis=1)

InterIctal_Delta_Testing_Features = np.concatenate((InterIctal_Delta_Testing_Mean, InterIctal_Delta_Testing_STD,
                                                    InterIctal_Delta_Testing_VAR, InterIctal_Delta_Testing_Median,
                                                    InterIctal_Delta_Testing_kurtosis, InterIctal_Delta_Testing_skew,
                                                    InterIctal_Delta_Testing_Energy, InterIctal_Delta_Testing_Min,
                                                    InterIctal_Delta_Testing_Max), axis=1)
InterIctal_Theta_Testing_Features = np.concatenate((InterIctal_Theta_Testing_Mean, InterIctal_Theta_Testing_STD,
                                                    InterIctal_Theta_Testing_VAR, InterIctal_Theta_Testing_Median,
                                                    InterIctal_Theta_Testing_kurtosis, InterIctal_Theta_Testing_skew,
                                                    InterIctal_Theta_Testing_Energy, InterIctal_Theta_Testing_Min,
                                                    InterIctal_Theta_Testing_Max), axis=1)
InterIctal_Alpha_Testing_Features = np.concatenate((InterIctal_Alpha_Testing_Mean, InterIctal_Alpha_Testing_STD,
                                                    InterIctal_Alpha_Testing_VAR, InterIctal_Alpha_Testing_Median,
                                                    InterIctal_Alpha_Testing_kurtosis, InterIctal_Alpha_Testing_skew,
                                                    InterIctal_Alpha_Testing_Energy, InterIctal_Alpha_Testing_Min,
                                                    InterIctal_Alpha_Testing_Max), axis=1)
InterIctal_Beta_Testing_Features = np.concatenate((InterIctal_Beta_Testing_Mean, InterIctal_Beta_Testing_STD,
                                                   InterIctal_Beta_Testing_VAR, InterIctal_Beta_Testing_Median,
                                                   InterIctal_Beta_Testing_kurtosis, InterIctal_Beta_Testing_skew,
                                                   InterIctal_Beta_Testing_Energy, InterIctal_Beta_Testing_Min,
                                                   InterIctal_Beta_Testing_Max), axis=1)
InterIctal_Gamma_Testing_Features = np.concatenate((InterIctal_Gamma_Testing_Mean, InterIctal_Gamma_Testing_STD,
                                                    InterIctal_Gamma_Testing_VAR, InterIctal_Gamma_Testing_Median,
                                                    InterIctal_Gamma_Testing_kurtosis, InterIctal_Gamma_Testing_skew,
                                                    InterIctal_Gamma_Testing_Energy, InterIctal_Gamma_Testing_Min,
                                                    InterIctal_Gamma_Testing_Max), axis=1)


# Theta, Alpha & Beta
Normal_Training_Features = np.concatenate((Normal_Theta_Training_Features, Normal_Alpha_Training_Features,
                                           Normal_Beta_Training_Features), axis=1)
Normal_Testing_Features = np.concatenate((Normal_Theta_Testing_Features, Normal_Alpha_Testing_Features,
                                          Normal_Beta_Testing_Features), axis=1)
Ictal_Training_Features = np.concatenate((Ictal_Theta_Training_Features, Ictal_Alpha_Training_Features,
                                          Ictal_Beta_Training_Features), axis=1)
Ictal_Testing_Features = np.concatenate((Ictal_Theta_Testing_Features, Ictal_Alpha_Testing_Features,
                                         Ictal_Beta_Testing_Features), axis=1)
InterIctal_Training_Features = np.concatenate((InterIctal_Theta_Training_Features, InterIctal_Alpha_Training_Features,
                                               InterIctal_Beta_Training_Features), axis=1)
InterIctal_Testing_Features = np.concatenate((InterIctal_Theta_Testing_Features, InterIctal_Alpha_Testing_Features,
                                              InterIctal_Beta_Testing_Features), axis=1)


# #####################################################Classifiers#####################################################
Training_Data = np.concatenate((Normal_Training_Features, Ictal_Training_Features, InterIctal_Training_Features))
Testing_Data = np.concatenate((Normal_Testing_Features, Ictal_Testing_Features, InterIctal_Testing_Features))
# KNN CLassifier
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(Training_Data, trainingLabels.ravel())
test_predict = classifier.predict(Testing_Data)
target_names = ['Ictal', 'Normal', 'Interictal']
Accuracy_1 = accuracy_score(test_predict, testingLabels)
Report_1 = classification_report(testingLabels, test_predict, labels=[-1, 0, 2], target_names=target_names)
Precision_1, Sensitivity_1, fscore_1, support_1 = precision_recall_fscore_support(testingLabels,
                                                                                  test_predict, average='macro')
print("K Nearest Neighbours")
print('Report : ', Report_1)
print('Accuracy Score :', Accuracy_1 * 100, '%')
print('Precision Score :', Precision_1 * 100, '%')
print('Sensitivity Score :', Sensitivity_1 * 100, '%')


# Decision Tree Classifier
DTC = tree.DecisionTreeClassifier()
DTC.fit(Training_Data, trainingLabels)
DTC_Predict = DTC.predict(Testing_Data)
Accuracy_2 = accuracy_score(DTC_Predict, testingLabels)
Report_2 = classification_report(testingLabels, DTC_Predict, labels=[-1, 0, 2], target_names=target_names)
Precision_2, Sensitivity_2, fscore_2, support_2 = precision_recall_fscore_support(testingLabels,
                                                                                  DTC_Predict, average='macro')
print("Decision Tree")
print('Report : ', Report_2)
print('Accuracy Score :', Accuracy_2 * 100, '%')
print('Precision Score :', Precision_2 * 100, '%')
print('Sensitivity Score :', Sensitivity_2 * 100, '%')


# SVM Classifier
start = time.time()
#SVM = svm.SVC(gamma='scale', decision_function_shape='ovo')
SVM = svm.LinearSVC()
SVM.fit(Training_Data, trainingLabels.ravel())
# dec = SVM.decision_function([[1]])
SVM_Predict = SVM.predict(Testing_Data)
Accuracy_3 = accuracy_score(SVM_Predict, testingLabels)
Report_3 = classification_report(testingLabels, SVM_Predict, labels=[-1, 0, 2], target_names=target_names)
Precision_3, Sensitivity_3, fscore_3, support_3 = precision_recall_fscore_support(testingLabels,
                                                                                  SVM_Predict, average='macro')
print("Support Vector Machine")
print('Report : ', Report_3)
print('Accuracy Score :', Accuracy_3 * 100, '%')
print('Precision Score :', Precision_3 * 100, '%')
print('Sensitivity Score :', Sensitivity_3 * 100, '%')


# ## **************************************** Results *************************************** ## #
# # -------------------------- Normal & Ictal--------------------------# #
"""
[1] Removing the Delta caused a reduction in the accuracy to 95
[2] Removing the Theta caused the accuracy to be 100% meaning over fitting maybe?! 
[3] Removing Alpha, Beta and Gamma bands didn't cause any differences

[4] Delta alone -> 94% 
[5] Theta alone -> 93% 
[6] Alpha alone -> 92% 
[7] Beta alone -> 91% 
[8] Gamma alone -> 86%
"""
# ## Conclusion
"""
The best case was when Delta and Theta bands were used which resulted in 98.89%
When All 5 bands were together, 98.89% was resulted, but a lot more computation time.
"""

# # -------------------------- Normal, Ictal & InterIctal--------------------------# #
"""
[1] All 5 bands together resulted in an accuracy of 89.333% 
[2] Using Alpha and Theta bands alone resulted in an accuracy of 76%
[3] Removing the Gamma band alone didn't affect the accuracy of 89.333%
[4] Removing the Beta & Gamma bands lowered the accuracy to 88.667%
[5] Removing the Alpha & Gamma bands lowered the accuracy to 85.334%
[6] Removing the Delta & Gamma bands increased the accuracy to 91.333%
[7] Removing the Theta & Gamma bands lowered the accuracy tp 86.667%
"""
# ## Conclusion
"""
The best case was when using the Theta, Alpha and Beta bands which resulted in an accuracy of 91.333%
"""