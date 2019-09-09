#FrameWork Library Imports 
import ml_gt.preprocessing as pp
import ml_gt.learning as learning
import ml_gt.evaluation as evaluation 

#System 
import os, pickle

#Audio
import librosa

#ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC 

# Deep Learning
import tensorflow as tf 
from tensorflow.python.client import device_lib 
from keras.backend.tensorflow_backend import set_session 
from tensorflow.python.client import device_lib 
from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Activation, Flatten, merge, Dropout 
from keras.models import Model, Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import History, EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model, to_categorical

# Random Seed
from tensorflow import set_random_seed
from numpy.random import seed

#General 
import numpy as np 

#Plotting 
import seaborn 
import matplotlib.pyplot as plt 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)

#Paths 
mycwd = os.getcwd() 
resource_path = mycwd + "\\resources"
audio_path = mycwd + "\\audio\\test"
song_path = mycwd + "\\test_songs"
model_path = mycwd + "\\models"
plot_path = mycwd + "\\plots"
log_path = mycwd + "\\logs"

#General Parameters
sr = 44100
hop_length = 512 

#MFCC Parameters
n_fft = 2048 
n_mels = 128 
n_mfcc = 13 

#CQT Parameters
fminval = 36
fmin = librosa.midi_to_hz(fminval)
n_bins = 72

#pp.split_folder(labels=['EADGBE', 'DGDGBD', 'DAEGAD', 'DADGBE'], folder=song_path)


cf = "test_classes_windowed.npy"
lf = "test_labels_windowed.npy"
log_name_cqt = "test_cqt_win_knn_{0}_{1}_{2}".format(hop_length, fminval, n_bins)
files_path = "test_files_win"


pp.create_labels(audio_path=audio_path, output_classes=cf, output_labels=lf)
labels = np.load(resource_path + "\\labels\\" + lf)
labelencoder = LabelEncoder() 
labelencoder.classes_ = np.load(resource_path + "\\labels\\" + cf)
classes = labelencoder.transform(labels)
pp.one_hot(classes, "test_onehotlabels.npy")
onehotlabels = np.load(resource_path+"\\labels\\test_onehotlabels.npy")
print(labels.shape)
print(classes.shape)
print(onehotlabels.shape)


print(labels[10])

'''
#CREATE VECTORS
feature_vectors, files = pp.get_cqt_folder(path=audio_path)
pp.save_cqt(feature_vectors, log_name_cqt + ".pl")
np.save(resource_path + "\\files\\" + files_path + ".npy", files)
'''
scaled_feature_vectors = pickle.load(open(resource_path + "\\feature_vectors\\" + log_name_cqt + ".pl", "rb"))
#for cqt omit otherwise 
print(scaled_feature_vectors.shape)
#scaled_feature_vectors = scaled_feature_vectors.reshape(len(scaled_feature_vectors), n_bins)


model = load_model("C:/Users/deeon/OneDrive/Documents/Demo_diss/models/CNN/CQTweights.200-0.39.h5")

predictions = model.predict(scaled_feature_vectors, verbose=1)

predictions_round=np.around(predictions).astype('int')
predictions_int = np.argmax(predictions_round, axis=1)
predicted_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

#test_round=np.around(test_classes).astype('int')
#test_int = np.argmax(predictions_round, axis=1)
#test_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

EADcount=0
DADcount = 0 
DAEcount = 0 
DGDcount = 0

DADlen = 153 
EADlen = 107
DAElen = 127 
DGDlen = 103

for i in range(len(predicted_labels)): 
	if predicted_labels[i] == labels[i]: 
		if labels[i] == "EADGBE": 
			EADcount += 1 
		elif labels[i] == "DADGBE": 
			DADcount += 1 
		elif labels[i] == "DGDGBD": 
			DGDcount += 1 
		else: 
			DAEcount += 1 

print(str(DGDcount) + "/" + str(DGDlen) + " DGDGBD Predictions Correct")
print(str(EADcount) + "/" + str(EADlen) + " EADGBE Predictions Correct")
print(str(DADcount) + "/" + str(DADlen) + " DADGBE Predictions Correct")
print(str(DAEcount) + "/" + str(DAElen) + " DAEGAD Predictions Correct")




