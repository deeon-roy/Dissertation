#FrameWork Library Imports 
import ml_gt.preprocessing as pp
import ml_gt.learning as learning
import ml_gt.evaluation as evaluation 

#System 
import os, pickle

#Audio
import librosa

#ML
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
audio_path = mycwd + "\\audio\\Windowed"
song_path = mycwd + "\\songs"
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


cf = "classes_windowed.npy"
lf = "labels_windowed.npy"
log_name_cqt = "cqt_win_{0}_{1}_{2}".format(hop_length, fminval, n_bins)
files_path = "files_win"


model = load_model('C:/users/deeon/onedrive/documents/project/models/dnn/weights.99-0.01' + '.h5')

pp.create_labels(audio_path=audio_path, output_classes=cf, output_labels=lf)
labels = np.load(resource_path + "\\labels\\" + lf)
labelencoder = LabelEncoder() 
labelencoder.classes_ = np.load(resource_path + "\\labels\\" + cf)
classes = labelencoder.transform(labels)
pp.one_hot(classes, "onehotlabels.npy")
onehotlabels = np.load(resource_path+"\\labels\\onehotlabels.npy")
print(labels.shape)
print(classes.shape)

scaled_feature_vectors = pickle.load(open(resource_path + "\\feature_vectors\\CQT_SK\\" + "CQT_SK_VECTORS_44100_28_37_512.pl", "rb"))
#for cqt omit otherwise 
scaled_feature_vectors = scaled_feature_vectors.reshape(len(scaled_feature_vectors), 37)
print(scaled_feature_vectors.shape)
train_set, test_set, train_classes, test_classes, test_classes = pp.split_training_set(onehotlabels, scaled_feature_vectors)

test_predictions = model.predict(test_set)

predictions_round=np.around(test_predictions).astype('int')
predictions_int = np.argmax(predictions_round, axis=1)
predicted_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

test_round=np.around(test_classes).astype('int')
test_int = np.argmax(predictions_round, axis=1)
test_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

plt.figure(figsize=(18, 13))
evaluation.plot_confusion_matrix(predicted_labels, labelencoder.classes_, test_labels)
plt.savefig(plot_path + "\\DNN\\" + log_name_cqt[:-3] +".png")
wp = evaluation.wrong_predictions(predicted_labels, test_classes)
