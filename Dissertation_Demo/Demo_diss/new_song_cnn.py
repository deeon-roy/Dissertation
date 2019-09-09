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

mycwd = os.getcwd() 

model = load_model("C:/Users/deeon/OneDrive/Documents/Project/models/CNN/weights.200-0.04.h5")

resource_path = mycwd + "\\resources"
pp.split_folder(folder="test_songs", audio_path=mycwd + "\\audio\\test")

cf = "classes_windowed.npy"
lf = "labels_windowed.npy"
labels = np.load(resource_path + "\\labels\\" + lf)
labelencoder = LabelEncoder() 
labelencoder.classes_ = np.load(resource_path + "\\labels\\" + cf)
classes = labelencoder.transform(labels)
pp.one_hot(classes, "onehotlabels.npy")
onehotlabels = np.load(resource_path+"\\labels\\onehotlabels.npy")

#CREATE VECTORS
feature_vectors, files = pp.get_cqt_folder(path=mycwd+"\\audio\\test\\windowed\\other")
pp.save_cqt(feature_vectors, "test_songs" + ".pl")
np.save(resource_path + "\\files\\" + "test_songs" + ".npy", files)
test_set = pickle.load(open(resource_path + "\\feature_vectors\\MFCC_DEFAULT_PARAMS_DL.pl", "rb"))



predictions = model.predict(test_set, verbose=1)
predictions_round=np.around(predictions).astype('int')
predictions_int = np.argmax(predictions_round, axis=1)
predicted_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

#test_round=np.around(test_classes).astype('int')
#test_int = np.argmax(predictions_round, axis=1)
#test_labels= labelencoder.inverse_transform(np.ravel(predictions_int))
count = 0
for i in predicted_labels: 
	if i == "EADGBE": 
		count += 1 
	print(i)

print(count/len(predicted_labels))

'''
plt.figure(figsize=(18, 13))
evaluation.plot_confusion_matrix(predicted_labels, labelencoder.classes_, test_labels)
plt.savefig(plot_path + "\\DNN\\" + log_name_cqt +".png")
plt.close()
evaluation.plot_history(hist)
plt.savefig(plot_path + "\\DNN\\" + log_name_cqt + "_hist" +".png")
plt.close()
wp = evaluation.wrong_predictions(predicted_labels, test_labels)
'''
files = np.load(resource_path + "\\files\\" + "files_win.npy")
