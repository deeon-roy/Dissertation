import numpy as np 
import pickle 
import itertools 

#System
import os, fnmatch 

#data 
import pandas as pd 

#Visualisation 
import seaborn
import matplotlib.pyplot as plt 
from IPython.core.display import HTML, display, Image 

#ML 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC 
from sklearn.externals import joblib
from sklearn.decomposition import PCA 

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
from keras.utils import plot_model

# Random Seed
from tensorflow import set_random_seed
from numpy.random import seed
seed(0)
set_random_seed(0)

# Audio
import librosa.display, librosa
from librosa.util import normalize as normalise 
import IPython.display as ipd 

def plot_history(hist): 
	loss_list = [s for s in hist.history.keys() if 'loss' in s and 'val' not in s]
	val_loss_list = [s for s in hist.history.keys() if "loss" in s and "val" in s]
	acc_list = [s for s in hist.history.keys() if "acc" in s and "val" in s]

	if len(loss_list) == 0: 
		print("Loss is missing")
		return 

	epochs = range(1, len(hist.history[loss_list[0]]) + 1 )

	#loss
	plt.figure(1)
	for l in loss_list: 
		plt.plot(epochs, hist.history[l], 'r', label="Training Loss")
	for l in val_loss_list: 
		plt.plot(epochs, hist,history[l], 'b', label="Validation Loss")

	plt.title('Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show() 

def plot_confusion_matrix(predicted_labels, 
						  classes,
						  test_classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	
	cm = confusion_matrix(test_classes, predicted_labels)
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	"""
	#print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def wrong_predictions(predicted_labels, test_classes): 
	wrong_predictions = [i for i, (e1, e2) in enumerate(zip(predicted_labels, test_classes)) if e1 != e2]

	return wrong_predictions






