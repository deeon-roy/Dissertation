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
from sklearn.metrics import classification_report

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

def create_model(fc_layers=[16], activation="sigmoid", activation_1="softmax", optimizer='sgd', loss="categorical_crossentropy"): 
	model = Sequential() 
	print(test_set.shape)
	for i, size in enumerate(fc_layers): 
		if i == 0: 
			model.add(Convolution2D(size, 3, activation=activation, input_shape=(train_set.shape[1],train_set.shape[2],train_set.shape[3])))
			#model.add(Convolution2D(size, 3, activation=activation, input_shape=(train_set.shape[1], train_set.shape[2])))
			model.add(MaxPooling2D(pool_size=(2, 2)))
		else: 
			model.add(Convolution2D(size, 3, activation=activation))
			model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())

	model.add(Dense(fc_layers[0], activation=activation))
	model.add(Dense(4, activation=activation_1))
	model.compile(optimizer= optimizer, loss=loss, metrics=['accuracy'])
	return model

cf = "classes_windowed.npy"
lf = "labels_windowed.npy"
#log_name_cqt = "MFCC_DEFAULT_PARAMS_DL".format(hop_length, fminval, n_bins)
log_name_cqt =  "cqt_CNN_win_knn_{0}_{1}_{2}".format(hop_length, fminval, n_bins)
files_path = "files_win"


pp.create_labels(audio_path=audio_path, output_classes=cf, output_labels=lf)
labels = np.load(resource_path + "\\labels\\" + lf)
labelencoder = LabelEncoder() 
labelencoder.classes_ = np.load(resource_path + "\\labels\\" + cf)
classes = labelencoder.transform(labels)
pp.one_hot(classes, "onehotlabels.npy")
onehotlabels = np.load(resource_path+"\\labels\\onehotlabels.npy")
print(labels.shape)
print(classes.shape)
print(onehotlabels.shape)

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
print(scaled_feature_vectors.shape)
train_set, test_set, train_classes, test_classes, test_index = pp.split_training_set(onehotlabels, scaled_feature_vectors)

model = create_model(fc_layers=[12, 12], activation="sigmoid", activation_1="softmax", optimizer="sgd")

hist = History() 


#es = EarlyStopping(monitor='loss', min_delta=0.01, restore_best_weights=True, patience=10, verbose=1)
mc = ModelCheckpoint(model_path + '\\CNN\\CQTweights.{epoch:02d}-{loss:.2f}.h5', monitor='loss', save_best_only=False, verbose=1, period=1)

KerasCallback = [hist, mc]

model.fit(train_set, train_classes, batch_size=50, epochs=200, callbacks=KerasCallback, verbose=1)


predictions = model.predict(test_set, verbose=1)
predictions_round=np.around(predictions).astype('int')
predictions_int = np.argmax(predictions_round, axis=1)
predicted_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

test_round=np.around(test_classes).astype('int')
test_int = np.argmax(predictions_round, axis=1)
test_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

print(classification_report(test_labels, predicted_labels))

plt.figure(figsize=(18, 13))
evaluation.plot_confusion_matrix(predicted_labels, labelencoder.classes_, test_labels)
plt.savefig(plot_path + "\\CNN\\" + log_name_cqt +".png")
plt.close()
evaluation.plot_history(hist)
plt.savefig(plot_path + "\\CNN\\" + log_name_cqt + "_hist" +".png")
plt.close()
wp = evaluation.wrong_predictions(predicted_labels, test_labels)

files = np.load(resource_path + "\\files\\" + "files_win.npy")

with open(log_path + "\\CNN\\" + log_name_cqt + ".txt", "w") as f: 
	f.write(str(np.array(labels)[test_index[wp]]) + "\n")
	f.write(str(predicted_labels[wp].T)+ "\n")
	f.write(str(labelencoder.inverse_transform(predicted_labels[wp]))+ "\n")
	f.write(str(np.array(files)[test_index[wp]])+ "\n")
	f.write("#General Parameters \n sr = {0} \nhop_length = {1}\n #MFCC Parameters \nn_fft = {2}\n n_mels = {3}\n n_mfcc = {4} \n#CQT Parameters\n fmin = {5}\n n_bins = {6}".format(sr, hop_length, n_fft, n_mels, n_mfcc, fminval, n_bins))

