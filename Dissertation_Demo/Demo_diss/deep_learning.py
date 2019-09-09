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

def create_model(fc_layers=[4], activation="relu", optimizer='rmsdrop'): 
	model = Sequential() 

	for i, size in enumerate(fc_layers): 
		print(size)
		if i == 0: 
			model.add(Dense(size, activation=activation, input_shape=(train_set.shape[1],)))
		else: 
			model.add(Dense(size, activation=activation))
	model.add(Dense(4, activation='softmax'))
	model.compile(optimizer= optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
	return model

cf = "classes_windowed.npy"
lf = "labels_windowed.npy"
log_name_cqt = "cqt_win_{0}_{1}_{2}".format(hop_length, fminval, n_bins)
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

'''
#CREATE VECTORS
feature_vectors, files = pp.get_cqt_folder(path=audio_path)
pp.save_cqt_sk(feature_vectors, log_name_cqt + ".pl")
np.save(resource_path + "\\files\\" + files_path + ".npy", files)
'''

for root, dirnames, filenames in os.walk(resource_path + "\\feature_vectors\\MFCC_SK"): 
	for name in filenames:
	 
		n_bins = int(name[24:26])
		log_name_cqt = name
		scaled_feature_vectors = pickle.load(open(resource_path + "\\feature_vectors\\MFCC_SK\\" + name, "rb"))
		#for cqt omit otherwise 
		#scaled_feature_vectors = scaled_feature_vectors.reshape(len(scaled_feature_vectors), n_bins)
		print(scaled_feature_vectors.shape)
		train_set, test_set, train_classes, test_classes, test_classes = pp.split_training_set(onehotlabels, scaled_feature_vectors)
		print(train_set.shape, test_set.shape, train_classes.shape, test_classes.shape)


		model = create_model(fc_layers=[12], activation="relu", optimizer="adam")

		hist = History() 


		#es = EarlyStopping(monitor='loss', min_delta=0.01, restore_best_weights=True, patience=10, verbose=1)
		mc = ModelCheckpoint(model_path + '\\DNN\\weights.{epoch:02d}-{loss:.2f}.h5', monitor='loss', save_best_only=False, verbose=1, period=1)

		KerasCallback = [hist, mc]

		model.fit(train_set, train_classes, batch_size=5, epochs=150, callbacks=KerasCallback, verbose=1)
										
		predictions = model.predict(test_set, verbose=1)
		predictions_round=np.around(predictions).astype('int')
		predictions_int = np.argmax(predictions_round, axis=1)
		predicted_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

		test_round=np.around(test_classes).astype('int')
		test_int = np.argmax(predictions_round, axis=1)
		test_labels= labelencoder.inverse_transform(np.ravel(predictions_int))

		plt.figure(figsize=(18, 13))
		evaluation.plot_confusion_matrix(predicted_labels, labelencoder.classes_, test_labels)
		plt.savefig(plot_path + "\\DNN\\" + log_name_cqt +".png")
		plt.close()
		evaluation.plot_history(hist)
		plt.savefig(plot_path + "\\DNN\\" + log_name_cqt + "_hist" +".png")
		plt.close()
		wp = evaluation.wrong_predictions(predicted_labels, test_labels)

		files = np.load(resource_path + "\\files\\" + "files_win.npy")

		with open(log_path + "\\DNN\\" + log_name_cqt[:-3] + ".txt", "w") as f: 
			f.write(str(np.array(labels)[test_classes[wp]]) + "\n")
			f.write(str(predicted_labels[wp].T)+ "\n")
			f.write(str(labelencoder.inverse_transform(predicted_labels[wp]))+ "\n")
			f.write(str(np.array(files)[test_classes[wp]])+ "\n")
			f.write("#General Parameters \n sr = {0} \nhop_length = {1}\n #MFCC Parameters \nn_fft = {2}\n n_mels = {3}\n n_mfcc = {4} \n#CQT Parameters\n fmin = {5}\n n_bins = {6}".format(sr, hop_length, n_fft, n_mels, n_mfcc, fminval, n_bins))







'''
# One Hot encode
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded_train_classes =  train_classes.reshape(len( train_classes), 1)
onehot_encoded_train_classes = onehot_encoder.fit_transform(integer_encoded_train_classes,1)
integer_encoded_test_classes =  test_classes.reshape(len( test_classes),1)
onehot_encoded_test_classes = onehot_encoder.fit_transform(integer_encoded_test_classes,1)
'''


#model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=5, verbose=1)

'''
param_grid = {'fc_layers': [[12],[12,6],[6,6,6], [6]],
			  'activation':['relu','tanh'],
			  'optimizer':('rmsprop','adam'),
			  'epochs':[100, 150],
			  'batch_size':[5]}

grid = GridSearchCV(model, param_grid=param_grid, return_train_score=True, cv=5)

grid_results = grid.fit(train_set, train_classes, verbose=1)
print("best parameters are: ")
print(grid_results.best_params_)

print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#categorical_encoded_train_classes = to_categorical()
'''
