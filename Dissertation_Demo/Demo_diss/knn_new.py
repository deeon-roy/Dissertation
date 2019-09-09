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
from sklearn.decomposition import PCA 

#General 
import numpy as np 

#Plotting 
import seaborn 
import matplotlib.pyplot as plt 


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

#Window Params 

cf = "classes_windowed.npy"
lf = "labels_windowed.npy"
log_name_cqt = "cqt_win_knn_{0}_{1}_{2}".format(hop_length, fminval, n_bins)
files_path = "files_win"


pp.create_labels(audio_path=audio_path, output_classes=cf, output_labels=lf)
labels = np.load(resource_path + "\\labels\\" + lf)
labelencoder = LabelEncoder() 
labelencoder.classes_ = np.load(resource_path + "\\labels\\" + cf)
classes = labelencoder.transform(labels)
print(labels.shape)
print(classes.shape)


#CREATE VECTORS
feature_vectors, files = pp.get_cqt_folder(path=audio_path)
pp.save_cqt_sk(feature_vectors, log_name_cqt + ".pl")
np.save(resource_path + "\\files\\" + files_path + ".npy", files)

scaled_feature_vectors = pickle.load(open(resource_path + "\\feature_vectors\\CQT_SK\\" + log_name_cqt +".pl", "rb"))
#for cqt omit otherwise 
scaled_feature_vectors = scaled_feature_vectors.reshape(len(scaled_feature_vectors), n_bins)
print(scaled_feature_vectors.shape)
train_set, test_set, train_classes, test_classes, test_index = pp.split_training_set(classes, scaled_feature_vectors)


model = learning.kNN(train_set, train_classes)
learning.save_model(model, model_path+"\\kNN\\kNN_grid_windowed")

predicted_labels = learning.validate(model, test_set, test_classes)

plt.figure(figsize=(18, 13))
evaluation.plot_confusion_matrix(predicted_labels, labelencoder.classes_, test_classes)
plt.savefig(plot_path + "\\kNN\\" + log_name_cqt +".png")
wp = evaluation.wrong_predictions(predicted_labels, test_classes)

files = np.load(resource_path + "\\files\\" + "files_win.npy")

with open(log_path + "\\kNN\\" + log_name_cqt + ".txt", "w") as f: 
	f.write(str(np.array(labels)[test_index[wp]]) + "\n")
	f.write(str(predicted_labels[wp].T)+ "\n")
	f.write(str(labelencoder.inverse_transform(predicted_labels[wp]))+ "\n")
	f.write(str(np.array(files)[test_index[wp]])+ "\n")
	f.write("#General Parameters \n sr = {0} \nhop_length = {1}\n #MFCC Parameters \nn_fft = {2}\n n_mels = {3}\n n_mfcc = {4} \n#CQT Parameters\n fmin = {5}\n n_bins = {6}".format(sr, hop_length, n_fft, n_mels, n_mfcc, fminval, n_bins))

