#System
import pkg_resources
import os, fnmatch, pickle  

#Audio
from pydub import AudioSegment 
from pydub.utils import make_chunks
import librosa
from librosa.util import normalize as normalise
from scipy.io import wavfile 

#Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.decomposition import PCA 

#Other
import numpy as np

#ML
from keras import backend as K

mycwd = os.getcwd() 
resource_path = mycwd + "\\resources"
audio_path = mycwd + "\\audio"
song_path = mycwd + "\\songs"
 
 
if os.path.exists(resource_path + "\\labels\\labels.npy"): 
	labels = np.load(resource_path + "\\labels\\labels.npy")
else: 
	print("No Labels found in {0}, please create some labels first.".format(resource_path + "\\labels\\labels.npy"))
	labels = None 

def check_dir(label, audio_path=audio_path):
	if not os.path.exists(audio_path + "\\%s" % label): 
		os.makedirs(audio_path + "\\%s" % label)
		yield 0
	elif not os.listdir(audio_path + "\\%s" % label): 
		yield 0
	else: 
		for filename in os.listdir(audio_path + "\\%s" % label): 
			name, _ = os.path.splitext(filename) 
			yield int(name[7:])


def split_audio(labels=labels, chunk_length=2000, file=None, audio_path=audio_path): 
	if not file: 
		filename = song_path + "\\" + input("Which file would you like to split: ") + ".wav"
	else: 
		filename = file 
		for name in labels: 
			if fnmatch.fnmatchcase(filename, '*'+name+'*'): 
				label = name 
				break 
		else: 
			print("Label not Found. Please add label aptly. saved as other")
			label = "other"
		audio = AudioSegment.from_file(filename, "wav")
		chunks = make_chunks(audio, chunk_length)
		count = max(check_dir(label))
		print(count)
		for i, chunk in enumerate(chunks): 
			chunk_name = label + "_{0}.wav".format(i+count)
			print("exporting", chunk_name)
			file_string = str(audio_path + "\\{0}\\" + chunk_name).format(label)
			chunk.export(file_string, format="wav")

def split_audio_windowed(labels=labels, chunk_length=4, file=None, window=3, audio_path=audio_path): 
	if not file: 
		filename = song_path + "\\" + input("Which file would you like to split: ") + ".wav"
	else: 
		filename = file 
		for name in labels: 
			if fnmatch.fnmatchcase(filename, '*'+name+'*'): 
				label = name 
				break 
		else: 
			print("Label not Found. Please add label aptly. saved as other")
			label = "other"

		fs, y = wavfile.read(filename)
		print("Y LENGTH")
		print(len(y))
		slices = range(0, len(y), chunk_length*fs-window*fs)
		count = max(check_dir(label, audio_path=audio_path + "\\Windowed"))
		print(count)
		for i in slices: 
			chunk_name = label + "_{0}.wav".format(int((i/fs)+count))
			print("exporting", chunk_name)
			file_string = str(audio_path + "\\windowed\\{0}\\" + chunk_name).format(label)
			start_audio = i
			end_audio = i+window
			wavfile.write(file_string, fs, y[i:i+chunk_length*fs])

def split_folder(labels=labels, chunk_length=2000, folder=None, windowed=True, audio_path=audio_path):
	if not folder: 	
		folder = mycwd + "\\" + input("which folder would you like to split: ")
	if windowed:
		for filename in os.listdir(folder): 
			print(filename)
			split_audio_windowed(labels=labels, file=folder + "\\" + filename, audio_path=audio_path)
	else:
		for filename in os.listdir(folder): 
			print(filename)
			split_audio(labels=labels, file=folder + "\\" + filename)

def get_mfcc_features(y, sr=44100, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13):
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
	mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
	feature_vector = np.mean(mfcc,1)
	#feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
	return feature_vector

def get_mfcc_folder(sr=44100, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13, path=audio_path, cnn=False, duration=4):
	files = []
	feature_vectors = []
	for root, dirnames, filenames in os.walk(path): 
		for filename in fnmatch.filter(filenames, "*.wav"): 
			files.append(os.path.join(root, filename))
	print("found %d audio files in %s"%(len(files), path))
	for i, f in enumerate(files): 
		print ("get %d of %d = %s"%(i+1, len(files), f))
		try:
			y, sr = librosa.load(f, sr=sr)
			y/=y.max() #Normalize
			if len(y) < duration:
				print("Error loading %s" % f)
				continue
			if cnn:
				feat = get_mfcc_dl(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, duration=duration)
			else:
				feat = get_mfcc_features(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
			feature_vectors.append(feat)
		except Exception as e:
			print("Error loading %s. Error: %s" % (f,e))
	return feature_vectors, files


def get_mfcc_dl(y, sr=44100, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13, duration=4):
	y = normalise(y)
	duration_in_samples = librosa.time_to_samples(duration, sr=sr)
	y_pad = librosa.util.fix_length(y, duration_in_samples)
	S = librosa.feature.melspectrogram(y_pad, sr=sr, n_mels=n_mels)
	mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
	#feature_vector = np.mean(mfcc,1)
	#print(mfcc.shape)
	return mfcc

def save_mfcc(feature_vectors, filename, resource_path=resource_path): 
	scaler = StandardScaler() 
	scaled_feature_vectors = scaler.fit_transform(np.array(feature_vectors))
	with open(resource_path + "\\feature_vectors\\" + filename, "wb") as f: 
		pickle.dump(scaled_feature_vectors, f)

def save_mfcc_dl(feature_vectors, filename, resource_path=resource_path):
	scaled_feature_vectors = []  
	print(np.array(feature_vectors).shape)
	scaler = StandardScaler() 
	dtype = K.floatx()
	for i in feature_vectors: 
		data = scaler.fit_transform(np.array(i))
		data = np.expand_dims(data, axis=3)
		scaled_feature_vectors.append(data)
	with open(resource_path + "\\feature_vectors\\" + filename, "wb") as f: 
		pickle.dump(np.array(scaled_feature_vectors), f)

def get_cqt_features(y, sr=44100, fmin=librosa.midi_to_hz(29), n_bins=88, hop_length=512, duration=4): 
	y, index = librosa.effects.trim(y, top_db=60)
	y = normalise(y)
	duration_in_samples = librosa.time_to_samples(duration, sr=sr)
	y_pad = librosa.util.fix_length(y, duration_in_samples)
	y_cqt = librosa.cqt(y_pad, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
	y_spec = librosa.amplitude_to_db(abs(y_cqt), np.max)
	return y_spec 

def get_cqt_folder(sr=44100, fmin=librosa.midi_to_hz(36), n_bins=72, hop_length=512, duration=2, path=audio_path): 
	files = []
	feature_vectors = []
	for root, dirnames, filenames in os.walk(path): 
		for filename in fnmatch.filter(filenames, "*.wav"): 
			files.append(os.path.join(root, filename))
	print("found %d audio files in %s"%(len(files), path))
	for i, f in enumerate(files): 
		print ("get %d of %d = %s"%(i+1, len(files), f))
		try:
			y, sr = librosa.load(f, sr=sr)
			y/=y.max() #Normalize
			if len(y) < duration:
				print("Error loading %s" % f)
				continue
			feat = get_cqt_features(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length, duration=duration)
			feature_vectors.append(feat)
		except Exception as e:
			print(type(f), type(y), type(duration), type(fmin), type(hop_length), type(n_bins), type(sr))
			print("Error loading %s. Error: %s" % (f,e))
	return feature_vectors, files

def save_cqt(feature_vectors, filename): 
	scaled_feature_vectors = []
	scaler = StandardScaler() 
	dtype = K.floatx()
	for y_spec in feature_vectors:
		data = scaler.fit_transform(y_spec).astype(dtype)
		#data = np.expand_dims(data, axis=0)
		data = np.expand_dims(data, axis=3)
		scaled_feature_vectors.append(data)
	with open(resource_path + "\\feature_vectors\\" + filename, "wb") as f: 
		pickle.dump(np.array(scaled_feature_vectors), f)

def save_cqt_sk(feature_vectors, filename, resource_path=resource_path): 
	scaled_feature_vectors = []
	pca = PCA(n_components=1)
	scaler = StandardScaler() 
	dtype = K.floatx()
	for y_spec in feature_vectors:
		data = scaler.fit_transform(y_spec).astype(dtype)
		data = pca.fit_transform(data).astype(dtype)
		scaled_feature_vectors.append(data)
	scaled_feature_vectors = np.array(scaled_feature_vectors)
	scaled_feature_vectors.reshape(len(scaled_feature_vectors), scaled_feature_vectors.shape[1])
	print(scaled_feature_vectors.shape)
	with open(resource_path + "\\feature_vectors\\" + filename, "wb") as f: 
		pickle.dump(np.array(scaled_feature_vectors), f)

def create_labels(audio_path=audio_path, classes=['EADGBE', 'DGDGBD', 'DAEGAD', 'DADGBE'], resource_path=resource_path, output_classes="classes.npy", output_labels="labels.npy"): 
	files = [] 
	for root, dirnames, filenames in os.walk(audio_path):
		for filename in fnmatch.filter(filenames, "*.wav"):
			files.append(os.path.join(root, filename))

	print("found %d audio files in %s" % (len(files), audio_path))
	labels = [] 
	for filename in files: 
		for name in classes: 
			if fnmatch.fnmatchcase(filename, "*"+name+"*"): 
				labels.append(name)
				break 
		else: 
			labels.append("other")

	labelencoder = LabelEncoder() 
	labelencoder.fit(labels)
	np.save(resource_path+"\\labels\\"+output_classes, labelencoder.classes_)
	np.save(resource_path+"\\labels\\"+output_labels, labels)

def one_hot(classes, output_labels): 
	encoder = OneHotEncoder(sparse=False, categories="auto")
	onehot_labels=encoder.fit_transform(classes.reshape(len(classes), 1))
	np.save(resource_path+"\\labels\\"+output_labels, onehot_labels)

def split_training_set(labels, scaled_feature_vectors, n_splits=1, test_size=0.25, random_state=0): 
	splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
	splits = splitter.split(scaled_feature_vectors, labels)
	for train_index, test_index in splits: 
		train_set = scaled_feature_vectors[train_index]
		test_set = scaled_feature_vectors[test_index]
		train_classes = labels[train_index]
		test_classes = labels[test_index]

	return train_set, test_set, train_classes, test_classes, test_index

def create_vectors(type, folder, filename, sr=44100, n_fft=None, n_mels=None, n_mfcc=None, fminval=36, n_bins=72, hop_length=512, cnn=False, audio_path=audio_path): 
	if not cnn: 
		if type=="cqt": 
			feature_vectors, files = get_cqt_folder(sr=sr, fmin=librosa.midi_to_hz(fminval), n_bins=n_bins, hop_length=hop_length, duration=4, path=folder) 
			save_cqt_sk(feature_vectors, filename + ".pl")
		elif type=="mfcc": 
			feature_vectors, files = get_mfcc_folder(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, path=audio_path, duration=4)
			save_mfcc(feature_vectors, filename)
	else: 
		if type=="cqt": 
			feature_vectors, files = get_cqt_folder(sr=sr, fmin=librosa.midi_to_hz(fminval), n_bins=n_bins, hop_length=hop_length, duration=4, path=folder) 
			save_cqt(feature_vectors, filename + ".pl")
		elif type=="mfcc": 
			feature_vectors, files = get_mfcc_folder(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, path=audio_path, cnn=True)
			save_mfcc(feature_vectors, filename)

