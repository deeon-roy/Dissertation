import ml_gt.preprocessing as pp

import os, itertools

mycwd = os.getcwd() 

folder = mycwd + "\\audio\\Windowed"

sr = [44100] 
n_fft = [2048]
n_mfcc = [20, 13]
n_mels = [128, 64, 256]
fminval = [36, 24, 28]
n_bins = [72, 88, 37]
hop_length = [512, 1024]



for x in itertools.product(sr, fminval, n_bins, hop_length): 
	filename = "CQT_SK_VECTORS_{0}_{1}_{2}_{3}".format(x[0], x[1], x[2], x[3])
	print(x[0], x[1], x[2], x[3])
	if not filename == "CQT_SK_VECTORS_44100_36_72_512":
		pp.create_vectors(type="cqt", folder=folder, filename=filename, sr=x[0], fminval=x[1], n_bins=x[2],hop_length=x[3])
for x in itertools.product(sr, fminval, n_bins, hop_length): 
	filename = "CQT_DL_VECTORS_{0}_{1}_{2}_{3}".format(x[0], x[1], x[2], x[3])
	pp.create_vectors(type="cqt", folder=folder, filename=filename, sr=x[0], fminval=x[1], n_bins=x[2],hop_length=x[3], cnn=True)
for x in itertools.product(sr, n_fft, hop_length, n_mels, n_mfcc): 
	filename = "MFCC_SK_VECTORS_{0}_{1}_{2}_{3}_{4}".format(x[0], x[1], x[2], x[3], x[4])
	pp.create_vectors(type="mfcc", folder=folder, filename=filename, sr=x[0], n_fft=x[1], hop_length=x[2], n_mels=x[3], n_mfcc=x[4])
for x in itertools.product(sr, n_fft, hop_length, n_mels, n_mfcc): 
	filename = "MFCC_DL_VECTORS_{0}_{1}_{2}_{3}_{4}".format(x[0], x[1], x[2], x[3], x[4])
	pp.create_vectors(type="mfcc", folder=folder, filename=filename, sr=x[0], n_fft=x[1], hop_length=x[2], n_mels=x[3], n_mfcc=x[4], cnn=True)

