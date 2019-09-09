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

print(device_lib.list_local_devices())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)

