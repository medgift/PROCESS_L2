import sys
sys.path.append('../camnet/')
sys.path.append('../rcv-tool/')
import rcv
#rcv.get_all_color_measures('hello')
from models import *
from functions import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import cv2
import os
from os import listdir
from os.path import join, isfile, exists, splitext
from random import shuffle
import numpy as n
from PIL import Image
#from extract_xml import get_opencv_contours_from_xml
from skimage.transform.integral import integral_image, integrate
from skimage.viewer import ImageViewer
import skimage
from skimage import io
from util import otsu_thresholding
from extract_xml import *
from functions import *
from integral import patch_sampling_using_integral
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tflearn.data_utils import shuffle
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from models import *
import time
import logging
import h5py as hd
import shutil
from datasets import Dataset 
import warnings
warnings.simplefilter("ignore")
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
print("Using Tensorflow {}".format(tf.__version__))
import keras
print("Using Keras {}".format(keras.__version__))
from keras.engine import Layer
import keras.backend as K
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sys
sys.path.append('../rcv-tool/')
import rcv