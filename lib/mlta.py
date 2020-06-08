import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append('../camnet/')
from models import *
import keras
from models import getModel, standardPreprocess
from functions import parseTrainingOptions, parseLoadOptions
import h5py as hd
from tflearn.data_utils import shuffle, to_categorical
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import sys
from keras.engine import Layer
import keras.backend as K
import numpy as np
import sklearn.metrics
import shutil
import warnings
warnings.filterwarnings("ignore")
"""
Credits for the Gradient Reversal Layer
https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
"""
def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    hp_lambda = hp_lambda
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1
    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        grad = tf.negative(grad)
        final_val = grad * hp_lambda
        return [final_val]
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        #self.hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda, name='hp_lambda')
    def build(self, input_shape):
        self.trainable_weights = []
        return
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  'hp_lambda': K.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def my_acc_f(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where)#indices where the item of y_true is -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)

    n1 = tf.shape(indices)[0]
    batch_size = tf.shape(y_true)[0]
    n2 = batch_size - n1
    sliced_y_true = tf.reshape(sliced_y_true, [n1, -1])
    # the activation function is here.... but this means that I should
    # apply this activation function to the logits also when I run the
    # regression
    sliced_y_pred = tf.sigmoid(sliced_y_pred)

    y_pred_rounded = K.round(sliced_y_pred)
    acc = tf.equal(y_pred_rounded, sliced_y_true)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc
def my_accuracy(y_true, y_pred):
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    acc = tf.equal(y_pred_rounded, y_true)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc

def classifier_loss(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    n1 = tf.shape(indices)[0]
    batch_size = tf.shape(y_true)[0]
    n2 = batch_size - n1
    sliced_y_true = tf.reshape(sliced_y_true, [n1, -1])
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=sliced_y_true,logits=sliced_y_pred)
    n1_ = tf.cast(n1, tf.float32)
    n2_ = tf.cast(n2, tf.float32)
    multiplier = (n1_+ n2_) / n1_
    zero_class = tf.constant(0, dtype=tf.float32)
    where_class_is_zero=tf.cast(tf.equal(sliced_y_true, zero_class), dtype=tf.float32)
    class_weight_zero = tf.cast(tf.divide(n1_, 2. * tf.cast(tf.reduce_sum(where_class_is_zero), dtype=tf.float32)+0.001), dtype=tf.float32)
    one_class = tf.constant(1, dtype=tf.float32)
    where_class_is_one=tf.cast(tf.equal(sliced_y_true, one_class), dtype=tf.float32)
    class_weight_one = tf.cast(tf.divide(n1_, 2. * tf.cast(tf.reduce_sum(where_class_is_one),dtype=tf.float32)+0.001), dtype=tf.float32)
    #class_weight_vector = [1-ytrue]*[cwzero] + [ytrue][cwone] = A + B
    A = tf.ones(tf.shape(sliced_y_true), dtype=tf.float32) - sliced_y_true 
    A = tf.scalar_mul(class_weight_zero, A)
    B = tf.scalar_mul(class_weight_one, sliced_y_true) 
    class_weight_vector=A+B
    ce = class_weight_vector * ce
    ce = tf.scalar_mul(multiplier,ce)
    zero_tensor = tf.zeros([n2, 1])
    final_ce = tf.cond(n2>0,
                       lambda: tf.concat([ce, zero_tensor], axis=0),
                       lambda: tf.scalar_mul(multiplier,ce))
    return tf.reduce_mean(ce)

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_keys(l):
    db_name=l.split(', ')[0]
    entry_path=l.split(', ')[1]
    patch_no=l.split(', ')[2]
    return db_name, entry_path, int(patch_no)
def get_class(l, entry_path):
    if l.split(', ')[-1].strip('\n')=='test':
        return -1.
    if l.split(', ')[-1].strip('\n')=='test2':
        return -1
    elif l.split(', ')[-1].strip('\n')=='validation':
        return -1
    elif l.split(', ')[-1].strip('\n')=='train':
        if entry_path.split('/level7')[0]=='normal':
            return 0.
        else:
            return 1.
    else:
        print 'I should not enter here, really'
        if entry_path.split('/level7')[0]=='normal':
            return 0.
        else:
            return 1.
def get_test_label(entry_path):
    if entry_path.split('/level7')[0]=='normal':
        return 0.
    else:
        return 1.


def get_domain(db_name, entry_path):
    if db_name=='cam16':
        return 5
    else:
        center_no = entry_path.split('/centre')[1].split('/patient')[0]
        return int(center_no)
    
def zero_loss(y_pred, y_true):
    zero_constant = tf.constant(0, dtype=tf.float32)
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    
    return tf.scalar_mul(zero_constant, mse)
    
def compute_acc(y_pred, y_true):
    #should these be floats?
    y_pred = tf.sigmoid(y_pred)
    y_pred = tf.round(y_pred)
    y_pred = tf.Print(y_pred, [y_pred], 'y_pred: ')
    acc = tf.equal(y_pred, y_true)
    acc = tf.Print(acc, [acc], 'acc equal: ')
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc
