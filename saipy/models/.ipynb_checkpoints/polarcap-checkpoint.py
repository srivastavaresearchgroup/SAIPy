import h5py
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import sys

from math import sqrt
from random import randint
from tqdm import tqdm

from keras import backend as K
from keras import layers
from keras.layers import Input, LSTM
from keras.models import Model

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, '..')

#____________________________________________

def norm(X):
    maxi = np.max(abs(X), axis = 1)
    X_ret = X.copy()
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret
        
class PolarCAP:
    def __init__(self, path):
        if not os.path.exists(path):
            path = './saipy/saved_models/' 
        self.model = keras.models.load_model(path+'PolarCAP.h5')
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def get_model(self, untrained = False):
        if untrained:
            drop_rate = 0.3
            lr = 1e-3

            X = Input(shape = (64,1))
            x = layers.Conv1D(32, 32, activation = 'relu', padding = 'same')(X)
            x = layers.Dropout(drop_rate)(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2, padding='same')(x)
            x = layers.Conv1D(8, 16, activation = 'relu', padding = 'same')(x)
            x = layers.BatchNormalization()(x)
            enc = layers.MaxPooling1D(2, padding='same')(x)

            x = layers.Conv1D(8, 16, activation = 'tanh', padding = 'same')(enc)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling1D(2)(x)
            x = layers.Conv1D(32, 32, activation = 'relu', padding = 'same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling1D(2)(x)
            dec = layers.Conv1D(1, 64, padding = 'same', activation = 'tanh')(x)

            x_flat = layers.Flatten()(enc)
            
            p = layers.Dense(2, activation = 'softmax')(x_flat)


            model = Model(X, [dec, p])

            hub = tf.keras.losses.Huber(delta=0.5, name='huber_loss') 
            model.compile(optimizer = 'adam', loss=['mse', hub], loss_weights = [1,200],
                          metrics = ['mse','acc'])

            K.set_value(model.optimizer.learning_rate, lr)
            return model
        return self.model
    
    def predict(self, X):
        model = self.model
        y_pred = model.predict(norm(X))
        
        pol_pred = np.argmax(y_pred[1], axis = 1)
        pred_prob = np.max(y_pred[1], axis = 1)
        predictions = []
        polarity = ['Negative', 'Positive']
        for pol, prob in zip(pol_pred, pred_prob):
            predictions.append((polarity[pol], prob))
                
        return predictions
