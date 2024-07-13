import random
import os,sys
sys.path.insert(0, '..')
import os
import pandas as pd
import h5py
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm
from random import randint
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM
from keras import layers
import tensorflow as tf
from keras import backend as K

def norm(X):
    maxi = np.max(abs(X), axis = 1)
    X_ret = X.copy()
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret
        
class PolarCAP:
    def __init__(self):
        self.model = keras.models.load_model('../saipy/saved_models/PolarCAP.h5')
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def get_model(self, untrained = False):
        # get trained model or model with random initialization
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
        # get model predictions 1 (up) and 0 (down) for a waveform
        model = self.model
        y_pred = model.predict(norm(X))
        
        pol_pred = np.argmax(y_pred[1], axis = 1)
        pred_prob = np.max(y_pred[1], axis = 1)
        predictions = []
        polarity = ['Negative', 'Positive']
        for pol, prob in zip(pol_pred, pred_prob):
            predictions.append((polarity[pol], prob))
                
        return predictions
