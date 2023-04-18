import os
import pandas as pd
import h5py
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm
from random import randint
import keras
from keras.models import Model
from keras.layers import Input, LSTM
from keras import layers

def mag_estimation_loss(y_true, y_pred):
    
    mask = y_true[-1] 

    return (y_true - y_pred) * mask 

def custom_loss2(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    
    weights = [0.4, 0.4, 0.2]
    
    
    return weights[0] * mse(y_true, y_pred) + weights[1] * mae(y_true, y_pred) + weights[2] * mag_estimation_loss(y_true, y_pred)

def norm(X):
    maxi = np.max(abs(X), axis = 1)
    X_ret = X.copy()
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret

class CREIME:
    def __init__(self):
        self.model = keras.models.load_model('Models/CREIME.h5', custom_objects = {'custom_loss2': custom_loss2})
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary


        
    def get_model(self, untrained = False):
        if untrained:
            X = Input(shape = (512,3))
            x = layers.Conv1D(32, 16, padding = 'same')(X)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(16, 16, padding = 'same')(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(8, 16, padding = 'same')(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(x)
            x = layers.Bidirectional(LSTM(256, return_sequences=False, dropout = 0.2))(x)
            y = layers.Dense(512)(x)
            model = Model(X, y)
   
            model.compile(optimizer = 'rmsprop', loss=custom_loss2)
    
            return model
        return self.model
    
    def predict(self, X):
        model = self.model
        y_pred = model.predict(X)
        
        pp = 512 - np.array([np.sum((y > -0.5).astype(int)) for y in y_pred])
        mp = [np.mean(y[-10:]) for y in y_pred]
        
        n_p = [mval < -0.5 for mval in mp]
        
        predictions = []
        
        for n,m,p in zip(n_p, mp, pp):
            if n:
                predictions.append([0,None,None])
            else:
                predictions.append((1, m, p))
                
        return y_pred, predictions
            
        
class PolarCAP:
    def __init__(self):
        self.model = keras.models.load_model('Models/PolarCAP.h5')
       
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
        #     x_flat = layers.Dense(8, activation = 'relu')(x_flat)
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