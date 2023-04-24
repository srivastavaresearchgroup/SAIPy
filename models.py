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
import tensorflow as tf

def mag_estimation_loss(y_true, y_pred):
    
    mask = y_true[-1] 

    return (y_true - y_pred) * mask 

def magn_estimation_loss(y_true, y_pred):
    
    m_true = tf.gather(y_true,-1, axis = -1)
    m_pred = tf.gather(y_pred,-1, axis = -1)
    
#     print(K.mean(tf.multiply(tf.subtract(m_true, m_pred), m_true)))
    mse = tf.keras.losses.MeanSquaredError()
    
    return mse(m_true, m_pred)# * tf.square(tf.add(m_true,5))


def custom_loss2(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    
    weights = [0.4, 0.4, 0.2]
    
    
    return weights[0] * mse(y_true, y_pred) + weights[1] * mae(y_true, y_pred) + weights[2] * mag_estimation_loss(y_true, y_pred)

def custom_loss3(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    cos = tf.keras.losses.CosineSimilarity()
     
#     print(mse(y_true, y_pred) * (1 + cos(y_true, y_pred)))

    
    return mse(y_true, y_pred) * (1 + cos(y_true, y_pred)) + 5 * magn_estimation_loss(y_true, y_pred)

def norm(X):
    maxi = np.max(abs(X), axis = 1)
    X_ret = X.copy()
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret

from keras.layers import Dense, GlobalAveragePooling1D, Activation, Reshape, Permute, multiply
from scipy.signal import stft

def spec(X):
    X_spec = []
    for i in range(3):
        X_spec.append(np.absolute(stft(X[:,i], 100)[2]))
    
    return np.transpose(np.array(X_spec), (1,2,0))

def SEBlock(se_ratio = 16, activation = "relu", data_format = 'channels_last', ki = "he_normal"):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''

    def f(input_x):

        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input_x.shape[channel_axis]

        reduced_channels = input_channels // se_ratio

        #Squeeze operation
        x = GlobalAveragePooling1D()(input_x)
        x = Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(reduced_channels, kernel_initializer= ki)(x)
        x = Activation(activation)(x)
        #Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = multiply([input_x, x])

        return x

    return f


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
    
class CREIME_RT:
    def __init__(self):
        with open('Models/CREIME_RT.json', 'r') as json_file:
            json_savedModel= json_file.read()
        #load the model architecture 
        self.model = tf.keras.models.model_from_json(json_savedModel)
        self.model.load_weights('Models/CREIME_RT.h5')
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary


        
    def get_model(self, untrained = False):
        if untrained:
            lr = 1e-3
    
            X1 = Input(shape = (6000,3))
            X2 = Input(shape = (129,48,3))

            x = layers.Conv1D(64, 8, padding = 'same')(X1)
            x = layers.PReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(32, 8, padding = 'same')(x)
            x = layers.PReLU()(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(16, 4, padding = 'same')(x)
            x = layers.PReLU()(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(8, 16, padding = 'same')(x)
            x = layers.PReLU()(x)
            x1 = layers.MaxPooling1D(4, padding='same')(x)

            x2 = layers.Conv2D(64, (4,4), padding = 'same')(X2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Conv2D(32, 4)(x2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Conv2D(8, 4)(x2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Reshape((39, 8))(x2)
        #     print(x2)

            x = layers.Concatenate(axis=1)([x1, x2])
            x = SEBlock()(x)

            x = layers.Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(x)
            x = layers.Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(x)
            x = layers.Bidirectional(LSTM(512, return_sequences=False, dropout = 0.2))(x)
            x = layers.PReLU()(x)
            y = layers.Dense(6000)(x)
            model = Model([X1, X2], y)

        #     cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

            model.compile(optimizer = 'adam', loss=custom_loss3)

            K.set_value(model.optimizer.learning_rate, lr)
            return model
        return self.model
    
    def predict(self, X):
        model = self.model
        
        X_test = np.zeros((len(X), 6000, 3))
        
        for i,x in enumerate(X):
            X_test[i,:len(x),:] = x
        
        X_test_stft = np.array(list(map(spec, list(X_test))))
        y_pred = model.predict([X_test, X_test_stft])
        
        mp = [np.mean(y[-10:]) for y in y_pred]
        
        n_p = [mval < -0.5 for mval in mp]
        
        predictions = []
        
        for n,m in zip(n_p, mp):
            if n:
                predictions.append([0,None])
            else:
                predictions.append((1, m))
                
        return y_pred, predictions