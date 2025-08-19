import h5py
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import sys

from math import sqrt
from random import randint
from scipy.signal import stft
from tqdm import tqdm

from keras.models import Model
from keras import layers
from keras import backend as K

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, '..')

#____________________________________________________

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



def spec(X):
    X_spec = []
    for i in range(3):
        X_spec.append(np.absolute(stft(X[:,i], 100)[2]))
    
    return np.transpose(np.array(X_spec), (1,2,0))

'''
transformer block code from  S Mostafa Mousavi, William L Ellsworth, Weiqiang Zhu, Lindsay Y Chuang, and Gregory C Beroza. Earthquake
transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature
communications, 11(1):3952, 2020.
'''

class SeqSelfAttention(keras.layers.Layer):
    """Layer initialization. modified from https://github.com/CyberZHG
    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
    :param units: The dimension of the vectors that used to calculate the attention weights.
    :param attention_width: The width of local attention.
    :param attention_type: 'additive' or 'multiplicative'.
    :param return_attention: Whether to return the attention weights for visualization.
    :param history_only: Only use historical pieces of data.
    :param kernel_initializer: The initializer for weight matrices.
    :param bias_initializer: The initializer for biases.
    :param kernel_regularizer: The regularization for weight matrices.
    :param bias_regularizer: The regularization for biases.
    :param kernel_constraint: The constraint for weight matrices.
    :param bias_constraint: The constraint for biases.
    :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                              in additive mode.
    :param use_attention_bias: Whether to use bias while calculating the weights of attention.
    :param attention_activation: The activation used for calculating the weights of attention.
    :param attention_regularizer_weight: The weights of attention regularizer.
    :param kwargs: Parameters for parent class.
    """
        
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        input_len = inputs.get_shape().as_list()[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}
    
class LayerNormalization(keras.layers.Layer):
    
    """ 
    
    Layer normalization layer modified from https://github.com/CyberZHG based on [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    
    Parameters
    ----------
    center: bool
        Add an offset parameter if it is True. 
        
    scale: bool
        Add a scale parameter if it is True.     
        
    epsilon: bool
        Epsilon for calculating variance.     
        
    gamma_initializer: str
        Initializer for the gamma weight.     
        
    beta_initializer: str
        Initializer for the beta weight.     
                    
    Returns
    -------  
    data: 3D tensor
        with shape: (batch_size, …, input_dim) 
            
    """   
              
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):

        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
      

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
   
class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer. modified from https://github.com/CyberZHG 
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """
    
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y

def transformer(drop_rate, width, name, inpC): 
    # Returns a transformer block containing one addetive attention and one feed  forward layer with residual connections '
    x = inpC
    
    att_layer, weight = SeqSelfAttention(return_attention =True,                                       
                                         attention_width = width,
                                         name=name)(x)
   
    att_layer2 = layers.add([x, att_layer])    
    norm_layer = layers.LayerNormalization()(att_layer2)
    
    FF = FeedForward(units=128, dropout_rate=drop_rate)(norm_layer)
    
    FF_add = layers.add([norm_layer, FF])    
    norm_out = layers.LayerNormalization()(FF_add)
    
    return norm_out, weight
'''
End of transformer block code from  S Mostafa Mousavi, William L Ellsworth, Weiqiang Zhu, Lindsay Y Chuang, and Gregory C Beroza. Earthquake
transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature
communications, 11(1):3952, 2020.
'''

class CREIME:
    def __init__(self, path):
        if not os.path.exists(path):
            path = './saipy/saved_models/'        
        self.model = keras.models.load_model(path+'/CREIME.h5', custom_objects = {'custom_loss2': custom_loss2})
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def get_model(self, untrained = False):
        if untrained:
            X = layers.Input(shape = (512,3))
            x = layers.Conv1D(32, 16, padding = 'same')(X)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(16, 16, padding = 'same')(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(8, 16, padding = 'same')(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout = 0.2))(x)
            x = layers.Bidirectional(layers.LSTM(256, return_sequences=False, dropout = 0.2))(x)
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
                predictions.append((0,None,None))
            else:
                predictions.append((1, round(m,1), p))
                
        return y_pred, predictions
            
       
class CREIME_RT:
    def __init__(self, path):   
        if not os.path.exists(path):
            path = './saipy/saved_models/'
    
        with open(path+'CREIME_RT.json', 'r') as json_file:
            json_savedModel= json_file.read()
        #load the model architecture 
        self.model = tf.keras.models.model_from_json(json_savedModel,
                                                     custom_objects = {'SeqSelfAttention': SeqSelfAttention,
                                                                       'FeedForward': FeedForward})
        self.model.load_weights(path+'CREIME_RT.h5')
       
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary


        
    def get_model(self, untrained = False):
        if untrained:
            lr = 1e-3
    
            X1 = layers.Input(shape = (6000,3))
            X2 = layers.Input(shape = (129,48,3))

            x = layers.Conv1D(8, 128, padding = 'same')(X1)
            x = layers.PReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(16, 64, padding = 'same')(x)
            x = layers.PReLU()(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(32, 32, padding = 'same')(x)
            x = layers.PReLU()(x)
            x = layers.MaxPooling1D(4, padding='same')(x)
            x = layers.Conv1D(64, 8, padding = 'same')(x)
            x = layers.PReLU()(x)
            x1 = layers.MaxPooling1D(4, padding='same')(x)

            x2 = layers.Conv2D(8, (4,4), activation='relu', padding = 'same')(X2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Conv2D(32, (4,4), activation='relu')(x2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Conv2D(64, (4,4), activation='relu')(x2)
            x2 = layers.MaxPooling2D((2,2))(x2)
            x2 = layers.Reshape((39, 64))(x2)

            x = layers.Concatenate(axis=1)([x1, x2])
            x, weightdD0 = transformer(0.1, None, 'attentionD0', x)

            x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout = 0.2))(x)
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=False, dropout = 0.2))(x)
            x = layers.PReLU()(x)
            y = layers.Dense(6000)(x)
            model = Model([X1, X2], y)

            model.compile(optimizer = 'adam', loss=custom_loss3)

            K.set_value(model.optimizer.learning_rate, lr)
            return model
        return self.model
    
    def predict(self, X):
        model = self.model
        X_test = np.zeros((len(X), 6000, 3))
        
        for i,x in enumerate(X):
            X_test[i,:len(x),:] = x
  
        y_test = np.zeros((len(X), 6000))
        batch_size = 256
        
        def test_generator():
            while True:
                for start in range(0, len(y_test), batch_size):
                    
                    end = min(start + batch_size, len(y_test))
                    x_batch = X_test[start:end]
                    x_batch_stft = np.array(list(map(spec, list(x_batch))))
                    y_batch = np.array(y_test[start:end])

                    yield ([x_batch, x_batch_stft], y_batch)
                    
        y_pred = model.predict(test_generator(), steps = len(y_test) // batch_size + 1)
        
        mp = [np.mean(y[-10:]) for y in y_pred]
        
        n_p = [mval < -0.5 for mval in mp]
        
        predictions = []
        
        for n,m in zip(n_p, mp):
            if n:
                predictions.append((0,None))
            else:
                predictions.append((1, round(m,1)))     
        return y_pred, predictions
