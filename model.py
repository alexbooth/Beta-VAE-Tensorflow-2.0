from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import Flatten, Reshape


def Conv(n_filters, filter_width, strides=2, activation="relu"):
    return Conv2D(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation)

def Deconv(n_filters, filter_width, strides=2, activation="relu"):
    return Conv2DTranspose(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation)

class BetaVAE(tf.keras.Model):
    def __init__(self, latent_dim=16):
        super(BetaVAE, self).__init__()

        # encoder layers
        self.conv_e0 = Conv(32, 4)
        self.conv_e1 = Conv(32, 4)
        self.conv_e2 = Conv(32, 4)
        self.conv_e3 = Conv(32, 4)
        self.f_e4 = Flatten()
        self.fc_e5 = Dense(256, activation="relu")
        self.fc_e6 = Dense(10,  activation="relu")
        self.fc_e7_mu = Dense(latent_dim)
        self.fc_e8_std = Dense(latent_dim) # TODO actually log_var please label correctly

        # decoder layers
        self.fc_d0 = Dense(10,  activation="relu")
        self.fc_d1 = Dense(256,  activation="relu")
        self.reshape_d2 = Reshape((4, 4, 16))
        self.deconv_d3 = Deconv(32, 4)
        self.deconv_d4 = Deconv(32, 4)
        self.deconv_d5 = Deconv(32, 4)
        self.deconv_d6 = Deconv(32, 4)
        self.conv_e7 = Conv(1, 1, strides=1, activation="sigmoid")
        self.reshape_d8 = Reshape((64, 64, 1))

    def encode(self, inputs):
        X = self.conv_e0(inputs)
        X = self.conv_e1(X)
        X = self.conv_e2(X)
        X = self.conv_e3(X)
        X = self.f_e4(X)
        X = self.fc_e5(X)
        X = self.fc_e6(X)
        Z_mu = self.fc_e7_mu(X)
        Z_std = self.fc_e8_std(X)
        return Z_mu, Z_std

    def reparameterize(self, Z_mu, Z_std):
        # Using reparameterization trick, sample random latent vectors Z from 
        # the latent Gaussian distribution which has mean = Z_mu and std = Z_std
        epsilon = tf.random.normal(tf.shape(Z_mu))
        return Z_mu + tf.sqrt(tf.math.exp(0.5 * Z_std)) * epsilon

    def decode(self, Z):
        X = self.fc_d0(Z)
        X = self.fc_d1(X)
        X = self.reshape_d2(X)
        X = self.deconv_d3(X)
        X = self.deconv_d4(X)
        X = self.deconv_d5(X)
        X = self.deconv_d6(X)
        X = self.conv_e7(X)
        X = self.reshape_d8(X)
        return X 

    def call(self, inputs, training=None, mask=None):
        Z_mu, Z_std = self.encode(inputs)
        Z = self.reparameterize(Z_mu, Z_std)
        X_pred = self.decode(Z)
        return X_pred, Z_mu, Z_std

