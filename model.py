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
    def __init__(self, latent_dim=64):
        super(BetaVAE, self).__init__()

        # encoder layers
        self.conv_e0 = Conv(16, 5)
        self.conv_e1 = Conv(16, 5)
        self.conv_e2 = Conv(16, 5, strides=1)
        self.conv_e3 = Conv(16, 5, strides=1)
        self.conv_e4 = Conv(16, 3, strides=1)
        self.conv_e5 = Conv(16, 3, strides=1)
        self.f_e6 = Flatten()
        self.fc_e7 = Dense(128, activation="relu")
        self.fc_e8 = Dense(64,  activation="relu")
        self.fc_e9 = Dense(32,  activation="relu")
        self.fc_e10_mu = Dense(latent_dim)
        self.fc_e10_std = Dense(latent_dim) # TODO actually log_var please label correctly

        # decoder layers
        self.fc_d0 = Dense(32,  activation="relu")
        self.fc_d1 = Dense(64,  activation="relu")
        self.fc_d2 = Dense(128, activation="relu")
        self.fc_d3 = Dense(64,  activation="relu")
        self.reshape_d4 = Reshape((2, 2, 16))
        self.deconv_d5 = Deconv(16, 3)
        self.deconv_d6 = Deconv(16, 3)
        self.deconv_d7 = Deconv(16, 5)
        self.deconv_d8 = Deconv(16, 5)
        self.deconv_d9 = Deconv(16, 5)
        self.conv_d10 = Conv2D(1, 1, activation="sigmoid")

    def encode(self, inputs):
        X = self.conv_e0(inputs)
        X = self.conv_e1(X)
        X = self.conv_e2(X)
        X = self.conv_e3(X)
        X = self.conv_e4(X)
        X = self.conv_e5(X)
        X = self.f_e6(X)
        X = self.fc_e7(X)
        X = self.fc_e8(X)
        X = self.fc_e9(X)
        Z_mu = self.fc_e10_mu(X)
        Z_std = self.fc_e10_std(X)
        return Z_mu, Z_std

    def reparameterize(self, Z_mu, Z_std):
        # Using reparameterization trick, sample random latent vectors Z from 
        # the latent Gaussian distribution which has mean = Z_mu and std = Z_std
        epsilon = tf.random.normal(tf.shape(Z_mu))
        return Z_mu + tf.sqrt(tf.math.exp(Z_std)) * epsilon

    def decode(self, Z):
        X = self.fc_d0(Z)
        X = self.fc_d1(X)
        X = self.fc_d2(X)
        X = self.fc_d3(X)
        X = self.reshape_d4(X)
        X = self.deconv_d5(X)
        X = self.deconv_d6(X)
        X = self.deconv_d7(X)
        X = self.deconv_d8(X)
        X = self.deconv_d9(X)
        X = self.conv_d10(X)
        return X 

    def call(self, inputs, training=None, mask=None):
        Z_mu, Z_std = self.encode(inputs)
        Z = self.reparameterize(Z_mu, Z_std)
        X_pred = self.decode(Z)
        return X_pred, Z_mu, Z_std

