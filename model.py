from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam


def Conv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2D(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation, name=name)


def Deconv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2DTranspose(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation, name=name)


class Reparameterize(tf.keras.layers.Layer):
    """
    Custom layer.
     
    Reparameterization trick, sample random latent vectors Z from 
    the latent Gaussian distribution which has the following parameters 

    mean = Z_mu
    std = exp(0.5 * Z_logvar)
    """
    def call(self, inputs):
        Z_mu, Z_logvar = inputs
        epsilon = tf.random.normal(tf.shape(Z_mu))
        sigma = tf.math.exp(0.5 * Z_logvar)
        return Z_mu + sigma * epsilon


class BetaVAE:
    def __init__(self, input_shape, latent_dim=32, loss_type="mse", learning_rate=0.0005):
        self.latent_dim = latent_dim
        self.C = 0
        self.gamma = 100

        channels = input_shape[2]

        # create encoder
        encoder_input = Input(shape=input_shape)
        X = Conv(32, 4)(encoder_input)
        X = Conv(32, 4)(X)
        X = Conv(32, 4)(X)
        X = Conv(32, 4)(X)
        X = Flatten()(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(256,  activation="relu")(X)
        Z_mu = Dense(self.latent_dim)(X)
        Z_logvar = Dense(self.latent_dim, activation="relu")(X)
        Z = Reparameterize()([Z_mu, Z_logvar])

        # create decoder
        output_activation = "sigmoid" if channels == 1 else None
        decoder_input = Input(shape=(self.latent_dim,))
        X = Dense(256,  activation="relu")(decoder_input)
        X = Dense(256,  activation="relu")(X)
        X = Dense(512,  activation="relu")(X)
        X = Reshape((4, 4, 32))(X)
        X = Deconv(32, 4)(X)
        X = Deconv(32, 4)(X)
        X = Deconv(32, 4)(X)
        decoder_output = Deconv(channels, 4, activation=output_activation)(X)

        # define vae losses
        def reconstruction_loss(X, X_pred):
            if loss_type == "bce":
                bce = tf.losses.BinaryCrossentropy() 
                return bce(X, X_pred) * np.prod(input_shape)
            elif loss_type == "mse":
                mse = tf.losses.MeanSquaredError()
                return mse(X, X_pred) * np.prod(input_shape)
            else:
                raise ValueError("Unknown reconstruction loss type. Try 'bce' or 'mse'")

        def kl_divergence(X, X_pred):
            self.C += (1/1440) # TODO use correct scalar
            self.C = min(self.C, 35) # TODO make variable
            kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu**2 - tf.math.exp(Z_logvar))
            return self.gamma * tf.math.abs(kl - self.C)

        def loss(X, X_pred):
            return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)

        # create models
        self.encoder = Model(encoder_input, [Z_mu, Z_logvar, Z])
        self.decoder = Model(decoder_input, decoder_output)
        self.vae = Model(encoder_input, self.decoder(Z))
        self.vae.compile(optimizer='adam', loss=loss, metrics=[reconstruction_loss, kl_divergence])

    def predict(self, inputs, mode=None):
        if mode == "encode":
            _, _, self.Z = self.encoder.predict(inputs)
            return self.Z
        if mode == "decode":
            return self.decoder.predict(inputs)
        if mode == None:
            return self.vae.predict(inputs) 
        raise ValueError("Unsupported mode during call to model.") 
