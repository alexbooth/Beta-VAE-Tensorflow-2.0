from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Lambda
from tensorflow.keras.layers import Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam


def Conv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2D(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation, name=name)


def Deconv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return Conv2DTranspose(n_filters, filter_width, 
                  strides=strides, padding="same", activation=activation, name=name)


class BetaVAE:
    def __init__(self, input_shape, latent_dim=32, loss_type="mse"):
        super(BetaVAE, self).__init__()
        self.loss_type = loss_type
        self.latent_dim = latent_dim
        self.channels = input_shape[2]
        self.dims = np.prod(input_shape) 
        self.beta = 0

        # create encoder
        encoder_input = Input(shape=input_shape)
        X = Conv(32, 4)(encoder_input)
        X = Conv(32, 4)(X)
        X = Conv(32, 4)(X)
        X = Conv(32, 4)(X)
        X = Flatten()(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(256,  activation="relu")(X)
        self.Z_mu = Dense(self.latent_dim)(X)
        self.Z_logvar = Dense(self.latent_dim, activation="relu")(X) + 1e-5
        self.Z = Lambda(self.reparameterize, output_shape=(self.latent_dim,), name="encoder_output")([self.Z_mu, self.Z_logvar])
        self.encoder = Model(encoder_input, [self.Z_mu, self.Z_logvar, self.Z])

        # create decoder
        output_activation = "sigmoid" if self.channels == 1 else None
        decoder_input = Input(shape=(self.latent_dim,))
        X = Dense(256,  activation="relu", name="decoder_input")(decoder_input)
        X = Dense(256,  activation="relu")(X)
        X = Dense(512,  activation="relu")(X)
        X = Reshape((4, 4, 32))(X)
        X = Deconv(32, 4)(X)
        X = Deconv(32, 4)(X)
        X = Deconv(32, 4)(X)
        decoder_output = Deconv(self.channels, 4, activation=output_activation, name="decoder_output")(X)
        self.decoder = Model(decoder_input, decoder_output)

        def reconstruction_loss(X, X_pred):
            if self.loss_type == "bce":
                bce = tf.losses.BinaryCrossentropy() 
                return bce(X, X_pred) * self.dims
            elif self.loss_type == "mse":
                mse = tf.losses.MeanSquaredError()
                return mse(X, X_pred) * self.dims
            else:
                raise ValueError("Unknown reconstruction loss type. Try 'bce' or 'mse'")

        def kl_divergence(X, X_pred):
            self.beta += (1/1440) / 4 # TODO use correct scalar
            self.beta = min(self.beta, 25)
            return self.beta * -0.5 * tf.reduce_mean(1 + self.Z_logvar - self.Z_mu**2 - tf.math.exp(self.Z_logvar))

        def loss(X, X_pred):
            return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)

        # create vae
        optimizer = Adam(0.0005) # TODO move to train, TODO use flag learning rate
        self.vae = Model(encoder_input, self.decoder(self.Z))
        self.vae.compile(optimizer=optimizer, loss=loss, metrics=[reconstruction_loss, kl_divergence])

    def reparameterize(self, args):
        """
        Reparameterization trick, sample random latent vectors Z from 
        the latent Gaussian distribution which has the following parameters 

        mean = self.Z_mu
        std = exp(0.5 * self.Z_logvar)
        """
        self.Z_mu, self.Z_logvar = args
        epsilon = tf.random.normal(tf.shape(self.Z_mu))
        sigma = tf.math.exp(0.5 * self.Z_logvar)
        return self.Z_mu + sigma * epsilon

    def predict(self, inputs, mode=None):
        if mode == "encode":
            _, _, Z = self.encoder.predict(inputs)
            return Z
        if mode == "decode":
            return self.decoder.predict(inputs)
        if mode == None:
            return self.vae.predict(inputs) 
        raise ValueError("Unsupported mode during call to model.") 
