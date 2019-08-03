from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from utils import progress_bar, gpu_setup, path_setup
from data_manager import DspritesManager

from absl import app
from absl import flags

from model import BetaVAE
from tensorflow.keras.optimizers import Adam


flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch-size", 32, "batch size")
flags.DEFINE_float("learning-rate", 0.0001, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep-training", False, "continue training same weights")
flags.DEFINE_boolean("keep_best", False, "only save model if it got the best loss")
FLAGS = flags.FLAGS

best_loss = np.inf
model_path = None

def custom_loss(X, X_pred, Z_mu, Z_std):
    bce = tf.losses.BinaryCrossentropy() # TODO don't declare this each iteration
    reconstruction_error = tf.reduce_mean(bce(X, X_pred)) # TODO rename... not really reconstruction error is it?
    beta = 2
    kl_divergence = -0.5 * tf.reduce_mean(1 + Z_std - Z_mu**2 - tf.math.exp(Z_std))
    return reconstruction_error + beta * kl_divergence

def train(model):
    dm = DspritesManager() 
    n_batches = dm.training_set_size // FLAGS.batch_size
    optimizer = Adam(FLAGS.learning_rate)

    Z_mu, Z_std = None, None
    for epoch in range(FLAGS.epochs):
        for batch in range(n_batches):
            X, _ = dm.get_batch(FLAGS.batch_size)
            
            with tf.GradientTape() as tape:
                X_pred, Z_mu, Z_std = model(X)
                loss = custom_loss(X, X_pred, Z_mu, Z_std)
                progress_bar(batch, n_batches, loss, epoch, FLAGS.epochs)
            gradients = tape.gradient(loss, model.trainable_variables)
            # TODO apply gradient clipping
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        out = model.decode(model.reparameterize(Z_mu, Z_std))[0]
        plt.imshow(out.numpy().reshape((64,64)))
        plt.show()

        #save_model(model, epoch, loss) # TODO keep this
    print("Finished training.")  
    

def save_model(model, epoch, loss):
    """Write logs and save the model"""
    train_summary_writer = tf.summary.create_file_writer(summary_path)
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)

    # save model
    global best_loss
    if not FLAGS.keep_best: 
        model.save(model_path)
    elif loss < best_loss:
        best_loss = loss
        model.save(model_path)

def load_model():
    """Set up and return the model."""
    model = BetaVAE()
    model.build(tf.TensorShape([None, 64, 64, 1]))

    # TODO add this loading functionality back in
    # load most recent weights if model_path exists  
    #if os.path.isfile(model_path):
    #    print("Loading model from", model_path)
    #    model.load_weights(model_path)

    model.summary()
    return model

def main(argv):
    path_setup(FLAGS)
    model = load_model()
    train(model)

if __name__ == '__main__':
    gpu_setup()
    app.run(main)
