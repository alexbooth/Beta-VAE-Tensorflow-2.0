""" Beta-VAE training script """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils import * # TODO rename util
from utils import sampling
import dataset

from absl import app
from absl import flags

import numpy as np

flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", False, "continue training same weights")
flags.DEFINE_boolean("keep_best", False, "only save model if it got the best loss")
flags.DEFINE_integer("latent_dim", 32, "size of latent vector Z")
flags.DEFINE_integer("gamma", 1000, "gamma parameter")
flags.DEFINE_integer("capacity_iters", 1000, "number of iterations to reach max capacity")
flags.DEFINE_integer("max_capacity", 25, "maximum capacity (in nats) of the vae")
FLAGS = flags.FLAGS


def train(model, data):
    n_batches = data.training_set_size // FLAGS.batch_size
    
    def sample(step):
        if step % 2: #TODO sample at log(step) intervals
            sampling.append_frame(base_dir=timestamp, decoder=model.vae, frame_num=1)

    def print_info(batch, recon_err, kl, loss, n_batches):
        # Print training info
        str_out = " recon: {}".format(round(float(recon_err), 2)) # TODO make this neater
        str_out += " kl: {}".format(round(float(kl),2))
        str_out += " beta: {}".format(round(float(model.beta), 2))
        progress_bar(batch, n_batches, loss, epoch, FLAGS.epochs, suffix=str_out)

    for epoch in range(FLAGS.epochs):
        for batch in range(n_batches):
            X = data.get_batch(FLAGS.batch_size)
            loss, recon_err, kl = model.vae.train_on_batch(X, X)

            print_info(batch, recon_err, kl, loss, n_batches)
            sample(epoch*n_batches + batch)

        save_model(model.vae, epoch, loss, kl, recon_err)
    print("Finished training.")  


def main(argv):
    """ (1) General setup (directories, gpu, etc.)
        (2) Load data manager for the desired dataset
        (3) Load model and begin training """
    setup(FLAGS)
    dm = dataset.DspritesManager(batch_size=FLAGS.batch_size, color=True) 
    model = load_model(dm)
    train(model, dm)

if __name__ == '__main__':
    app.run(main)
