""" Beta-VAE training script """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils import * # TODO rename util file or dir
from utils import sampling
import dataset

from absl import app
from absl import flags


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
    """ Trains model with the given data """
    n_batches = data.training_set_size // FLAGS.batch_size
    
    def sample(step):
        """ Create latent traversal animation """
        sampling.append_frame(timestamp, model, data, step)

    def print_info(batch, epoch, recon_err, kl, loss):
        """ Print training info """
        str_out = " recon: {}".format(round(float(recon_err), 2))
        str_out += " kl: {}".format(round(float(kl),2))
        str_out += " capacity (nats): {}".format(round(float(model.C), 2))
        progress_bar(batch, n_batches, loss, epoch, FLAGS.epochs, suffix=str_out)

    # Training loop
    for epoch in range(FLAGS.epochs):
        for batch in range(n_batches):
            X = data.get_batch(FLAGS.batch_size)
            loss, recon_err, kl = model.vae.train_on_batch(X, X)

            print_info(batch, epoch, recon_err, kl, loss)
            sample(epoch*n_batches + batch)

        save_model(model.vae, epoch, loss, kl, recon_err)

    print("Finished training.")  


def main(argv):
    """ (1) General setup (directories, gpu, etc.)
        (2) Load data manager for the desired dataset
        (3) Load model and begin training """
    setup(FLAGS)
    dm = dataset.DspritesManager(batch_size=FLAGS.batch_size, color=True) # TODO make neater
    model = load_model(dm)
    train(model, dm)


if __name__ == '__main__':
    app.run(main)
