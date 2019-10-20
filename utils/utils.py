""" All the boring stuff! """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from datetime import datetime

from model import BetaVAE

best_loss = float("inf")
model_path = None
summary_path = None
FLAGS = None
timestamp = str(datetime.now())


# TODO make utils a helper class instead of functions?
def progress_bar(iteration, total, loss, epoch, total_epochs, prefix='', suffix='', decimals=1, bar_length=40):
    """
    Print iterations progress
    Source: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    loss_str = "Epoch: " + str(epoch+1) + "/" + str(total_epochs) + " Loss: " + str(loss)

    if iteration < total:
        sys.stdout.write('\r%s |%s| %s%s %s %s' % (prefix, bar, percents, '%', suffix, loss_str))
    else:
        sys.stdout.write('\r%s |%s| 100.0% %s\n' % (prefix, bar, suffix))
    
    sys.stdout.flush()



def setup(flags):
    global FLAGS
    FLAGS = flags

    if tf.__version__[:2] != "2.":
        print("Tensorflow 2.x is not installed. Exiting.")
        sys.exit()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")

    path_setup()



def path_setup():
    """Create log and trained_model dirs. """
    global model_path, summary_path, timestamp
    model_base_dir = "./trained_model"
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(model_base_dir, exist_ok=True)
    model_dir = os.path.join(model_base_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    if FLAGS.keep_training and os.listdir(FLAGS.logdir):
        files = filter(os.path.isdir, glob.glob(FLAGS.logdir + "/*"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        timestamp = os.path.basename(os.path.normpath(list(reversed(files))[0]))

    model_path = os.path.abspath(os.path.join(model_dir, "betaVAE.h5"))
    summary_path = os.path.join(FLAGS.logdir, timestamp)



def load_model(dm):
    """Build and return the model."""
    input_shape = dm.train_input_shape
    loss_type = "mse" if dm.color else "bce"
    model = BetaVAE(input_shape, latent_dim=FLAGS.latent_dim, loss_type=loss_type) 
    model.vae.summary()
    return model


# TODO fix this 
def save_model(model, epoch, loss, recon, kl):
    """Write logs and save the model"""
    train_summary_writer = tf.summary.create_file_writer(summary_path)
    with train_summary_writer.as_default():
        tf.summary.scalar("Total Loss", loss, step=epoch)
        tf.summary.scalar("KL Divergence", kl, step=epoch)
        tf.summary.scalar("Reconstruction Loss", recon, step=epoch)

    # save model
    """global best_loss
    if not FLAGS.keep_best: 
        model.save(model_path)
    elif loss < best_loss:
        best_loss = loss
        model.save(model_path)"""
    print("saving model to:", model_path)
    model.save(model_path)
