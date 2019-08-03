# All of the boring stuff!

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from datetime import datetime


""" 
    Print iterations progress
    Source: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
"""
def progress_bar(iteration, total, loss, epoch, total_epochs, prefix='', suffix='', decimals=1, bar_length=40):
    """
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
    loss_str = "Epoch: " + str(epoch+1) + "/" + str(total_epochs) + " Loss: " + str(loss.numpy())

    if iteration < total:
        sys.stdout.write('\r%s |%s| %s%s %s %s' % (prefix, bar, percents, '%', suffix, loss_str))
    else:
        sys.stdout.write('\r%s |%s| 100.0% %s\n' % (prefix, bar, suffix))
    
    sys.stdout.flush()

def gpu_setup():
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

def path_setup(FLAGS):
    """Create log and trained_model dirs. """
    global model_path, summary_path
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs("./trained_model", exist_ok=True)
    timestamp = str(datetime.now())

    if FLAGS.keep_training and os.listdir(FLAGS.logdir):
        files = filter(os.path.isdir, glob.glob(FLAGS.logdir + "/*"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        timestamp = os.path.basename(os.path.normpath(list(reversed(files))[0]))

    model_path = os.path.join("./trained_model/DAE-model-" + timestamp + ".h5")
    summary_path = os.path.join(FLAGS.logdir, timestamp)
