from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import urllib.request
import tensorflow as tf
from matplotlib import pyplot as plt



class DataManager:
    def __init__(self):
        self.X = np.zeros(0)
        self.Y = np.zeros(0)
        self.training_set_size = 0

    def init_dir(self, dir_path):
        """Ensures directory dir_path exists"""
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


    def download_file(self, filename, url):
        """Download file from url if it isn't already on disk"""
        if os.path.isfile(filename):
            return
        urllib.request.urlretrieve(url, filename)

    def get_batch(self, N):
        print("Function 'get_batch' not implemented in child class!")

    def summary(self):
        print("Found", self.X.shape[0], "images")
        print("size:", self.X.shape[1], "x", self.X.shape[2])
        print("type:", self.X.dtype)
        

class DspritesManager(DataManager):
    def __init__(self, batch_size=32, color=False):
        super(DspritesManager, self).__init__()
        self.batch_size = batch_size
        self.color = color
        self.data_shape = (64, 64)
        self.filepath = "data/dsprites"
        self.filename = "dsprites.npz"
        self.in_channels = 3 if color else 1
        self.train_input_shape = tf.TensorShape([64, 64, self.in_channels])
        self.init_dsprites()

    def init_dsprites(self):
        self.init_dir(self.filepath)
        os.chdir(self.filepath)
        self.download_dsprites()
        self.X = np.load(self.filename)["imgs"]
        self.training_set_size = self.X.shape[0]
        self.make_dataset()

    def download_dsprites(self):
        """Download dsprites to disk"""
        url = "https://github.com/deepmind/dsprites-dataset/raw/master"\
              "/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.download_file(self.filename, url)

    def get_nth_sample(self, n):
        """
        Returns the nth sample in the shape of self.train_input_shape
        Useful for running a specific example through the network
        """
        x = self.X[n, :].astype(np.float32)
        return self.add_color_to_sample(x) if self.color else x

    def add_color_to_sample(self, x):
        if not self.color:
            return x
        rand = np.random.uniform(size=x.shape) 
        mask = x==1
        r = x.copy()
        r[mask] = np.random.uniform()
        g = x.copy() 
        g[mask] = np.random.uniform()
        b = x.copy() 
        b[mask] = np.random.uniform()
        x = np.stack((r, g, b), axis=-1)
        return x

    def get_batch(self, N):
        if (N != self.batch_size):
            self.batch_size = N
            self.dataset = self.dataset.batch(batch_size=self.batch_size)
            self.dataset_iterator = iter(self.dataset)
        return next(self.dataset_iterator)

    def show_nth_sample(self, n):
        """Plots the nth sample from the dataset"""
        x = self.get_nth_sample(n).squeeze()
        plt.imshow(x)
        plt.show()

    def show_random_sample(self):
        """Plots a single random sample from the dataset"""
        n = np.random.randint(self.training_set_size)
        self.show_nth_sample(n)

    def generate(self):
        while True:
            n = np.random.randint(self.training_set_size)
            x = self.X[n, :].astype(np.float32)
            x = self.add_color_to_sample(x)
            yield x

    def make_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(self.generate, tf.float32, output_shapes=self.train_input_shape)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=self.batch_size)
        self.dataset_iterator = iter(self.dataset)
