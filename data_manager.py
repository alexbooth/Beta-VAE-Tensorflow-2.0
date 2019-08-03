from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import urllib.request
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
    def __init__(self):
        super(DspritesManager, self).__init__()
        self.filepath = "data/dsprites"
        self.filename = "dsprites.npz"
        self.init_dsprites()

    def init_dsprites(self):
        self.init_dir(self.filepath)
        os.chdir(self.filepath)
        self.download_dsprites()
        self.X = np.load(self.filename)["imgs"]
        self.training_set_size = self.X.shape[0] # TODO better way to assign var..?

    def download_dsprites(self):
        """Download dsprites to disk"""
        url = "https://github.com/deepmind/dsprites-dataset/raw/master"\
              "/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.download_file(self.filename, url)

    def get_batch(self, N):
        """Returns random batch of N samples from X"""
        indexes = np.random.randint(self.X.shape[0], size=N)
        batch = self.X[indexes, :].astype(np.float32)
        batch = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))
        return np.array(batch, dtype=np.float32), np.array(batch, dtype=np.float32)

    def show_random_sample(self): # TODO modify to show an example from get_batch
        """Plots a single random sample from the dataset"""
        x, _ = self.get_batch(1)
        x = x.reshape((64, 64))
        x = np.array(x, dtype=np.float32)
        plt.imshow(x)
        plt.show()
        

dm = DspritesManager()
dm.summary()
dm.show_random_sample()

