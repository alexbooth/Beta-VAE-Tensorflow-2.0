from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils import * # TODO rename util file or dir
import matplotlib.pyplot as plt

import numpy as np

last_frame = 0
squares = 0


def append_frame(base_dir, model, data, step):
    global last_frame, squares
    """ Appends a frame to the animation in base_dir 

        Frames are appended at an interval of every other perfect square of sqrt(step)
    """
    if not np.sqrt(step).is_integer():
        return

    squares += 1

    if squares % 2 != 0:
        return

    print("SAMPLING")
    inputs = np.array([data.get_nth_sample(0)])
    out = model.predict(inputs, mode="encode")[0]

    x = np.array([out])
    out = model.predict(x, mode="decode")[0]
    plt.imshow(out)
    plt.show()
    
    inputs = np.array([data.get_nth_sample(0)])
    out = model.predict(inputs)[0]
    plt.imshow(out)
    plt.show()
