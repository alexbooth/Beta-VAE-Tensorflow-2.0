from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils import * # TODO rename util file or dir
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

frame_num = 0
inputs = None

def append_frame(base_dir, model, data, step):
    global frame_num, squares, inputs
    """ Appends a frame to the animation in base_dir 

        Frames are appended at an interval of every other perfect square of sqrt(step)
    """
    if not np.sqrt(step).is_integer():
        return

    if frame_num == 0:
        inputs = np.array([data.get_nth_sample(0)])

    frame_num += 1

    print(" SAMPLING FRAME {}".format(frame_num))
    latent_vars = model.predict(inputs, mode="encode")[0]

    frames_per_traverse = 60
    step_size = 6 / frames_per_traverse

    sample_batch = np.array([latent_vars]*32)
    for i in range(model.latent_dim):
        sample_batch[i][i] = -3 + 2*np.arccos(np.cos(step_size))
    sample_batch = model.predict(sample_batch, mode="decode")
    sample_batch = np.clip(sample_batch, 0, 1)
    for i, im in enumerate(sample_batch):
        save_dir = "latent_{}".format(i)
        os.makedirs(save_dir, exist_ok=True)
        im_path = os.path.join(save_dir, "frame_{}.png".format(frame_num))
        matplotlib.image.imsave(im_path, im)
