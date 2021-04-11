from typing import Tuple, List, Union, Optional
import os
import time
import math

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


############################################
############################################
def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


############################################
############################################
def toc(t_start: float, name: Optional[str] = "Operation", ftime=False) -> None:
    """ Timing Function """
    assert isinstance(t_start, float)
    sec: float = time.time() - t_start
    if ftime:
        duration = time.strftime("%H:%M:%S", time.gmtime(sec))
        print(f'\n############ {name} took: {str(duration)} ############\n')
    else:
        print(f'\n############ {name} took: {sec:.4f} sec. ############\n')


########################################################################################

def calc_n_out(n_in: int, k: int, s: int, p: int):
    """
    n_{out} = floor[(n_{in} + 2p - k) / s] + 1
    :param n_in: num of input features
    :param k: convolution kernel size
    :param p: convolution padding size
    :param s: convolution strode size
    :return n_out: num of output features
    """
    return math.floor((n_in + 2*p - k) / s) + 1


def im_show(img: torch.Tensor, title=None):
    """
    show image use matplotlib
    """
    # TODO: unnormalize
    # img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.show()


def display_image_grid(images_filepaths=None, images_array_lst=None, predicted_labels=(), cols=5):

    if images_filepaths is not None:
        rows = len(images_filepaths) // cols
    else:
        rows = len(images_array_lst) // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    if images_array_lst:
        for i, img_array in enumerate(images_array_lst):
            img_array = np.transpose(img_array.numpy(), (1, 2, 0))

            ax.ravel()[i].imshow(img_array.astype('uint8'))
    else:
        for i, image_filepath in enumerate(images_filepaths):
            image = cv2.imread(image_filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
            predicted_label = predicted_labels[i] if predicted_labels else true_label
            color = "green" if true_label == predicted_label else "red"
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_title(predicted_label, color=color)
            ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
