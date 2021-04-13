from typing import Tuple, List, Union, Optional
import os
import time
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
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


############################################
############################################

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
    return math.floor((n_in + 2 * p - k) / s) + 1


############################################
############################################

def matplotlib_imshow(img: torch.Tensor, one_channel=False, title=None):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


########################################################################################

def display_image_grid(images_filepaths=None, images_array_lst=None, predicted_labels=None, true_labels=None, cols=5):
    title_label = ''
    color = "blue"
    if images_filepaths is not None:
        rows = len(images_filepaths) // cols
    else:
        rows = len(images_array_lst) // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    if images_array_lst:
        for i, img_array in enumerate(images_array_lst):
            img_array = np.transpose(img_array.numpy(), (1, 2, 0))

            if predicted_labels is not None and true_labels is not None:
                predicted_label = predicted_labels[i]
                true_label = true_labels[i]
                color = "green" if true_label == predicted_label else "red"
                title_label = str(predicted_label)
            elif predicted_labels is not None:
                predicted_label = str(int(predicted_labels[i]))
                title_label = predicted_label
            elif true_labels is not None:
                true_label = true_labels[i]
                title_label = str(int(true_label))
            else:
                raise RuntimeError

            ax.ravel()[i].imshow(img_array.astype('uint8'))
            ax.ravel()[i].set_title(title_label, color=color)
            ax.ravel()[i].set_axis_off()
    else:
        for i, image_filepath in enumerate(images_filepaths):
            image = cv2.imread(image_filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if predicted_labels is not None and true_labels is not None:
                predicted_label = predicted_labels[i]
                true_label = true_labels[i]
                color = "green" if true_label == predicted_label else "red"
                title_label = str(predicted_label)
            elif predicted_labels is not None:
                predicted_label = str(int(predicted_labels[i]))
                title_label = predicted_label
            elif true_labels is not None:
                true_label = true_labels[i]
                title_label = str(int(true_label))
            else:
                raise RuntimeError

            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_title(title_label, color=color)
            ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


########################################################################################

def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    :param net: NN
    :type: torch.nn.Module
    :param images: image tensor
    :type: torch.Tensor
    :param labels: True labels
    :param classes: should be format like classes = ('cat', 'dog', 'bird')

    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
