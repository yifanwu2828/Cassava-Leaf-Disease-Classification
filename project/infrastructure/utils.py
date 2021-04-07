import numpy as np
import math


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
