import numpy as np
import math


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


def im_show(img: torch.Tensor):
    """
    show image use matplotlib
    """
    # TODO: unnormalize
    # img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
