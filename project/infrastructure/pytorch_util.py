from typing import Union

import numpy as np
import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(input_size: int,
              output_size: int,
              n_layers: int,
              size: int,
              activation: Activation = 'relu',
              output_activation: Activation = 'relu',
              ) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size: int = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


#####################################################
#####################################################
# Best practice: copy this func under __init__ of custom module and change model to self and remove return
def initialize_weights(model: torch.nn.Module):
    """weight initialization"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # default Leaky relu
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

#####################################################
#####################################################

device = None


def init_gpu(use_gpu=True, gpu_id=0) -> None:
    """ init device('cuda:0' or 'cpu') """
    # device is global, call ptu.device to retrieve
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(device: Union[torch.device, int]) -> None:
    """
    :type device: torch.device
    """
    torch.cuda.set_device(device)


#####################################################
#####################################################

def from_numpy(*args, **kwargs) -> torch.FloatTensor:
    """ Convert numpy array to torch tensor  and send to device('cuda:0' or 'cpu') """
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor: Union[torch.FloatTensor, torch.Tensor]) -> np.ndarray:
    """ Convert torch tensor to numpy array and send to CPU """
    return tensor.to('cpu').detach().numpy()

