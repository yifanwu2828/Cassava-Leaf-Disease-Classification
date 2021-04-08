import argparse
import sys
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from sklearn import metrics, model_selection

from tqdm import tqdm
import albumentations
import matplotlib.pyplot as plt
import pandas as pd

from project.infrastructure.model_trainer import Model
from project.infrastructure.img_dataset import ImageDataset
import project.infrastructure.utils as utils


if __name__ == '__main__':
    df = pd.read_csv("../../data/train.csv")
