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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

from project.infrastructure.model_trainer import Model
import project.infrastructure.utils as utils


class MyModel(Model):
    """ inherent from Model (subclass of nn.Module)"""
    def __init__(self, params: dict):
        super().__init__()
        self.params = params

        self.conv1 = nn.Conv2d(
            in_channels=params["input_size"],
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, params["output_size"])

    def fetch_optimizer(self):
        """set optimizer"""
        opt = optim.Adam(self.parameters(), lr=self.params["learning_rate"])
        return opt

    @staticmethod
    def calc_loss(outputs, targets, criterion=None):
        """ calculate loss """
        if targets is None or criterion is None:
            print("Targets is None or Criterion is None")
            return None
        return criterion(outputs, targets)

    def forward(self, x, targets=None):
        """
        forward function return logits and loss
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)

        criterion = nn.CrossEntropyLoss()
        loss = self.calc_loss(out, targets, criterion)

        return out, loss

    def check_accuracy(self, loader):
        """ simple acc check"""
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        num_correct = 0
        mum_samples = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores, _ = self(x, y)
                # 64x10
                _, preds = scores.max(1)
                num_correct += torch.sum((preds == y), -1)
                mum_samples += preds.size(0)
            accuracy = float(num_correct) / float(mum_samples)
            print(f"Got {num_correct} / {mum_samples} with accuracy: {accuracy * 100: .2f}%")
        self.train()
        return accuracy
##################################################################################################


def main():
    """
    main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-exp', type=str, default='exp_1')
    parser.add_argument('--data_dir', '-data', type=str, default='../../data')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument(
        '--max_epochs', '-n', type=int, default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--min_epochs', '-min', type=int, default=1,
        help='Number of epochs'
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=64,
        help='Size of train batches'
    )
    parser.add_argument(
        '--valid_batch_size', type=int, default=128,
        help='Size of valid batches'
    )

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--video_log_freq', type=int, default=-1)  # -1 not log video
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true', default=False)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../log_dir')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    # RUN TRAINING    #
    ###################
    print("###### PARAM ########")
    params["train_batch_size"]= 64
    params['fp16'] = False
    print(params)

    path = os.getcwd()
    print(f"Working Dir:{path}")

    # TODO: run training loop
    # load data
    train_dataset = datasets.MNIST(
        root=params["data_dir"],
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = datasets.MNIST(
        root=params["data_dir"],
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # Init GPU if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(device)

    m = MyModel(params)
    m.init_trainer(params)
    m.fit(train_dataset=train_dataset, train_batch_size=params["train_batch_size"],
          valid_dataset=test_dataset, valid_batch_size=params["valid_batch_size"],
          max_epochs=params["max_epochs"], device=device,
          num_workers=6, use_fp16=params['fp16'],
          )
    m.check_accuracy(m.train_loader)
    m.check_accuracy(m.valid_loader)


if __name__ == '__main__':
    print(torch.__version__)
    torch.backends.cudnn.benchmark = True
    start_time = utils.tic("###### Start Training ######")
    main()
    print("Done!")
    utils.toc(start_time, "Finish Training")
