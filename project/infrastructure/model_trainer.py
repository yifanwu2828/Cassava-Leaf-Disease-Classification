import abc
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


class Model(nn.Module, metaclass=abc.ABCMeta):
    """ simple model trainer """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data Loader
        self.train_loader = None
        self.valid_loader = None

        # Optimizer
        self.optimizer = None
        self.scheduler = None

        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0

        self.device = None
        self.fp16 = False

    #######################################################
    @abc.abstractmethod
    def fetch_optimizer(self, *args, **kwargs):
        """
        must overwrite define optimizer
        """
        return

    def fetch_scheduler(self, *args, **kwargs):
        """
        overwrite it if use LR scheduler
        """
        return

    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        super().forward(*args, **kwargs)
        raise NotImplementedError

    def fit(
            self,
            train_dataset,
            train_batch_size,
            valid_dataset,
            valid_batch_size,
            max_epochs,
            device,
            train_sampler=None,
            valid_sampler=None,
            shuffle=True,
            num_workers=4,
            fp16=False
    ):
        """ fit the model """
        if self.train_loader is None:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                batch_sampler=None,
                num_workers=num_workers,
                collate_fn=None,
                pin_memory=False
            )

        if self.valid_loader is None and valid_dataset is not None:
            self.valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,  # not going to shuffle data for validation
                sampler=valid_sampler,
                batch_sampler=None,
                num_workers=num_workers,
                collate_fn=None,
                pin_memory=False
            )
        self.device = device
        if next(self.parameters()).device != device:
            self.to(device)

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        for epoch in tqdm(range(max_epochs)):
            train_epoch_loss = self.train_one_epoch(self.train_loader)

    def train_one_epoch(self, data_loader):
        """ train_one_epoch """
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            # batch is a tuple: (data, targets)
            step_loss = self.train_one_step(batch)
            epoch_loss += step_loss
        return epoch_loss / (batch_idx + 1)

    def train_one_step(self, batch):
        """ take one gradient step """
        self.optimizer.zero_grad()

        data, targets = batch
        data = data.to(device=self.device)

        targets = targets.to(device=self.device)
        _, loss = self.forward(data, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss


###########################################

class MyModel(Model):
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


if __name__ == '__main__':
    # load data
    train_dataset = datasets.MNIST(
        root='../../data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = datasets.MNIST(
        root='../../data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # Init GPU if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(device)

    params = {
        "input_size": 1,
        "output_size": 10,
        "learning_rate": 1e-3,
        "train_batch_size": 64,
        "valid_batch_size": 128,
        "max_epochs": 3,
    }
    m = MyModel(params)
    m.fit(train_dataset=train_dataset, train_batch_size=params["train_batch_size"],
          valid_dataset=test_dataset, valid_batch_size=params["valid_batch_size"],
          max_epochs=params["max_epochs"], device=device
          )
    m.check_accuracy(m.train_loader)
    m.check_accuracy(m.valid_loader)
