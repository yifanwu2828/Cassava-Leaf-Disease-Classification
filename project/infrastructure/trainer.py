from typing import Any, Dict, Iterable, List, Optional, Union

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

import project.infrastructure.pytorch_util as ptu
from logger import Logger


class DL_Trainer(object):
    """ DL_Trainer """

    def __init__(self, params: dict):

        #############
        # INIT
        #############

        self.params = params
        # self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed: int = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # init gpu
        self.params['use_gpu'] = not self.params['no_gpu']
        ptu.init_gpu(
            use_gpu=self.params['use_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.device: torch.device = ptu.device

        ################
        ### Init Data
        ################
        self.train_loader = None
        self.val_loader = None

        ################
        ### Init Model
        ################
        model_class = self.params['model_class']
        # TODO: define class
        # self.model = model_class(self.params['model_params'])
        self.model = self.params['model']
        #####################
        ### Training Setting
        ####################
        self.max_epochs = params["max_epochs"]
        if params["min_epochs"] is not None:
            self.min_epochs = params["min_epochs"]
        else:
            self.min_epochs = 1

        self.criterion = params["criterion"]
        self.optimizer = params["optimizer"]
        self.scheduler = params["scheduler"]

        # Logging Flag
        self.log_video = None
        self.log_metrics = None

    ##################################

    def __repr__(self) -> str:
        return f"DL_Trainer"

    ##################################

    def fit(self, model, train_dataset, batch_size, shuffle=True, device='cpu'):
        # Create DataLoader if it has not been created
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
        # Send model to proper device
        if next(model.parameters()).device != device:
            model.to(device)

        # Start training loop
        self.train(model, self.train_loader)

    def train(self, train_dataloader):
        for epoch in range(self.max_epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                train_step_loss=self.training_one_step(batch_idx,
                                                       batch,
                                                       device=self.device
                                                       )

    def training_one_step(self, batch_idx, batch, device='cpu'):
        self.optimizer.zero_grad()
        # for k, v in batch.items():
        #     batch[k] = v.to(device)
        # model forward function
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {"Train_Loss": loss}

