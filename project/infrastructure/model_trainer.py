import abc
from collections import defaultdict
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import psutil
from tqdm import tqdm

import project.infrastructure.pytorch_util as ptu


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model(nn.Module, metaclass=abc.ABCMeta):
    """ simple model trainer """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Param Dict
        self.params = None

        # Dataset
        self.train_dataset = None
        self.valid_dataset = None

        # Sampler
        self.train_sampler = None
        self.valid_sampler = None

        # Data Loader
        self.train_loader = None
        self.valid_loader = None

        # Optimizer
        self.optimizer = None
        self.scheduler = None

        # history record
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0

        # GPU or CPU
        self.device = None
        self.num_workers = None

        # Use automatic mixed precision training in GPU
        self.fp16 = False
        self.scaler = None

        # Timer
        self.start_time = None
        self.end_time = None

        self.metrics = {"train": {}, "valid": {}, "test": {}}

    #######################################################

    def init_trainer(self, params: dict):
        """ Init Trainer"""
        self.params = params

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # init gpu
        self.params['use_gpu'] = not self.params['no_gpu']
        ptu.init_gpu(
            use_gpu=self.params['use_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.device = ptu.device
        print(f"############ {self.device} ############")

        self.fp16 = params["fp16"]
        if self.fp16 and self.params['use_gpu']:
            self.scaler = torch.cuda.amp.GradScaler()

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

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        return super().forward(*args, **kwargs)

    def model_fn(self, batch):
        data, targets = batch
        data = data.to(device=self.device, non_blocking=True)
        targets = targets.to(device=self.device, non_blocking=True)

        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss = self(data, targets)
        else:
            output, loss = self(data, targets)

        metrics = self.monitor_metrics(output, targets)
        return output, loss, metrics

    #####################################################################
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
            use_fp16=False
    ):
        """ fit the model """
        self.fp16 = use_fp16

        # use all workers if -1
        self.num_workers = psutil.cpu_count() if num_workers == -1 else num_workers
        pin_memory = True if torch.cuda.is_available() else False

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                batch_sampler=None,
                num_workers=self.num_workers,
                collate_fn=None,
                pin_memory=pin_memory
            )

        if self.valid_loader is None and valid_dataset is not None:
            self.valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,  # not going to shuffle data for validation
                sampler=valid_sampler,
                batch_sampler=None,
                num_workers=self.num_workers,
                collate_fn=None,
                pin_memory=pin_memory
            )

        self.device = device
        if next(self.parameters()).device != device:
            self.to(device)

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        # device indicator for progress bar
        if torch.cuda.is_available():
            DEVICE = 'AMP' if self.fp16 else 'cuda'
        else:
            DEVICE = 'cpu'

        history = defaultdict(list)
        self.start_time = time.time()
        n_epochs_loop = tqdm(range(max_epochs), desc="LDC", leave=True)
        for _ in n_epochs_loop:

            # Training Phase
            train_loss, _ = self.train_one_epoch(self.train_loader)
            train_loss = np.mean(train_loss)
            history["train_loss"].append(train_loss)

            # Validation phase
            if self.valid_loader:
                val_loss, val_metrics = self.validate_one_epoch(self.valid_loader)
                val_loss = np.mean(val_loss)
                history["val_loss"].append(val_loss)

                for k, v in val_metrics.items():
                    val_metrics[k]= np.mean(v)
                    history[k].append(v)
            else:
                val_loss = {}
                val_metrics = {}

            # update progress bar
            description = f'({DEVICE}) epoch: {self.current_epoch}'
            n_epochs_loop.set_description(description)
            n_epochs_loop.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss,
                **val_metrics
            )
            self.current_epoch += 1
        self.end_time = time.time() - self.start_time
        return history

    def train_one_epoch(self, train_loader):
        """ train_one_epoch """
        self.train()
        train_losses = []
        train_metrics = None
        # train_metrics = defaultdict(list)

        # tk0 = tqdm(train_loader, total=len(train_loader), leave=False)
        tk0 = train_loader
        for batch_idx, train_batch in enumerate(tk0):
            # train_batch is a tuple: (data, targets)
            loss, metrics = self.train_one_step(train_batch)
            train_losses.append(ptu.to_numpy(loss))
            # for k, v in metrics.items():
            #     train_metrics[k].append(v)
            # self.current_train_step += 1
        return train_losses, train_metrics

    def train_one_step(self, batch):
        """ take one gradient step """
        loss: torch.FloatTensor
        metrics: dict

        self.optimizer.zero_grad()
        _, loss, metrics = self.model_fn(batch)

        with torch.set_grad_enabled(True):
            # USE AUTOMATIC MIXED PRECISION
            if self.fp16 and torch.cuda.is_available():
                assert self.scaler is not None, "amp.GradScaler() is not init"
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        return loss, metrics

    def monitor_metrics(self, *args, **kwargs):
        """ show metrics"""
        return

    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def validate_one_epoch(self, valid_loader):
        self.eval()
        metrics: dict
        val_losses = []
        val_metrics = defaultdict(list)

        # tk0 = valid_loader
        # tk0 = tqdm(valid_loader, total=len(valid_loader))
        tk0 = valid_loader
        for batch_idx, batch in enumerate(tk0):
            with torch.no_grad():
                loss, metrics = self.validate_one_step(batch)
            val_losses.append(ptu.to_numpy(loss))
            for k, v in metrics.items():
                val_metrics[k].append(v)
            self.current_valid_step += 1
        return val_losses, val_metrics

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output
