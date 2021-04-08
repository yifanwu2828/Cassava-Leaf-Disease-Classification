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
        super().__init__(*args, **kwargs)

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

    #######################################################

    def init_trainer(self, params: dict):
        """ Init"""
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

    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        return super().forward(*args, **kwargs)

    def model_fn(self, batch):
        data, targets = batch
        data = data.to(device=self.device)
        targets = targets.to(device=self.device)

        metrics = None
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss = self(data, targets)
        else:
            output, loss = self(data, targets)
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
        if num_workers == -1:
            num_workers = psutil.cpu_count()
        self.num_workers = num_workers
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

        self.start_time = time.time()
        n_epochs_loop = tqdm(range(max_epochs), desc="LDC", leave=True)
        for epoch in n_epochs_loop:
            train_epoch_loss = self.train_one_epoch(self.train_loader)

            # update progress bar
            DEVICE = 'AMP' if self.fp16 else 'cuda'
            description = f'({DEVICE}) epoch {epoch} loss: {train_epoch_loss:.4f}'
            n_epochs_loop.set_description(description)
        self.end_time = time.time() - self.start_time

    def train_one_epoch(self, train_loader):
        """ train_one_epoch """
        self.train()
        losses = AverageMeter()
        metrics_meter = None

        for batch_idx, train_batch in enumerate(train_loader):
            # train_batch is a tuple: (data, targets)
            loss, metrics = self.train_one_step(train_batch)
            losses.update(loss.item(), train_loader.batch_size)

            # if batch_idx == 0:
            #     metrics_meter = {k: AverageMeter() for k in metrics}
            #
            # monitor = {}
            # for m_m in metrics_meter:
            #     metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
            #     monitor[m_m] = metrics_meter[m_m].avg
            # self.current_train_step += 1
            # tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
            # tk0.close()
        return losses.avg

    def train_one_step(self, batch):
        """ take one gradient step """
        self.optimizer.zero_grad()
        _, loss, metrics= self.model_fn(batch)

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

    def validate_one_epoch(self, data_loader):
        self.eval()
        losses = AverageMeter()
        for batch_idx, batch in enumerate(data_loader):
            with torch.no_grad():
                loss, metrics = self.validate_one_step(batch)
            losses.update(loss.item(), data_loader.batch_size)

        return losses.avg

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

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
