import abc
from collections import defaultdict
from typing import Tuple, List, Dict, Union, Optional, Any
import time

import numpy as np
import torch
import torch.nn as nn
import psutil
from tqdm import tqdm

import project.infrastructure.pytorch_util as ptu


class Model(nn.Module, metaclass=abc.ABCMeta):
    """ simple model trainer """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Param Dict
        self.params: Optional[dict] = None

        # Dataset
        self.train_dataset: Union[torch.utils.data.Dataset, Any] = None
        self.valid_dataset: Union[torch.utils.data.Dataset, Any] = None

        # Sampler
        self.train_sampler: Optional[torch.utils.data.Sampler] = None
        self.valid_sampler: Optional[torch.utils.data.Sampler] = None

        # Data Loader
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None

        # Optimizer
        self.optimizer = None
        self.scheduler: Optional[Any] = None

        # history record
        self.current_epoch: int = 0
        self.current_train_step: int = 0
        self.current_valid_step: int = 0

        # GPU or CPU
        self.device: torch.device = None
        self.num_workers: int = None

        # Use automatic mixed precision training in GPU
        self.fp16: bool = False
        self.scaler: torch.cuda.amp.GradScaler = None

        # Timer
        self.start_time: float = None
        self.end_time: float = None

        # Metrics
        # TODO: record metrics as class attribute
        self.metrics: dict = {"train": {}, "valid": {}, "test": {}}

    #######################################################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    #######################################################

    def init_trainer(self, params: dict) -> None:
        """ Init Trainer"""
        self.params: dict = params

        # Set random seeds
        seed: int = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # init gpu
        self.params['use_gpu']: bool = not self.params['no_gpu']
        ptu.init_gpu(
            use_gpu=self.params['use_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.device: torch.device = ptu.device

        self.fp16: bool = params["fp16"]
        if self.fp16 and self.params['use_gpu']:
            self.scaler = torch.cuda.amp.GradScaler()

        print(f"############ {self.device}: AMP={self.fp16} ############")

    #######################################################

    @abc.abstractmethod
    def config_optimizer(self, *args, **kwargs):
        """
        must overwrite define optimizer
        """
        return

    def config_scheduler(self, *args, **kwargs):
        """
        overwrite it if use LR scheduler
        """
        assert self.optimizer is not None, "Please set up optimizer first"
        return

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        return super().forward(*args, **kwargs)

    def model_fn(
            self,
            batch: Union[Tuple, Dict]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict]:

        assert len(batch) == 1 or len(batch) == 2,  "batch should be a pair for supervised learning"
        output: torch.FloatTensor
        loss: torch.FloatTensor
        metrics: dict = {}
        target: Optional[torch.Tensor] = None

        # TODO: fix the param to forward
        # if batch is a dictionary
        if isinstance(batch, dict):
            # adjust for different name assign for keys
            name = tuple(batch.keys())
            # unsupervised learning has no target
            if len(name) == 1:
                data = batch[name[0]].to(self.device, non_blocking=True)
            # supervised learning (data, target) pair
            else:  # len(name) == 2
                data = batch[name[0]].to(self.device, non_blocking=True)
                targets = batch[name[1]].to(self.device, non_blocking=True)
            # amp
            if self.fp16:
                with torch.cuda.amp.autocast():
                    output, loss = self(data, targets)
            # cuda or cpu
            else:
                output, loss = self(data, targets)

        else:
            # batch is a List or Tuple
            data, targets = batch
            data = data.to(device=self.device, non_blocking=True)
            targets = targets.to(device=self.device, non_blocking=True)

            # amp
            if self.fp16:
                with torch.cuda.amp.autocast():
                    output, loss = self(data, targets)
            # cuda or cpu
            else:
                output, loss = self(data, targets)

        # Record metrics if has target

        metrics = self.monitor_metrics(output, targets)

        print(metrics)
        return output, loss, metrics

    #####################################################################
    def fit(
            self,
            train_dataset,
            train_batch_size: int,
            valid_dataset,
            valid_batch_size: int,
            max_epochs: int,
            device: torch.device,
            train_sampler: Optional[torch.utils.data.Sampler] = None,
            valid_sampler: Optional[torch.utils.data.Sampler] = None,
            shuffle: bool = True,
            num_workers: int = 4,
            use_fp16: bool = False
    ):
        """ fit the model """
        self.fp16: int = use_fp16

        # use all workers if -1
        self.num_workers: int = psutil.cpu_count() if num_workers == -1 else num_workers
        pin_memory: bool = True if torch.cuda.is_available() else False

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

        self.device: torch.device = device
        if next(self.parameters()).device != device:
            self.to(device)

        if self.optimizer is None:
            self.optimizer = self.config_optimizer()

        if self.scheduler is None:
            assert self.optimizer is not None, "Please set up optimizer first"
            self.scheduler: Optional[Any] = self.config_scheduler()

        # device indicator for progress bar
        DEVICE: str
        if torch.cuda.is_available():
            DEVICE = 'AMP' if self.fp16 else 'cuda'
        else:
            DEVICE = 'cpu'

        history: Dict[str, List] = defaultdict(list)
        self.start_time: float = time.time()
        n_epochs_loop = tqdm(range(max_epochs), desc="LDD", leave=True)
        for _ in n_epochs_loop:
            train_loss: Union[torch.FloatTensor, np.ndarray]
            val_loss: Union[torch.FloatTensor, np.ndarray, dict]
            val_metrics: dict

            # Training Phase
            train_loss, _ = self.train_one_epoch(self.train_loader)
            train_loss: np.ndarray = np.mean(train_loss)
            history["train_loss"].append(train_loss)

            # Validation phase
            if self.valid_loader:
                val_loss, val_metrics = self.validate_one_epoch(self.valid_loader)
                val_loss: np.ndarray = np.mean(val_loss)
                history["val_loss"].append(val_loss)

                for k, v in val_metrics.items():
                    val_metrics[k]= np.mean(v)
                    history[k].append(v)
            else:
                val_loss: dict = {}
                val_metrics: dict = {}

            # update progress bar
            description: str = f'({DEVICE}) epoch: {self.current_epoch}'
            n_epochs_loop.set_description(description)
            n_epochs_loop.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss,
                **val_metrics
            )
            self.current_epoch += 1
        self.end_time: float = time.time() - self.start_time
        return history

    def train_one_epoch(self, train_loader) -> Tuple[List, Optional[dict]]:
        """ train_one_epoch """
        self.train()
        train_losses: List = []
        train_metrics: Optional[dict] = None
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

    def train_one_step(self, batch) -> Tuple[torch.FloatTensor, Dict]:
        """ take one gradient step """
        loss: torch.FloatTensor
        metrics: dict

        '''
        The second code snippet does not zero the memory of each individual parameter
        also the subsequent backward pass uses assignment instead of addition to store gradients,
        this reduces the number of memory operations.
        Setting gradient to None has a slightly different numerical behavior than setting it to zero
        '''
        # 1st most comment way: set grad to zero grad
        # self.optimizer.zero_grad()

        # 2nd faster way: set grad to None
        for param in self.parameters():
            param.grad = None

        _, loss, metrics = self.model_fn(batch)

        with torch.set_grad_enabled(True):
            # USE AUTOMATIC MIXED PRECISION
            if self.fp16 and torch.cuda.is_available():
                assert self.scaler is not None, "torch.cuda.amp.GradScaler is not init"
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # if scheduler is used
            if self.scheduler is not None:
                self.scheduler.step()

        return loss, metrics

    def monitor_metrics(self, *args, **kwargs):
        """ show metrics"""
        return

    def validate_one_step(self, data) -> Tuple[torch.FloatTensor, Dict]:
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def validate_one_epoch(self, valid_loader) -> Tuple[List, Dict]:
        self.eval()
        metrics: dict
        val_losses: List = []
        val_metrics: Dict[str, list] = defaultdict(list)

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

    def predict_one_step(self, data) -> torch.FloatTensor:
        output, _, _ = self.model_fn(data)
        return output
