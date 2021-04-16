import abc
from collections import defaultdict
from typing import Tuple, List, Dict, Union, Optional, Any
import os
import time
import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
import psutil
from tqdm import tqdm

import project.infrastructure.pytorch_util as ptu
import project.infrastructure.utils as utils



class Model(nn.Module, metaclass=abc.ABCMeta):
    """ simple model trainer """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Param Dict
        self.params: Optional[dict] = None
        self.writer = None
        self.exp_name = None

        # Dataset
        self.train_dataset: Union[Dataset, Any] = None
        self.valid_dataset: Union[Dataset, Any] = None

        # Sampler
        self.train_sampler: Optional[Sampler] = None
        self.valid_sampler: Optional[Sampler] = None

        # Data Loader
        self.train_loader: DataLoader = None
        self.valid_loader: DataLoader = None

        # Optimizer
        self.optimizer = None
        # TODO: change self.clipping if want to modify gradients
        self.clip_grad = False
        self.scheduler: Optional[Any] = None

        # loss function
        self.criterion = None

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

        # Phase
        self.phase: str = ''

        # Timer
        self.epoch_start: float = None
        self.duration: float = None
        self.since: float = None
        self.runtime: float = None

        # Metrics
        self.history = None

    #######################################################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    #######################################################

    def init_trainer(self, params: dict) -> None:
        """ Init Trainer"""
        self.params: dict = params
        self.exp_name = params.get('exp', 'my_experiment')

        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../runs')

        if not (os.path.exists(log_path)):
            os.makedirs(log_path)

        logdir = self.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(log_path, logdir)

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        log_dir = params.get('log_dir', logdir)

        self.writer = SummaryWriter(log_dir=log_dir, comment=self.exp_name)

        # Set random seeds
        seed: int = self.params.get('seed', 42)
        utils.set_random_seed(seed)

        # init gpu
        self.params['use_gpu']: bool = not self.params['no_gpu']
        ptu.init_gpu(
            use_gpu=self.params['use_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.device: torch.device = ptu.device

        # self.fp16: bool = params["fp16"]
        self.fp16: bool = params.get("fp16", False)
        if self.fp16 and self.params['use_gpu']:
            self.scaler = torch.cuda.amp.GradScaler()
        print(f"############ {self.device}: AMP={self.fp16} ############")

        self.optimizer = params.get("optimizer")
        self.scheduler = params.get("scheduler")
        self.criterion = params.get("criterion")
    #######################################################

    @abc.abstractmethod
    def config_optimizer(self, *args, **kwargs):
        """
        must overwrite define optimizer
        """
        return

    def config_scheduler(self, *args, **kwargs) -> Optional[Any]:
        """
        overwrite it if use LR scheduler
        """
        return

    def config_criterion(self, *args, **kwargs) -> Optional[Any]:
        """
        must overwrite define optimizer
        """
        return

    #######################################################

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        return super().forward(*args, **kwargs)

    def model_fn(self, batch: Union[Dict, List, Tuple]) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict]:
        """
        Forward function of the model
        :param batch: one batch of data contain
        :return: output, loss, metrics
        """
        assert len(batch) == 1 or len(batch) == 2, "batch should be a pair for supervised learning"
        data: torch.FloatTensor
        targets: Optional[torch.Tensor] = None
        output: torch.FloatTensor
        loss: Optional[torch.FloatTensor]
        metrics: Optional[dict] = {}

        # if batch is a dictionary
        if isinstance(batch, dict):
            # adjust for different name assign for keys
            name = tuple(batch.keys())
            # unsupervised learning has no target
            if len(name) == 1:
                data = batch[name[0]].to(self.device, non_blocking=True)
            # supervised learning (data, target) pair
            elif len(name) == 2:
                data = batch[name[0]].to(self.device, non_blocking=True)
                targets = batch[name[1]].to(self.device, non_blocking=True)
            else:  # more elements
                raise NotImplementedError("Check if more elements Needed")

        # batch is a List or Tuple
        elif isinstance(batch, tuple) or isinstance(batch, list):
            data, targets = batch
            data = data.to(device=self.device, non_blocking=True)
            targets = targets.to(device=self.device, non_blocking=True)
        else:
            raise TypeError("Implemented batch contain type other than Tuple, List, Dict  ")

        #####################################################################################

        """
        This method is prefer but in case of pytorch < 1.6, torch.cuda.amp is not defined
        '''
           `enabled`, an optional convenience argument to autocast and GradScaler. 
           If False, autocast and GradScaler’s calls become no-ops. 
           This allows switching between default precision and mixed precision without if/else statements.
        '''
        """
        # amp / (cuda or cpu)
        # with torch.cuda.amp.autocast(enabled=self.fp16):
        #     output = self(data)
        #     loss = self.loss_fn(output, targets)

        # amp
        if self.fp16:
            with torch.cuda.amp.autocast():
                output = self(data)
                loss = self.loss_fn(output, targets)
        # cuda or cpu
        else:
            output = self(data)
            loss = self.loss_fn(output, targets)

        # Record metrics if has target
        if targets is not None:
            metrics = self.monitor_metrics(output, targets)
            if self.phase == 'train':
                try:
                    metrics['train_acc'] = metrics.pop("acc")
                except KeyError:
                    pass

            elif self.phase == 'eval':
                try:
                    metrics['val_acc'] = metrics.pop("acc")
                except KeyError:
                    pass

        return output, loss, metrics

    #####################################################################

    def loss_fn(self, *args, **kwargs):
        """
        calculate loss
          overwrite this function to computer a much complex loss function
          template:
          if targets is None or self.criterion is None:
            print("Targets is None or Criterion is None")
            return None
        return self.criterion(outputs, targets)
        """
        return

    #####################################################################
    def fit(
            self,
            train_dataset: Dataset,
            train_batch_size: int,
            valid_dataset: Dataset,
            valid_batch_size: int,
            max_epochs: int,
            device: torch.device,
            train_sampler: Optional[Sampler] = None,
            valid_sampler: Optional[Sampler] = None,
            shuffle: bool = True,
            num_workers: int = 4,
            use_fp16: bool = False,
            save_best: bool = False,
            better_than: float = 0.8
    ):
        """ fit the model """
        assert isinstance(max_epochs, int)
        assert isinstance(train_batch_size, int)
        assert isinstance(valid_batch_size, int)
        assert isinstance(num_workers, int)
        assert isinstance(better_than, float)
        assert isinstance(device, torch.device)
        assert max_epochs > 0
        assert train_batch_size > 0
        assert valid_batch_size > 0
        assert 0 <= better_than <= 1  # if acc is better than float(better_than) save the model
        assert num_workers >= -1

        self.fp16: int = use_fp16
        self.since = time.time()

        # use all workers if -1
        self.num_workers: int = psutil.cpu_count() if num_workers == -1 else num_workers
        pin_memory: bool = True if torch.cuda.is_available() else False

        # Create Training Dataloader
        if self.train_loader is None and train_dataset is not None:
            print("\nCreating Training Dataloader...")
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                batch_sampler=None,
                num_workers=self.num_workers,
                collate_fn=None,
                pin_memory=pin_memory
            )
        else:
            raise ValueError("Please provide training dataset")

        # Create Validation Dataloader
        if self.valid_loader is None and valid_dataset is not None:
            print("\nCreating Validation Dataloader...")
            self.valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,  # not going to shuffle data for validation
                sampler=valid_sampler,
                batch_sampler=None,
                num_workers=self.num_workers,
                collate_fn=None,
                pin_memory=pin_memory
            )

        # If model is not on correct device send it to device
        self.device: torch.device = device
        if next(self.parameters()).device != device:
            self.to(device)

        # Fetch optimizer
        if self.optimizer is None:
            try:
                self.optimizer = self.config_optimizer()
            except NotImplementedError:
                raise NotImplementedError("Optimizer is not implemented")

        # Fetch scheduler if provided
        if self.scheduler is None:
            assert self.optimizer is not None, "Please set up optimizer first"
            self.scheduler: Optional[Any] = self.config_scheduler()

        # Fetch criterion if provided (if supervised learning)
        if self.criterion is None:
            self.criterion: Optional[Any] = self.config_criterion()

        # Device indicator for progress bar
        global DEVICE
        if torch.cuda.is_available():
            DEVICE = 'AMP' if self.fp16 else 'cuda'
        else:
            DEVICE = 'cpu'

        # Used for saving best model
        history: Dict[str, List] = defaultdict(list)
        best_acc: float = 0.0


        n_epochs_loop = tqdm(range(max_epochs), desc="LDC", leave=True)
        for itr in n_epochs_loop:
            self.epoch_start: float = time.time()
            # Each epoch has a training and validation phase
            train_epoch_loss: Union[torch.FloatTensor, np.ndarray]
            val_epoch_loss: Union[torch.FloatTensor, np.ndarray, dict]
            val_metrics: dict

            # Training Phase
            self.phase = 'train'
            train_epoch_loss, train_metrics = self.train_one_epoch(self.train_loader)

            # calculate mean of metrics
            # TODO: adjust in case metrics cannot take average
            avg_train_epoch_loss = np.mean(train_epoch_loss)
            history["train_loss"].append(avg_train_epoch_loss)

            # Record metrics
            for k, v in train_metrics.items():
                train_metrics[k] = np.mean(v)
                history[k].append(v)
            avg_train_epoch_acc: Union[Dict, np.ndarray] = train_metrics.get('train_acc', {})

            # Validation phase
            self.phase = 'eval'
            if self.valid_loader:
                val_epoch_loss, val_metrics = self.validate_one_epoch(self.valid_loader)
                avg_val_epoch_loss = np.mean(val_epoch_loss)
                history["val_loss"].append(avg_val_epoch_loss)

                for k, v in val_metrics.items():
                    val_metrics[k] = np.mean(v)
                    history[k].append(v)
                avg_val_epoch_acc: Union[Dict, np.ndarray] = val_metrics.get('val_acc', {})

                # log Training and Validation loss to Tensorboard
                if self.writer is not None:
                    self.writer.add_scalars(
                        'Epoch',
                        {
                            'Training_Loss': avg_train_epoch_loss,
                            'Training_Acc': avg_train_epoch_acc,
                            'Validation_Loss': avg_val_epoch_loss,
                            'Validation_Acc': avg_val_epoch_acc,
                        }, self.current_epoch + 1
                    )
                    # Call this method to make sure that all pending events have been written to disk.
                    self.writer.flush()


                # TODO: deep copy model every time meet max acc is not efficient
                # The best accuracy is 1
                if save_best and avg_val_epoch_acc is not None:
                    if isinstance(avg_val_epoch_acc, dict):
                        epoch_acc = avg_val_epoch_acc.values()
                    else:
                        assert avg_val_epoch_acc.size == 1, "Compare multiply array value with float is ambiguous"
                        epoch_acc = float(avg_val_epoch_acc)

                    acc_threshold = avg_val_epoch_acc >= better_than
                    if epoch_acc > best_acc and acc_threshold:
                        best_acc = epoch_acc
                        # best_model_wts = copy.deepcopy(self.state_dict())

                        # Save best model
                        PATH = f"../model/test_ldc.pth"
                        torch.save(
                            {
                                'epoch': itr,
                                'model_state_dict': self.state_dict(),
                                'acc': epoch_acc,
                            }, PATH
                        )
            else:
                print("\nPlease provide Validation Dataset...")
                avg_val_epoch_loss: dict = {}
                val_metrics: dict = {}

            self.current_epoch += 1
            self.duration: float = time.time() - self.epoch_start

            # Update progress bar
            description: str = f'({DEVICE}) Epoch: {self.current_epoch}'
            n_epochs_loop.set_description(description)
            n_epochs_loop.set_postfix(
                duration = self.duration,
                train_loss=avg_train_epoch_loss,
                val_loss=avg_val_epoch_loss,
                **train_metrics,
                **val_metrics,
            )

        history["epoch"].append(list(range(1, max_epochs+1)))
        history["duration"].append(self.duration)
        self.history = history
        self.runtime = time.time() - self.since
        print(f"\nFinish Training in {self.runtime} sec")
        return history

    def train_one_epoch(self, train_loader) -> Tuple[List, Optional[dict]]:
        """ train_one_epoch """
        assert self.phase == 'train', "self.phase is not 'train' in training one epoch"
        self.train()
        # print('\nTrain One Epoch...')

        train_losses: List = []
        train_metrics: Optional[dict] = defaultdict(list)

        tk0 = tqdm(train_loader, total=len(train_loader),
                   desc=f"Train One Epoch: {self.current_epoch}",
                   leave=False, disable=True)
        for batch_idx, train_batch in enumerate(tk0):
            # train_batch is a tuple: (data, targets)
            loss, metric = self.train_one_step(train_batch)
            train_losses.append(ptu.to_numpy(loss))
            for k, v in metric.items():
                train_metrics[k].append(v)
            self.current_train_step += 1
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
        # # or maybe equivalently and concisely (only available at later version)
        # self.optimizer.zero_grad(set_to_none=True)

        _, loss, metrics = self.model_fn(batch)

        with torch.set_grad_enabled(True):
            assert self.phase == 'train'
            # USE AUTOMATIC MIXED PRECISION
            if self.fp16 and torch.cuda.is_available():
                assert self.scaler is not None, "torch.cuda.amp.GradScaler is not init"
                self.scaler.scale(loss).backward()

                if self.clip_grad:
                    # Unscales the gradients of optimizer's assigned params in-place
                    # Note: unscale_ should only be called once per optimizer per step call,
                    self.scaler.unscale_(self.optimizer)
                    print("here at self.train_one_step/clip_grad")
                    # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                    # You may use the same value for max_norm here as you would without gradient scaling.
                    # TODO: change max_norm
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)

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
        assert self.phase == 'eval', "self.phase is not 'eval' in validate one epoch"
        self.eval()

        metrics: dict
        val_losses: List = []
        val_metrics: Dict[str, list] = defaultdict(list)

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
