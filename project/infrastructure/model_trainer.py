import abc
from collections import defaultdict
from typing import Tuple, List, Dict, Union, Optional, Any
import time
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
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

        try:
            self.optimizer = params["optimizer"]
        except KeyError:
            pass
        try:
            self.scheduler = params["scheduler"]
        except KeyError:
            pass
        try:
            self.criterion = params["criterion"]
        except KeyError:
            pass

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

    def config_criterion(self, *args, **kwargs):
        """
        must overwrite define optimizer
        """
        return



    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        overwrite forward function to return different stuff
        """
        return super().forward(*args, **kwargs)

    def model_fn(self, batch: Union[Dict, List, Tuple]) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict]:

        assert len(batch) == 1 or len(batch) == 2, "batch should be a pair for supervised learning"
        data: torch.FloatTensor
        targets: Optional[torch.Tensor] = None
        output: torch.FloatTensor
        loss: torch.FloatTensor
        metrics: Optional[dict] = {}


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

        # batch is a List or Tuple
        elif isinstance(batch, tuple) or isinstance(batch, list):
            data, targets = batch
            data = data.to(device=self.device, non_blocking=True)
            targets = targets.to(device=self.device, non_blocking=True)

        else:
            raise TypeError

        """
        This method is prefer but in case of pytorch < 1.6, torch.cuda.amp is not defined
        '''
           `enabled`, an optional convenience argument to autocast and GradScaler. 
           If False, autocast and GradScaler’s calls become no-ops. 
           This allows switching between default precision and mixed precision without if/else statements.
        '''
       
        # amp / (cuda or cpu)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            output = self(data)
            loss = self.loss_fn(output, targets)
         """
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

        return output, loss, metrics


    def loss_fn(self, *args, **kwargs):
        """ calculate loss """
        # can overwrite this function to computer a much complex loss function
        # if targets is None or self.criterion is None:
        #     print("Targets is None or Criterion is None")
        #     return None
        # return self.criterion(outputs, targets)
        return

    #####################################################################
    def fit(
            self,
            train_dataset,
            train_batch_size: int,
            valid_dataset,
            valid_batch_size: int,
            max_epochs: int,
            device: torch.device,
            train_sampler: Optional[Sampler] = None,
            valid_sampler: Optional[Sampler] = None,
            shuffle: bool = True,
            num_workers: int = 4,
            use_fp16: bool = False,
            save_best: bool = False,
    ):
        """ fit the model """
        self.fp16: int = use_fp16

        # use all workers if -1
        self.num_workers: int = psutil.cpu_count() if num_workers == -1 else num_workers
        pin_memory: bool = True if torch.cuda.is_available() else False

        if self.train_loader is None:
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

        if self.valid_loader is None and valid_dataset is not None:
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

        self.device: torch.device = device
        if next(self.parameters()).device != device:
            self.to(device)

        if self.optimizer is None:
            try:
                self.optimizer = self.config_optimizer()
            except NotImplementedError:
                raise NotImplementedError("Optimizer is not implemented")

        if self.scheduler is None:
            assert self.optimizer is not None, "Please set up optimizer first"
            self.scheduler: Optional[Any] = self.config_scheduler()

        if self.criterion is None:
            self.criterion = self.config_criterion()

        # device indicator for progress bar
        global DEVICE
        if torch.cuda.is_available():
            DEVICE = 'AMP' if self.fp16 else 'cuda'
        else:
            DEVICE = 'cpu'

        # Used for saving best model
        history: Dict[str, List] = defaultdict(list)
        best_model_wts = None
        best_acc = 0.0

        self.start_time: float = time.time()
        n_epochs_loop = tqdm(range(max_epochs), desc="LDC", leave=True)
        for itr in n_epochs_loop:
            # Each epoch has a training and validation phase
            train_epoch_loss: Union[torch.FloatTensor, np.ndarray]
            val_epoch_loss: Union[torch.FloatTensor, np.ndarray, dict]
            val_metrics: dict

            # Training Phase
            self.phase = 'train'
            train_epoch_loss, _ = self.train_one_epoch(self.train_loader)
            avg_train_epoch_loss = np.mean(train_epoch_loss)
            history["train_loss"].append(avg_train_epoch_loss)

            # Validation phase
            self.phase = 'eval'
            if self.valid_loader:
                val_epoch_loss, val_metrics = self.validate_one_epoch(self.valid_loader)
                avg_val_epoch_loss = np.mean(val_epoch_loss)
                history["val_loss"].append(avg_val_epoch_loss)

                for k, v in val_metrics.items():
                    val_metrics[k] = np.mean(v)
                    history[k].append(v)

                # deep copy the model
                if save_best and val_metrics["acc"] is not None:
                    assert val_metrics["acc"].size == 1, "Compare multiply array value with float is ambiguous"
                    epoch_acc = float(val_metrics["acc"])
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.state_dict())
                    # Save best model
                    PATH = f"../model/test_ldc.pth"
                    torch.save(
                        {
                            'epoch': itr,
                            'model_state_dict': best_model_wts,
                            'acc': epoch_acc,
                        }, PATH
                    )
            else:
                avg_val_epoch_loss: dict = {}
                val_metrics: dict = {}

            # update progress bar
            description: str = f'({DEVICE}) Epoch: {self.current_epoch}'
            n_epochs_loop.set_description(description)
            n_epochs_loop.set_postfix(
                train_loss=avg_train_epoch_loss,
                val_loss=avg_val_epoch_loss,
                **val_metrics
            )
            self.current_epoch += 1

        self.end_time: float = time.time() - self.start_time
        if best_model_wts is None:
            message = f"WARNING: Best_model_wts is None!!"
            warnings.warn(message, UserWarning, stacklevel=2)
        return history, best_model_wts

    def train_one_epoch(self, train_loader) -> Tuple[List, Optional[dict]]:
        """ train_one_epoch """
        assert self.phase == 'train', "self.phase is not 'train' in training one epoch"
        self.train()
        # print('\nTrain One Epoch...')

        train_losses: List = []
        train_metrics: Optional[dict] = None
        # train_metrics = defaultdict(list)

        # tk0 = tqdm(train_loader, total=len(train_loader), leave=False)
        tk0 = train_loader
        for batch_idx, train_batch in enumerate(tk0):
            # train_batch is a tuple: (data, targets)
            loss, train_metrics = self.train_one_step(train_batch)
            train_losses.append(ptu.to_numpy(loss))
            # for k, v in metrics.items():
            #     train_metrics[k].append(v)
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
        # for param in self.parameters():
        #     param.grad = None
        # or maybe equivalently and concisely
        self.optimizer.zero_grad(set_to_none=True)


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
