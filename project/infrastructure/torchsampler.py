from typing import List, Dict, Tuple, Optional, Sequence, Union

import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    modified from: https://github.com/ufoym/imbalanced-dataset-sampler.git
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
            self,
            dataset,
            indices: Optional[List] = None,
            num_samples: Optional[int] = None,
            callback_get_label=None
    ):

        self.indices: List[Sequence[int]]
        self.num_samples: int

        # if indices is not provided, all elements in the dataset will be considered
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(dataset)))

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count: Dict[Union[str, int], int] = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights: List = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx: int):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        # TODO: modify to used by custom image Dataset
        elif isinstance(dataset, ...):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
