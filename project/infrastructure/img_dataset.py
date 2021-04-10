from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
import cv2
from sklearn import preprocessing




class ImageDataset(torch.utils.data.Dataset):
    """ Image Dataset """
    def __init__(
            self,
            image_paths: str,
            targets: Union[List, Tuple, np.ndarray],
            augmentations=None,
            channel_first: bool = True,
            grayscale: bool = False,
            output_label=True,
            one_hot=False,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param augmentations: albumentations augmentations
        https://github.com/albumentations-team/albumentations#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation
        """
        super().__init__()
        self.image_paths: str = image_paths
        self.targets: Union[List, Tuple, np.ndarray] = targets

        self.augmentations = augmentations
        self.channel_first: bool = channel_first
        self.grayscale: bool = grayscale

        self.output_label: bool = output_label
        self.one_hot: bool = one_hot
        self.one_hot_label= None
        self.encoder= TargetEncoder()

        # TODO: test one hot
        if one_hot:
            self.targets = self.encoder.int2onehot(targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)

        # image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations is not None:
            augmented: dict = self.augmentations(image=image)
            image = augmented["image"]

        # bring channel to first index
        if self.channel_first is True and self.grayscale is False:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor: torch.FloatTensor = torch.from_numpy(image).float()

        if not self.output_label:
            return {
                "image": image_tensor,
                "target": None
            }

        # targets
        target = self.targets[idx]
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)
        target_tensor: torch.LongTensor = torch.from_numpy(target)

        if self.grayscale:
            # [H W] -> [H W C]
            image_tensor = image_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "target": target_tensor
        }


class TargetEncoder(object):
    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)

    def str2int(self, targets) -> np.ndarray:
        for target in targets:
            assert isinstance(target, str)
        if isinstance(targets, torch.Tensor):
            targets = targets.to('cpu').detach().numpy()
        elif isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, np.ndarray):
            targets = numpy.array(targets)
        else:
            raise ValueError("Targets should be type: Union[List, Tuple, torch.Tensor, numpy.ndarray]")
        integer_encoded = self.label_encoder.fit_transform(targets)
        return integer_encoded

    def int2onehot(self, targets) -> np.ndarray:
        if isinstance(targets, torch.Tensor):
            targets = targets.to('cpu').detach().numpy()
        elif isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, np.ndarray):
            targets = numpy.array(targets)
        else:
            raise ValueError("Targets should be type: Union[List, Tuple, torch.Tensor, numpy.ndarray]")
        targets = targets.reshape(-1, 1)
        one_hot_encoded = self.one_hot_encoder.fit_transform(targets)
        return one_hot_encoded

    def str2onehot(self, targets) -> np.ndarray:
        for target in targets:
            assert isinstance(target, str)
        if isinstance(targets, torch.Tensor):
            targets = targets.to('cpu').detach().numpy()
        elif isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, np.ndarray):
            targets = numpy.array(targets)
        else:
            raise ValueError("Targets should be type: Union[List, Tuple, torch.Tensor, numpy.ndarray]")
        integer_encoded = self.label_encoder.fit_transform(targets)
        return self.int2onehot(integer_encoded)
