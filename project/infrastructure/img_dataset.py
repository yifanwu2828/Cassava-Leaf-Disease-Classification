import numpy as np
import torch
import cv2
import albumentations as A


class ImageDataset(object):
    """ Image Dataset """
    def __init__(
            self,
            image_paths,
            targets,
            augmentations=None,
            channel_first=True,
            grayscale=False
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param augmentations: albumentations augmentations
        https://github.com/albumentations-team/albumentations#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation
        """
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.channel_first = channel_first
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        # targets
        target = self.targets[idx]
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)
        # image
        image = cv2.imread(self.image_paths[idx], )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        # bring channel to first index
        if self.channel_first is True and self.grayscale is False:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor = torch.from_numpy(image)

        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "target": torch.from_numpy(target)
        }
