import os
from typing import List, Dict, Optional, Union, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


DATA = Union[torch.Tensor, np.ndarray]


class Logger(object):
    """
    Tensorboard Logger
    Before going further, more details on TensorBoard can be found at https://www.tensorflow.org/tensorboard/
    """
    def __init__(self, log_dir: str, n_logged_samples: int = 10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    ##################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    ##################################

    @property
    def logdir(self) -> str:
        return self._log_dir

    @property
    def n_logged_samples(self) -> int:
        return self._n_logged_samples

    @property
    def summ_writer(self):
        return self._summ_writer

    #####################################################
    #####################################################

    def log_scalar(self, name: str, scalar: Union[int, float, str], step=None) -> None:
        """
        Add scalar data to summary
        """
        assert isinstance(name, str)
        self._summ_writer.add_scalar(tag=f'{name}', scalar_value=scalar, global_step=step)

    def log_scalars(self, group_name: str, scalar_dict: dict, step: Optional[int] = None, phase: int = None) -> None:
        """
        Adds many scalar data to summary
        Will log all scalars in the same plot.
        """
        assert isinstance(group_name, str)
        assert isinstance(scalar_dict, dict), "Need dict of scalars"
        self._summ_writer.add_scalars(main_tag=f'{group_name}_{phase}', tag_scalar_dict=scalar_dict, global_step=step)

    def log_image(self, name: str, image: DATA, step: Optional[int] = None) -> None:
        """
        Add image data to summary.
        Note that this requires the pillow package.
        """
        assert isinstance(name, str)
        assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray)
        assert(len(image.shape) == 3), "Need [C, H, W] input tensor for single image logging!"
        self._summ_writer.add_image(tag=f'{name}', img_tensor=image, global_step=step)

    def log_images(self, name: str,
                   images: DATA,
                   step: Optional[int] = None, data_formats='NCHW'
                   ):
        """
        Add batched image data to summary.
        Note that this requires the pillow package
        """
        assert isinstance(name, str)
        # Default is (N, 3, H, W)
        assert isinstance(images, torch.Tensor) or isinstance(images, np.ndarray)
        assert (len(images.shape) == 4), "Need [N, C, H, W] input tensor for images logging!"
        self._summ_writer.add_images(tag=name, img_tensor=images, global_step=step, dataformats=data_formats)

    def log_video(self, name: str, video_frames: torch.Tensor, step: Optional[int] = None, fps: int = 10) -> None:
        """
        Add video data to summary.
        Note that this requires the moviepy package.
        :param name: Data identifier
        :type: str
        :param video_frames: Video data [N, T, C, H, W]
        :type: torch.Tensor
        :param step: Global step value to record
        :type: int
        :param fps: Frames per second
        :type: int
        """
        assert isinstance(name, str)
        assert isinstance(video_frames, torch.Tensor)
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video(tag=f'{name}', vid_tensor=video_frames, global_step=step, fps=fps)

    def log_paths_as_videos(self, paths, step,
                            max_videos_to_save: int = 2, fps: int = 10,
                            video_title: str = 'video'
                            ) -> None:
        """
        This function is used in RL, logging sample trajectory as video
        :param paths: rollouts
        :param step: number of transition steps
        :param max_videos_to_save:
        :param fps:
        :param video_title: Frames per second
        """
        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(name=video_title, video_frames=videos, step=step, fps=fps)

    def log_hist(self, name: str, values: torch.Tensor, bins: str = 'tensorflow') -> None:
        """Add histogram to summary."""
        assert isinstance(name, str)
        assert isinstance(values, torch.Tensor), "values should be type torch.Tensor"
        self._summ_writer.add_histogram(tag=name, values=values, bins=bins)


    def log_figure(self, name: str,
                   figure: plt.figure,
                   step: Optional[int] = None,
                   close=True,
                   phase: int = None
                   ) -> None:
        """
        Render matplotlib figure into an image and add it to summary
        :param name:
        :param figure: matplotlib.pyplot figure handle
        :param step:
        :param close:
        :param phase:
        :return:
        """
        assert isinstance(name, str)
        self._summ_writer.add_figure(f'{name}_{phase}', figure, step, close=close)

    def log_figures(self, name: str,
                    figure: plt.figure,
                    step: Optional[int] = None,
                    close=True,
                    phase: int = None) -> None:
        """figure: matplotlib.pyplot figure handle"""
        assert isinstance(name, str)
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure(f'{name}_{phase}', figure, step, close=close)

    # def log_graph(self,
    #               model: torch.nn.Module,
    #               model_input: Union[torch.Tensor, List[torch.Tensor, None]] = None,
    #               verbose=False):
    #     """Add NN graph data to summary."""
    #     assert isinstance(model, torch.nn.Module), "model should be type torch.nn.Module"
    #     self._summ_writer.add_graph(model, input_to_model=model_input, verbose=verbose)

    def log_pr_curve(self,
                     name: str,
                     labels: DATA,
                     predictions: DATA,
                     step: int,
                     num_thresholds: int = 127,
                     weights=None) -> None:
        """
        Adds precision recall curve.
        :param name:
        :param labels: Ground truth data. Binary label for each element.
        :param predictions:The probability that an element be classified as true. Value should be in [0, 1]
        :param step:
        :param num_thresholds: The TensorBoard UI will let you choose the threshold interactively.
        :param weights:
        """
        assert isinstance(name, str)
        assert isinstance(labels, torch.Tensor) or isinstance(labels, np.ndarray)
        assert isinstance(predictions, torch.Tensor) or isinstance(predictions, np.ndarray)
        self._summ_writer.add_pr_curve(tag=name,
                                       labels=labels, predictions=predictions,
                                       global_step=step,
                                       num_thresholds=num_thresholds, weights=weights)

    # only works in torch.utils.tensorboard, not tensorflow's tensorboardX
    def log_hparams(self, hparam_dict: dict, metric_dict: dict,
                    hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
                    run_name=None
                    ) -> None:
        """
        Add a set of hyperparameters to be compared in TensorBoard.
        :param hparam_dict: name of the hyper parameter and it’s corresponding value.
        :param metric_dict: name of the metric and it’s corresponding value
        :param hparam_domain_discrete: dict contains names of the hyperparameters and all discrete values
        :param run_name: Name of the run, using current timestamp if None
        """
        # Note that the key used in metric_dict  should be unique in the tensorboard record.
        # Otherwise the value you added by add_scalar will be displayed in hparam plugin.
        assert isinstance(hparam_dict, dict), "hparam_dict should be dict"
        assert isinstance(metric_dict, dict), "metric_dict should be dict"
        self._summ_writer.add_hparams(hparam_dict,
                                      metric_dict,
                                      hparam_domain_discrete=hparam_domain_discrete,
                                      run_name=run_name)

    def flush(self):
        """
        Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._summ_writer.flush()

    def close(self):
        """ Close writer at the end of training """
        self._summ_writer.close()
