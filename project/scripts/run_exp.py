import argparse
import sys
import os
import time

import torch
import matplotlib.pyplot as plt

from project.infrastructure.trainer import DL_Trainer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-exp', type=str, default='exp_1')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument(
        '--max__epochs', '-n', type=int, default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--min__epochs', '-min', type=int, default=1,
        help='Number of epochs'
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=64,
        help='Size of train batches'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=64,
        help='Size of eval batches'
    )

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)  # -1 not log video
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true', default=False)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../log_dir')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    # RUN TRAINING    #
    ###################
    print("##### PARAM ########")
    path = os.getcwd()
    print (path)
    # TODO: run training loop
    print(params)

    trainer = DL_Trainer(params)




if __name__ == '__main__':
    print(torch.__version__)
    main()
    print("Done!")
