import argparse
import os

import numpy as np
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_dirs(options):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists(f"checkpoints/{options.exp_name}"):
        os.makedirs(f"checkpoints/{options.exp_name}")
    if not os.path.exists(f"checkpoints/{options.exp_name}/models"):
        os.makedirs(f"checkpoints/{options.exp_name}/models")
    os.system(f"cp main.py checkpoints/{options.exp_name}/main.py.backup")
    os.system(f"cp model.py checkpoints/{options.exp_name}/model.py.backup")
    os.system(f"cp data.py checkpoints/{options.exp_name}/data.py.backup")


def set_deterministic_seeds(options):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)


def add_config(parser):
    parser.add_argument('--exp_name', type=str, default='neural_scene_flow_prior', metavar='N', help='Name of the experiment.')
    parser.add_argument('--num_points', type=int, default=2048, help='Point number [default: 2048].')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size', help='Batch size.')
    parser.add_argument('--iters', type=int, default=5000, metavar='N', help='Number of iterations to optimize the model.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam'), help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR', help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0, metavar='M', help='SGD momentum (default: 0.9).')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Enables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='Random seed (default: 1234).')
    parser.add_argument('--dataset', type=str, default='KITTISceneFlowDataset',
                        choices=['FlyingThings3D', 'KITTISceneFlowDataset', 'ArgoverseSceneFlowDataset', 
                                 'NuScenesSceneFlowDataset'], metavar='N',
                        help='Dataset to use.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/kitti', metavar='N',
                        help='Dataset path.')
    parser.add_argument('--compute_metrics', action='store_true', default=True, help='whether compute metrics or not.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Show visuals.')
    parser.add_argument('--animation', action='store_true', default=False, help='create animations.')
    parser.add_argument('--time', dest='time', action='store_true', default=True, help='Count the execution time of each step.')
    parser.add_argument('--partition', type=str, default='val', metavar='p', help='Model to use.')
    # For neural prior
    parser.add_argument('--model', type=str, default='neural_prior', choices=['neural_prior'], metavar='N', help='Model to use.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='N', help='Weight decay.')
    parser.add_argument('--hidden_units', type=int, default=128, metavar='N', help='Number of hidden units in neural prior')
    parser.add_argument('--layer_size', type=int, default=8, metavar='N', help='Number of layers in neural prior')
    parser.add_argument('--use_all_points', action='store_true', default=False, help='use all the points or not.')
    parser.add_argument('--act_fn', type=str, default='relu', metavar='AF', help='activation function for neural prior.')
    parser.add_argument('--backward_flow', action='store_true', default=True, help='use backward flow or not.')
    parser.add_argument('--early_patience', type=int, default=100, help='patience in early stopping.')
