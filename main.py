import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

import wandb

from framework.base2 import main_worker
from framework.config2 import get_arch

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 2,1,4,5,6,3

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clean Train')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--mp_distributed', action='store_true')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--root', default='./dataset', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--arch', default='mnistnet', type=str)
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--inner_optim', default='SGD', type=str)
    parser.add_argument('--outer_optim', default='Adam', type=str)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--inner_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--label_lr_scale', default=1, type=float, help='scale the label lr')
    parser.add_argument('--num_per_class', default=1, type=int)
    parser.add_argument('--batch_per_class', default=1, type=int)
    parser.add_argument('--task_sampler_nc', default=10, type=int)
    parser.add_argument('--window', default=20, type=int, help='Number of unrolling computing gradients')
    parser.add_argument('--minwindow', default=0, type=int, help='Start unrolling from steps x')
    parser.add_argument('--totwindow', default=20, type=int, help='Number of total unrolling computing gradients')
    parser.add_argument('--num_train_eval', default=10, type=int, help='Num of training of network for evaluation')
    parser.add_argument('--train_y', action='store_true')
    parser.add_argument('--train_lr', action='store_true')
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--test_freq', default=5, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ddtype', default='standard', type=str)
    parser.add_argument('--cctype', default=0, type=int)
    parser.add_argument('--init_type', default=0, type=int)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--project', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--clip_coef', default=0.9, type=float)
    parser.add_argument('--fname', default='_test', type=str)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--comp_aug', action='store_true')
    parser.add_argument('--comp_aug_real', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mix_prob', default=0.8, type=str)
    parser.add_argument('--syn_strategy', default='flip_rotate', type=str)
    parser.add_argument('--real_strategy', default='flip_rotate', type=str)
    parser.add_argument('--ckptname', default='none', type=str)
    parser.add_argument('--loadname', default='none', type=str)
    parser.add_argument('--data_sample', default='none', type=str)
    parser.add_argument('--data_sample_prob', default=5, type=int)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--update_steps', default=1, type=int)
    parser.add_argument('--batch_update_steps', default=1, type=int)
    parser.add_argument('--continue_forward', default=0, type=int)
    parser.add_argument('--limit_train', action='store_true')
    parser.add_argument('--labelname', default='', type=str)
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--complete_random', action='store_true')
    #parser.add_argument('--gpu', default='6', type=str)
    
    
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.mp_distributed
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    args.num_train_eval = int(args.num_train_eval / ngpus_per_node)
    
    if args.mp_distributed:
        args.world_size = ngpus_per_node * args.world_size
        
        
        
            
        
        
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

        for i in range(5):
            torch.cuda.empty_cache()
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
    
