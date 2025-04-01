import argparse
import os
from helper.utils import mkdirs
import torch
import numpy as np
from datetime import datetime
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):  
        ## base
        self.parser.add_argument('--name', type=str, default='demo', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')

        #wandb
        self.parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
        self.parser.add_argument('--wandb_project_name', type=str, default='DiffDFKD', help='specify wandb project name')

        self.initialized = True
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.parser.add_argument('--current_time', type=str, default=current_time, help='current_time')


    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def customs(self, opt):
        pass

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.checkpoints_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs(self.opt.checkpoints_dir)

        # rename name checkpoints_dir
        self.customs(self.opt)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        self.opt.checkpoints_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs(self.opt.checkpoints_dir)

        # self.opt.checkpoints_dir = expr_dir
        if save:
            file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt