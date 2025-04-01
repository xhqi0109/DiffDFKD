from .base_options import BaseOptions
from datetime import datetime
import os

class DistillOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        ## base settings
        self.parser.add_argument('--save_log', action='store_true', help='if specified, then init wandb logging')
        self.parser.add_argument('--seed', type=int, default=0)

        ## knowledge distillation settings
        self.parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate for KD')
        self.parser.add_argument('--warmup', default=20, type=float, help='start CosineAnnealingLR')
        self.parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        self.parser.add_argument('--epochs', default=240, type=int, metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 128), this is the total')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
        self.parser.add_argument('--alpha', default=1, type=float, help='loss_div')
        self.parser.add_argument('--gamma', default=0, type=float, help='loss_cls')

        self.parser.add_argument('--t_model', type=str, default=None, help='dataset type')
        self.parser.add_argument('--s_model', type=str, default=None, help='dataset type')
        self.parser.add_argument('--teacher_pretrained_path', type=str, default=None)
        self.parser.add_argument('--T', default=10, type=float)

        ## dataset settings
        self.parser.add_argument('--data_type', type=str, default='cifar100', help='chooses how datasets are loaded.')
        self.parser.add_argument('--domain_name', default=None, type=str, help="['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']")
        self.parser.add_argument('--train_data_path', type=str, default=None, help="train dataset path")
        self.parser.add_argument('--test_data_path', type=str, default='./download', help="test dataset path")
        self.parser.add_argument('--steps', default=None, type=int, help='choice for generation steps')
        self.parser.add_argument('--nums', default=None, type=int, help='if None for All')
        self.parser.add_argument('--syn', action='store_true', help='if syn data to train')
        self.parser.add_argument('--aug', action='store_true', help='if aug syn data to train')
        self.parser.add_argument('--min_loss', action='store_true', help='if min_loss data to train')
        self.parser.add_argument('--use_aug', type=str, default=None, help='if min_loss data to train')

    def customs(self, opt):
        self.opt.name = str(opt.data_type) + "_" + str(opt.t_model)  + "_" +  str(opt.s_model)  + "_" +  "{}_nums:{}_T:{}".format(str(opt.name), opt.nums, opt.T) + str(opt.current_time)
        if opt.gamma >0 :
            self.opt.name = str(opt.gamma) + "_" + self.opt.name
        if opt.domain_name:
            self.opt.name =  opt.domain_name + "_" + self.opt.name
        if opt.aug:
            self.opt.name = "aug" + "_" + self.opt.name
        if opt.syn:
            self.opt.name = "syn" +"_" + self.opt.name
        if opt.use_aug:
            self.opt.name = opt.use_aug + "_" + self.opt.name
        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

