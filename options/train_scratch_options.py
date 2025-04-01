from .base_options import BaseOptions
from datetime import datetime
import os

class TrainScratchOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # Training model hyperparameter settings
        self.parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        self.parser.add_argument('--epochs', type=int, default=120, help="number of training epochs")
        self.parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
        self.parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
        self.parser.add_argument('--weight_decay', '--wd', default=2e-4,
                            type=float, metavar='W')
        self.parser.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")
        
        # model setting
        self.parser.add_argument('--model_name', type=str, 
                            default="resnet18",
                            help='model name to train')
        self.parser.add_argument('--model_path', type=str, 
                            default=None,
                            help='load model weight path')
        
        # dataset setting 
        self.parser.add_argument('--data_type', type=str, 
                            # choices=["imagenet1000", "domainnet", "imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "cub", "cars", "sat", "covid", "dermamnist", "bloodmnist" ],
                            default="domainnet",
                            help='data set type')
        self.parser.add_argument('--train_data_path', default=None, type=str, help='data path for train')
        self.parser.add_argument('--test_data_path', default=None, type=str, help='data path for test')
        self.parser.add_argument('--domain_name', default=None, type=str, help="['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']")
        
        # others setting
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
        self.parser.add_argument('--wandb', type=int, default=0,
                            help="set 1 for wandb logging")

    def customs(self, opt):
        if opt.domain_name:
            self.opt.name = "{}-{}-{}-{}".format(opt.domain_name, opt.model_name, opt.data_type, opt.name)
        else:
            self.opt.name = "{}-{}-{}".format(opt.model_name, opt.data_type, opt.name)
        