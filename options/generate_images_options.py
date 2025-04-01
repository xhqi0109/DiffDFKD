from .base_options import BaseOptions
import os

class GenerateImagesAdvOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # base setting
        self.parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default="runwayml/stable-diffusion-v1-5",
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        self.parser.add_argument(
            "--revision",
            type=str,
            default=None,
            required=False,
            help=(
                "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
                " float32 precision."
            ),
        )
        self.parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
        self.parser.add_argument('--model_name', type=str, default='resnet34_imagenet')
        self.parser.add_argument('--teacher_pretrained_path', type=str, default='checkpoints/cifar100_resnet34.pth')
        self.parser.add_argument('--save_log', action='store_true', help='if specified, then init wandb logging')
        self.parser.add_argument('--generate_nums', type=int, default=100, help='synthetic data nums for per class')

        # guided setting
        self.parser.add_argument('--bn', type=float, default=0.1, help='hyper-parameters for loss bn')
        self.parser.add_argument('--oh', type=float, default=1, help='hyper-parameters for loss oh')
        self.parser.add_argument('--adv', type=float, default=1, help='hyper-parameters for loss adv')
        self.parser.add_argument('--inference_nums', type=int, default=50, help='synthetic data nums for per class')
        self.parser.add_argument('--guided_scale', type=float, default=3, help='synthetic data nums for per class')
        self.parser.add_argument('--a_steps', type=int, default=10, help='synthetic data nums for per class')
        self.parser.add_argument('--save_syn_data_path', type=str, default='synthetic_data/', help="path for save synthetic dataset")

        # data setting
        self.parser.add_argument('--data_type', type=str, default='domainnet', help='chooses how datasets are loaded.')
        self.parser.add_argument('--domain_name', type=str, default=None, help=" sketch quickdraw infograph real painting clipart")
        self.parser.add_argument('--test_data_path', type=str, default='/share/xhqi/domainnet', help="test dataset path")

        # distill setting
        self.parser.add_argument('--model_name_s', type=str, default="resnet18_imagenet", help='dataset type')
        # self.parser.add_argument('--checkpoints_dir', type=str, default="generate_log", help='dataset type')
        self.parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate for KD')
        self.parser.add_argument('--lr_warmup_epochs', default=5, type=float, help='start CosineAnnealingLR')
        self.parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128), this is the total')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
        self.parser.add_argument('--T', default=30, type=float)
        self.parser.add_argument('--class_per_num_start', default=20, type=float,  help='ep_start < class_per_num start adv guided')
        self.parser.add_argument('--syn', action='store_false', help='true')
        self.parser.add_argument('--wandb', type=int, default=0,
                            help="set 1 for wandb logging")
        self.parser.add_argument('--label_name', action='store_true', help='默认 false')
        self.parser.add_argument('--batch_size_generation', type=int, default=1, help='if min_loss data to train')
        self.parser.add_argument('--customer_aug', type=int, default=0, help='if min_loss data to train')


    def customs(self, opt):
        # self.opt.name =  str(opt.model_name)  + str(opt.model_name_s)+ str(opt.data_type) + "_" +  str(opt.bn) + "_" + str(opt.oh)
        # opt.bn = 0.02
        # opt.oh = 0.2
        # self.opt.bn = 0.02
        # self.opt.oh = 0.2
        
        self.opt.name =  str(opt.model_name) + str(opt.model_name_s) + str(opt.data_type) + "_" +  str(opt.bn) + "_" + str(opt.oh) + "_" + str(opt.adv)
        self.opt.name =  str(opt.inference_nums) + "_" + self.opt.name
        self.opt.name =  str(opt.guided_scale) + "_" + self.opt.name
        if opt.domain_name:
            self.opt.name =  str(opt.domain_name) + "_" + self.opt.name
        if opt.label_name:
            self.opt.name = "LabelName" + "_" + self.opt.name
        if opt.customer_aug:
            self.opt.name = "customeAug" + "_" + self.opt.name
