import random
from helper.utils import setup_seed
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
import torch
import torchvision.datasets as datasets
# val_dataset = datasets.CIFAR100("download/torchdata", train=False, download=True)
from torchvision import datasets, transforms as T
from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from tqdm import tqdm

class Config:
    # "tench", "English springer", "cassette player", "chain saw",
    #                     "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    imagenet100 = [117, 70, 88, 133, 5, 97, 42, 60, 14, 3, 130, 57, 26, 0, 89, 127, 36, 67, 110, 65, 123, 55, 22, 21, 1, 71, 
                    99, 16, 19, 108, 18, 35, 124, 90, 74, 129, 125, 2, 64, 92, 138, 48, 54, 39, 56, 96, 84, 73, 77, 52, 20, 
                    118, 111, 59, 106, 75, 143, 80, 140, 11, 113, 4, 28, 50, 38, 104, 24, 107, 100, 81, 94, 41, 68, 8, 66, 
                    146, 29, 32, 137, 33, 141, 134, 78, 150, 76, 61, 112, 83, 144, 91, 135, 116, 72, 34, 6, 119, 46, 115, 93, 7]
    
    dict = {
        "imagenette" : imagenette,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagesquawk": imagesquawk,
        "imagenet100": imagenet100,
    }

config = Config()

NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet1k': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'imagenette': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'domainnet':dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}

NUMS_CLASSES_DICT = {
    'cifar10':  10,
    'cifar100': 100,
    'imagenet1k': 1000,
    'imagenette': 10,
    'domainnet': 10,
}

IMAGE_SIZE_DICT = {
    'cifar10':  32,
    'cifar100': 32,
    'imagenet1k': 224,
    'imagenette': 224,
    'domainnet': 224,
}

def get_transforms(name):
    image_size = IMAGE_SIZE_DICT[name]
    transforms = T.Compose([
                    T.Resize(
                        (image_size, image_size),
                        interpolation=T.InterpolationMode.BICUBIC,
                        # antialias=False,
                    ),
                    T.Normalize( **NORMALIZE_DICT[name] ),
                    ])
    return transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None, steps=None, nums=None, min_loss=False, args=None):
        # setup_seed(seed)
        self.root = root
        self.transform = transform
        dir_names = os.listdir(self.root)
        dir_names.sort()

        label = -1
        self.images, self.labels = [], []

        for dir_name in dir_names:
            label += 1
            dir_path = os.path.join(self.root, dir_name)
            image_names = os.listdir(dir_path)
            image_names.sort()
            sample_image_nums = 0

            for image_name in image_names:
                if min_loss and "s" in image_name:
                    continue

                if nums == None :
                    self.images.append(os.path.join(dir_path, image_name))
                    self.labels.append(label)
                elif sample_image_nums<nums:
                    self.images.append(os.path.join(dir_path, image_name))
                    self.labels.append(label)
                    sample_image_nums += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(DomainNet, self).__init__()
        self.data = data_paths
        self.target = data_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.target[index] 
        img = self.transforms(img)

        return img, label

def read_domainnet_data(data_root, domain_name, split="train"):
    data_paths = []
    data_labels = []
    # ['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet']
    choice_labels = [1, 73, 11, 19, 29, 31, 290, 121, 225, 39]
    split_file = path.join(data_root, "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            label = int(label) 
            if int(label) in choice_labels:
                data_path = path.join(data_root, data_path)
                data_paths.append(data_path)
                data_labels.append(choice_labels.index(label))
                
    return data_paths, data_labels

def get_dataset_train_syn(name: str, data_root: str='data', args=None):
    name = name.lower()
    data_root = os.path.expanduser( data_root )
    if name in ['cifar100', "cifar10"]:
        train_transform = T.Compose([
            T.Resize(32,interpolation=transforms.InterpolationMode.BICUBIC,),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
    elif name in ['imagenet1k', "imagenette"]:
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
    elif name == "domainnet":
        train_transform = T.Compose([
            T.Resize(256,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    # antialias=False
                    ),
            T.RandomResizedCrop(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    # antialias=False
                    ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
    else:
        raise NotImplementedError
    if not hasattr(args, 'steps'):
        args.steps = None
    if not hasattr(args, 'nums'):
        args.nums = None
    if not hasattr(args, 'min_loss'):
        args.min_loss = None
        
    train_dataset = ImageFolderDataset(data_root, train_transform, steps=args.steps, nums=args.nums, min_loss=args.min_loss, args=args)
    return train_dataset

def get_dataset_train(name: str, data_root: str='data', args=None):
    name = name.lower()
    # data_root = os.path.expanduser( data_root )
    if name=='cifar10':
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' )
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    elif name=='cifar100':
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dataset = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
    elif name=='imagenet1k':
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=data_root, 
                                                                transform=train_transform)
    elif name == "domainnet":
        if args.aug:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            # antialias=False
                            ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
                ])
        else:
            train_transform = transforms.Compose([
                T.Resize(
                        (224,224),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        # antialias=False,
                    ),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_DICT[name]),
                ])
            # train_transform = transforms.Compose([
            #         transforms.RandomResizedCrop(224, scale=(0.75, 1)),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         T.Normalize(**NORMALIZE_DICT[name]),

            #         ])

        data_paths, data_labels = read_domainnet_data(data_root=data_root, domain_name=args.domain_name, split="train")
        train_dataset = DomainNet(data_paths, data_labels, train_transform)
    else:
        raise NotImplementedError
    return train_dataset

def get_dataset_val(name: str, data_root: str='data', args=None):
    name = name.lower()
    data_root = os.path.expanduser( data_root )
    
    if name=='cifar10':
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' )
        val_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar100':
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        val_dataset = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name in ['imagenet1k', "imagenette", "imagenet100"]:
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root) 
        val_dataset = torchvision.datasets.ImageFolder(root=data_root, 
                                                                transform=val_transform)
        # val_dataset = datasets.ImageNet(data_root, split='val', transform=val_transform)
    elif name == "domainnet":
        val_transform  = transforms.Compose([
            T.Resize(
                (224,224),
                interpolation=T.InterpolationMode.BICUBIC,
                # antialias=False,
                ),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
            ])
        data_paths, data_labels = read_domainnet_data(data_root=data_root, domain_name=args.domain_name, split="test")
        val_dataset = DomainNet(data_paths, data_labels, val_transform)
    else:
        raise NotImplementedError
    return val_dataset

def get_dataset(args):
    if hasattr(args, 'syn') and args.syn:
        train_dataset = get_dataset_train_syn(args.data_type, data_root=args.train_data_path, args=args)
    else:
        args.syn = False
        train_dataset = get_dataset_train(args.data_type, data_root=args.train_data_path, args=args)

    val_dataset = get_dataset_val(args.data_type, data_root=args.test_data_path, args=args)
    print("load syn:{} {} dataset\n train,len:{} from {}.\n test, len:{} from {}.".format(
                                                                                    args.syn,
                                                                                    args.data_type, 
                                                                                    len(train_dataset),
                                                                                    args.train_data_path, 
                                                                                    len(val_dataset),
                                                                                    args.test_data_path))
    return train_dataset, val_dataset
