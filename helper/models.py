

import torch
import torchvision
import torchvision.models as models
from helper import classifiers
from torchvision import datasets, transforms as T
import random
from torchvision import datasets, transforms
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


import os
import torch
import torchvision
import torch.nn as nn 
from PIL import Image



MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': classifiers.wresnet.wrn_16_1,
    'wrn16_2': classifiers.wresnet.wrn_16_2,
    'wrn40_1': classifiers.wresnet.wrn_40_1,
    'wrn40_2': classifiers.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': classifiers.resnet_tiny.resnet8,
    'resnet20': classifiers.resnet_tiny.resnet20,
    'resnet32': classifiers.resnet_tiny.resnet32,
    'resnet56': classifiers.resnet_tiny.resnet56,
    'resnet110': classifiers.resnet_tiny.resnet110,
    'resnet8x4': classifiers.resnet_tiny.resnet8x4,
    'resnet32x4': classifiers.resnet_tiny.resnet32x4,
    'vgg8': classifiers.vgg.vgg8_bn,
    'vgg11': classifiers.vgg.vgg11_bn,
    'vgg13': classifiers.vgg.vgg13_bn,
    'shufflenetv2': classifiers.shufflenetv2.shuffle_v2,
    'mobilenetv2': classifiers.mobilenetv2.mobilenet_v2,
    
    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  classifiers.resnet.resnet50,
    'resnet18':  classifiers.resnet.resnet18,
    'resnet34':  classifiers.resnet.resnet34,
}

IMAGENET_MODEL_DICT = {
    'resnet50_imagenet': classifiers.resnet_in.resnet50,
    'resnet18_imagenet': classifiers.resnet_in.resnet18,
    'resnet34_imagenet': classifiers.resnet_in.resnet34,
    'mobilenetv2_imagenet': torchvision.models.mobilenet_v2,
    'vgg11': classifiers.vgg.vgg11_bn,
}

def get_model(name: str, num_classes, pretrained_path=None):
    if 'imagenet' in name or "domainnet" == name:
        if name == "resnet50_imagenet":
            print("load pretrained resnet50_imagenet!!!!")
            model = IMAGENET_MODEL_DICT[name](pretrained=True)
        else:
            model = IMAGENET_MODEL_DICT[name](pretrained=False)

        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    
    # load pretrained model
    if pretrained_path is not None:
        weight = torch.load(pretrained_path)
        if 'state_dict' in weight:
            checkpoint_tmp = weight['state_dict']
            checkpoint = {}
            for key in checkpoint_tmp:
                checkpoint[key.replace("module.", "")] = checkpoint_tmp[key]
        else:
            checkpoint = weight
        model.load_state_dict(checkpoint, strict=True)
        print("load pretrained model path:{}".format(pretrained_path))
    model.to("cuda")
    return model 