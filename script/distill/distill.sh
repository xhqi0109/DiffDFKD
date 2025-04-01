#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python distill.py --data_type domainnet --t_model resnet34_imagenet --s_model resnet18_imagenet \
        --domain_name quickdraw \
        --lr 0.1 \
        --epochs 220 \
        --batch_size 256 \
        --momentum 0.9 \
        --weight_decay 1e-4 \
        --train_data_path check_new_new_synthetic_data_domainnet_0/quickdraw_3.0_70_resnet34_imagenetNonedomainnet_0.1_1.0_0.0 \
        --test_data_path /share/xhqi/domainnet \
        --teacher_pretrained_path  pretrained_models/exp0/quickdraw-resnet34_imagenet-domainnet-exp0/model-best-epoch-31-88.93.pt \
        --T 30 \
        --syn \
        --save_log \
        --gamma 0 \
        --nums 4500 \
        --use_aug None \
        --name  check/45k0.1_1.0_0.0_3_70 \
        --seed 0 \
        --checkpoints_dir checkpoints_distill_domainnet


CUDA_VISIBLE_DEVICES=0  python distill.py --data_type cifar10 --t_model wrn40_2 --s_model wrn16_2 \
        --lr 0.2 \
        --warmup 20\
        --epochs 240 \
        --batch_size 256 \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --train_data_path synthetic_data_cifar10_3/3.0_50_wrn40_2Nonecifar10_0.1_1.0_0.0\
        --test_data_path ./download \
        --teacher_pretrained_path ./checkpoints/cifar10_wrn40_2.pth \
        --T 30\
        --gamma 0 \
        --seed 0 \
        --syn \
        --save_log \
        --nums 50000 \
        --use_aug None \
        --name  synthetic_data_cifar10_3_50 \
        --checkpoints_dir checkpoints_distill_cifar_new4


