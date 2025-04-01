#!/bin/bash  

CUDA_VISIBLE_DEVICES=0  python train_scratch.py \
        --data_type domainnet --model_name vgg11 \
        --name exp_aug \
        --checkpoints_dir pretrained_models \
        --lr 0.01 \
        --epochs 120 \
        --momentum 0.9 \
        --batch_size 64 \
        --weight_decay 2e-4 \
        --train_data_path /share/xhqi/domainnet \
        --test_data_path  /share/xhqi/domainnet \
        --domain_name sketch \
        --wandb 0