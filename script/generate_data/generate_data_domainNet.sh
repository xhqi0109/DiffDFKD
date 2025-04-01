# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate_images.py   \
        --data_type domainnet \
        --model_name resnet34_imagenet \
        --model_name_s  resnet18_imagenet \
        --domain_name sketch \
        --teacher_pretrained_path ./checkpoints_train_from_scratch/exp0/sketch-resnet34_imagenet-domainnet-exp0/model-best-epoch-113-57.79.pt\
        --save_syn_data_path ./synthetic_data_scaleNums\
        --generate_nums 200\
        --inference_nums 70 \
        --guided_scale 7 \
        --bn 0.1 --oh 1 --adv 1 --seed 0 \
        --a_steps 0 \
        --save_log \
        --test_data_path  /share/xhqi/domainnet \
        --epochs 30 \
        --lr 0.1 \
        --lr_warmup_epochs 5 \
        --batch_size 256 \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --T 30 \
        --class_per_num_start 20 \
        --checkpoints_dir generate_log_scaleNums




CUDA_VISIBLE_DEVICES=0 python generate_images.py   \
        --data_type cifar10 \
        --model_name resnet34 \
        --model_name_s  resnet18 \
        --teacher_pretrained_path ./checkpoints/cifar10_resnet34.pth\
        --save_syn_data_path ./synthetic_data_baseline_label\
        --generate_nums 5000\
        --inference_nums 10 \
        --label_name \
        --guided_scale 3 \
        --bn 0 --oh 1 --adv 0 --seed 5000 \
        --a_steps 0 \
        --save_log \
        --epochs 0 \
        --lr 0.1 \
        --lr_warmup_epochs 5 \
        --batch_size 256 \
        --momentum 0.9 \
        --weight_decay 1e-4 \
        --T 30 \
        --class_per_num_start 10 \
        --checkpoints_dir generate_log_label

