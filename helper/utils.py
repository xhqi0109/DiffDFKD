from pty import STDERR_FILENO
import cv2
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn
from torchvision.transforms import transforms, Resize, InterpolationMode, PILToTensor, ConvertImageDtype, Normalize, \
    ToPILImage
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, convert_image_dtype
import torch.nn.init as init
from PIL import Image
import random
import numpy as np
from PIL import Image
import torchvision.transforms as tfs
from torchvision.transforms import InterpolationMode

import torch
import torch.backends.cudnn as cudnn
import os
import sys
import time
import math
import copy
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


import sys
import time

import torch
# from .distiller_zoo import DistillKL2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import numpy as np
from PIL import Image
import torch
import random
from helper.dataset import get_transforms, NUMS_CLASSES_DICT, get_dataset
from helper.models import get_model
from helper.prompt import PROMPT_DICT


class HandV_translation(object):

    def __init__(self, image_gap=0):
        self.image_gap = image_gap

    def __call__(self, img):
        HorV = random.randint(0, 2)
        '''HandV translation '''
        if HorV != 1:
            left = img[:, :, :, :self.image_gap]
            right = img[:, :, :, self.image_gap:]
            img = torch.cat([right, left], dim=-1)
        if HorV != 0:
            top = img[:, :, :self.image_gap, :]
            bottom = img[:, :, self.image_gap:, :]
            img = torch.cat([bottom, top], dim=-2)
        return img

class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        bs,c , h, w = img.shape
        Pepper = torch.min(img).item()
        Salt = torch.max(img).item()
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(bs, 1, h, w), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=1)  # 在通道的维度复制，生成彩色的mask
        img[torch.from_numpy(mask == 0)] = Pepper/2  # 椒
        img[torch.from_numpy(mask == 1)] = Salt/2  # 盐
        return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        bs,c , h, w = img.shape
        N = np.random.normal(loc=self.mean, scale=self.variance, size=(bs, 1, h, w))
        N = np.repeat(N, c, axis=1).astype(np.float32)
        N = self.amplitude*N
        img = torch.from_numpy(N).cuda() + img
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class CutMix(object):
    def __init__(self, beta=0.0):
        self.beta = beta

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image0,image1):
        # bs,c , h, w = img.shape
        # N = np.random.normal(loc=self.mean, scale=self.variance, size=(bs, 1, h, w))
        # N = np.repeat(N, c, axis=1).astype(np.float32)
        # img = torch.from_numpy(N).cuda() + img
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')

        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        # rand_index = torch.randperm(image0.size()[0]).cuda()
        # labels_a = labels
        # labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image0.size(), lam)
        image0[:, :, bbx1:bbx2, bby1:bby2] = image1[:, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return image0,lam


def train_transforms(inputs):
    img_size = inputs.size(-1)
    image_gap = random.randint(2, 8)
    random_trans = tfs.RandomOrder([
        tfs.RandomApply(
            [
                tfs.Resize((img_size + image_gap, img_size + image_gap)),
                tfs.CenterCrop((img_size, img_size)),
            ],0.5
        ),
        tfs.RandomHorizontalFlip(),
        tfs.RandomApply(
            [tfs.RandomRotation(image_gap)],0.5
        ),
        tfs.RandomApply(
            [HandV_translation(image_gap)],0.5
        ),
        tfs.RandomApply(
            [
                tfs.Pad([int(image_gap / 2), int(image_gap / 2),
                                int(image_gap / 2), int(image_gap / 2)]),
                tfs.Resize((img_size, img_size)),
            ],0.5
        ),
        tfs.RandomApply(
            [tfs.GaussianBlur(3, sigma=(0.1, 1.0))],0.5
        ),
        tfs.RandomApply(
            [AddGaussianNoise(0.0, 1.0, 0.01)],0.5
        ),
        tfs.RandomApply(
            [AddSaltPepperNoise(0.01)],0.5
        ),
        tfs.RandomApply(
            [tfs.RandomAffine(image_gap)],0.5
        ),
        tfs.RandomApply(
            [tfs.RandomErasing(scale=(0.02, 0.22))],0.5
        ),
    ])
    return random_trans(inputs)

def mixup_data(x, y, args, alpha=0.4):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix(data, targets, alpha=0.25):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    # shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)

    return data, None


def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        new_lr = args.lr * (args.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_sd_tensors(imgs, preprocess):
    imgs = F.interpolate(imgs, size=preprocess[0].size, mode="bicubic")
    imgs = preprocess[-1](imgs)
    return imgs



def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)
    


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Training
def train(net, train_loader, optimizer, if_log=False):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if if_log:
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct, total, train_loss

# Distill Training
def distill_train(net, train_loader, criterion, optimizer, if_log=False, args=None):
    student, teacher = net
    optimizer = optimizer
    if args is not None and hasattr(args, 'gamma') and args.gamma > 0:
        criterion_cls = nn.CrossEntropyLoss()
    student.train()
    teacher.eval()
    total, correct = 0, 0
    loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        inputs, labels = inputs.cuda(), labels.cuda()
        if args is not None and hasattr(args, 'half') and args.half:
            inputs, labels = inputs.half(), labels.half()
        if hasattr(args, 'use_aug') and args.use_aug == 'mix_cut':
            random_type = np.random.randint(0, 2)
            if random_type == 1:
                inputs, _, _, _ = mixup_data(inputs, labels, args)
            else:
                inputs, _ = cutmix(inputs, labels)
            # import pdb;pdb.set_trace()
        elif hasattr(args, 'use_aug') and args.use_aug == 'mixup':
            inputs, _, _, _ = mixup_data(inputs, labels, args)
        elif hasattr(args, 'use_aug') and  args.use_aug == 'cutmix':
            inputs, _ = cutmix(inputs, labels)
        elif hasattr(args, 'use_aug') and args.use_aug == "traditional":
            inputs = train_transforms(inputs)

        with torch.no_grad():
            t_out = teacher(inputs)
        s_out = student(inputs.detach())
        loss_div = criterion(s_out, t_out.detach())
        
        if hasattr(args, 'gamma') and args.gamma > 0:
            loss_cls = criterion_cls(s_out, labels)
            loss_s = args.alpha * loss_div + args.gamma * loss_cls
        else:
            loss_s = loss_div
        optimizer.zero_grad()
        loss_s.backward()
        loss += loss_s.item()
        optimizer.step()

        _, predicted = s_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if if_log:
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (loss/(batch_idx+1), 100.*correct/total, correct, total))
    return loss

# Testing
def test_accuracy(net, testloader, if_log=False, args=None):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            if args is not None and hasattr(args, 'half') and args.half:
                inputs, targets = inputs.half(), targets.half()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if if_log:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100*(correct/total), test_loss



class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
    

def customer_aug_data(x_s, alpha=1.0, customer_aug=0):
    if customer_aug == 1:
        x_s = train_transforms(x_s)
    elif customer_aug == 2:
        lam = np.random.beta(alpha, alpha)
        x_s, _, _, _ = mixup_data(x_s, x_s, None, alpha=lam)
    elif customer_aug == 3:
        lam = np.random.beta(alpha, alpha)
        x_s, _ = cutmix(x_s, None, alpha=lam)
        
    return x_s, customer_aug, 0




def train_and_evaluate(model, model_s, train_dataset, test_dataset, args, accelerator):
    """
    Train and evaluate the student model using synthetic data.
    """
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Loss function and optimizer
    criterion = KLDiv(T=args.T)
    optimizer = torch.optim.SGD(
        model_s.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-4)

    # Prepare for acceleration
    model_s, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model_s, optimizer, train_loader, test_loader, scheduler
    )

    best_acc = -1.0
    for epoch in tqdm(range(args.epochs)):
        model_s.train()
        model.eval()
        total, correct = 0, 0
        loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.half().to(accelerator.device)
            labels = labels.half().to(accelerator.device)

            with torch.no_grad():
                t_out = model(inputs)
            s_out = model_s(inputs)
            
            loss_s = criterion(s_out, t_out.detach())

            optimizer.zero_grad()
            accelerator.backward(loss_s)
            optimizer.step()

            _, predicted = s_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Update learning rate after warmup
        if epoch > args.lr_warmup_epochs:
            scheduler.step()

        # Evaluation
        model_s.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.half().to(accelerator.device)
                targets = targets.half().to(accelerator.device)
                outputs = model_s(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100 * correct / total

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                model_s.state_dict(),
                os.path.join(args.checkpoints_dir, f'model-best-epoch-{best_acc:.2f}.pt')
            )
        print(f"Test epoch: {epoch}, Test Acc: {test_acc:.2f}, Best Acc: {best_acc:.2f}")
    
    return best_acc



def load_classification_models(args, accelerator):
    """
    Load classification models.
    """
    num_classes = NUMS_CLASSES_DICT[args.data_type]
    transform = get_transforms(args.data_type)
    
    # Load teacher and student models
    model = get_model(
        name=args.model_name,
        num_classes=num_classes,
        pretrained_path=args.teacher_pretrained_path
    ).half()
    
    model_s = get_model(
        name=args.model_name_s,
        num_classes=num_classes,
        pretrained_path=None
    ).half()
    
    # Convert to SyncBatchNorm for distributed training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
    
    # Prepare for acceleration
    model, model_s = accelerator.prepare(model, model_s)
    
    # Register hooks for BatchNorm layers
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.SyncBatchNorm):
            hooks.append(DeepInversionHook(m))
    
    # Set evaluation mode
    model.eval()
    model_s.eval()
    
    return model, model_s, hooks, transform


def precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet, args):
    """
    Precompute text embeddings for each class.
    """
    class_text_embeddings = []
    class_syn_nums = []
    
    for class_index in tqdm(range(len(class_prompts))):
        text_inputs = tokenizer(
            [class_prompts[class_index]] * args.batch_size_generation,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(unet.device)
        text_embeddings = text_encoder(text_input_ids)[0]
        class_text_embeddings.append(text_embeddings)
        class_syn_nums.append(0)
    
    return class_text_embeddings, class_syn_nums



def generate_class_prompts(args):
    """
    Generate class prompts based on dataset type.
    """
    if args.label_name:
        return [
            PROMPT_DICT[args.data_type]["template"].replace("template*", label)
            for label in PROMPT_DICT[args.data_type]["label_name"]
        ]
    else:
        return [
            PROMPT_DICT[args.data_type]["template"].replace(" of template*", ".")
            for label in PROMPT_DICT[args.data_type]["label_name"]
        ]