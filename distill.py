import os
import sys
import wandb
import torch
from tqdm import tqdm

from helper.utils import setup_seed, test_accuracy, KLDiv, distill_train, adjust_learning_rate
from helper.dataset import get_dataset
from helper.dataset import NUMS_CLASSES_DICT
from helper.models import get_model
from options.distill_options import DistillOptions
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main(args):
    # Redirect stdout to log file if logging is enabled
    if args.save_log == 1:
        output_file_path = os.path.join(args.checkpoints_dir, "log.txt")
        sys.stdout = open(output_file_path, 'w')

    # Initialize Weights & Biases (wandb) for experiment tracking if enabled
    if args.use_wandb:
        wandb.login(key="writer your keys")
        wandb.init(config=args, project="DiffDFKD", group="val-idea", name=args.name)
    
    print(vars(args))
    
    # Load dataset
    num_classes = NUMS_CLASSES_DICT[args.data_type]
    train_dataset, test_dataset = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load teacher and student models
    teacher = get_model(args.t_model, num_classes, pretrained_path=args.teacher_pretrained_path)
    student = get_model(args.s_model, num_classes, pretrained_path=None)

    # Evaluate and log teacher model accuracy
    test_acc, _ = test_accuracy(teacher, test_loader)
    print(f"Loading teacher accuracy: {test_acc:.2f}")

    # Define loss function (KL Divergence) and optimizer
    criterion = KLDiv(T=args.T)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    # Initialize tracking for best accuracy
    best_acc = -1

    # Distillation training loop
    for epoch in tqdm(range(args.epochs)):
        student.train()
        distill_train([student, teacher], train_loader, criterion, optimizer, if_log=False, args=args)
        adjust_learning_rate(epoch, args, optimizer)

        # Evaluate student model
        student.eval()
        test_acc, test_loss = test_accuracy(student, test_loader)
        
        # Save the best student model checkpoint
        if test_acc > best_acc:
            torch.save(student.state_dict(), os.path.join(args.checkpoints_dir, "model_best.pth"))
            best_acc = test_acc
        
        print(f"Student best, epoch {epoch}, acc: {test_acc:.2f}, best: {best_acc:.2f}, test_loss: {test_loss:.2f}")
        
        # Log test results to wandb if enabled
        if args.use_wandb:
            wandb.log({'test accuracy': test_acc})
    
    print("Training Completed!")

    # Close log file and restore stdout if logging was enabled
    if args.save_log == 1:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    print(f"Student best accuracy: {best_acc:.2f}")

if __name__ == '__main__':
    # Parse training options
    args = DistillOptions().parse()
    
    # Set random seed for reproducibility
    setup_seed(args.seed)
    
    # Start training process
    main(args)