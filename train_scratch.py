import os
import wandb
import torch
import torch.optim as optim
from helper.utils import setup_seed, train, test_accuracy
from helper.dataset import get_dataset
from helper.models import get_model
from tqdm import tqdm
from helper.dataset import NUMS_CLASSES_DICT
from options.train_scratch_options import TrainScratchOptions
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Parse training arguments
    args = TrainScratchOptions().parse()
    
    # Set random seed for reproducibility
    setup_seed(args.seed)

    # Initialize Weights & Biases (wandb) for experiment tracking if enabled
    if args.wandb == 1:
        wandb.init(config=args, project="DiffDFKD", group="train from scratch", name=args.name)

    # Load dataset
    train_dataset, test_dataset = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # Retrieve model and move to GPU
    num_classes = NUMS_CLASSES_DICT[args.data_type]
    model = get_model(args.model_name, num_classes)
    model.to('cuda')
    
    # Define optimizer (SGD with momentum and weight decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Initialize tracking variables for best accuracy
    best_acc = 0
    best_acc_epoch = -1

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        # Train model on the training dataset
        train(model, trainloader, optimizer, if_log=True)

        # Evaluate model on the test dataset
        acc, test_loss = test_accuracy(model, testloader, if_log=True)

        # Log test results to wandb if enabled
        if args.wandb == 1:
            wandb.log({"test_loss": test_loss / args.batch_size})
            wandb.log({"test_acc": acc})

        # Save the best model checkpoint
        if acc > best_acc:
            # Remove previous best model checkpoint
            previous_best_model = os.path.join(
                args.checkpoints_dir, f'model-best-epoch-{best_acc_epoch}-{best_acc:.2f}.pt'
            )
            if os.path.exists(previous_best_model):
                os.remove(previous_best_model)

            # Update best accuracy and save new best model
            best_acc = acc
            best_acc_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoints_dir, f'model-best-epoch-{best_acc_epoch}-{best_acc:.2f}.pt')
            )
    
        # Print progress
        print(f"Epoch {epoch}/{args.epochs} - Test Loss: {round(test_loss, 2)} | Test Acc: {round(acc, 2)} / Best Acc: {round(best_acc, 2)}")