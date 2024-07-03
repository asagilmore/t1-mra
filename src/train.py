import argparse
import os
import logging
import multiprocessing
import psutil

from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
import torch.optim as optim
from torch.nn import MSELoss

from T1mra_dataset import T1w2MraDataset
from PerceptualLoss import PerceptualLoss, VGG16FeatureExtractor
from UNet import UNet
from train_utils import train, validate

if __name__ == "__main__":

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dir "
                        "containing training data. "
                        "Must have train, valid, test subdirectories")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch"
                        "size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number "
                        "of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Number "
                        "of epochs to wait for improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum change to qualify as an improvement")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number "
                        "of workers for dataloader")
    parser.add_argument("--preload_dtype", type=str, default="float32",)
    args = parser.parse_args()

    # Early stopping parameters
    patience = args.patience
    min_delta = args.min_delta
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # logging
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setup perceptual loss
    feature_extractor = VGG16FeatureExtractor()
    feature_extractor.to(device)
    perceptual_loss = PerceptualLoss(feature_extractor, MSELoss)

    # def transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
    # check cpu count
    if args.num_workers == -1:
        num_workers = multiprocessing.cpu_count() - 1
    else:
        num_workers = args.num_workers

    # def datasets/dataloaders
    print(f'Loading datasets from {args.data_dir}')
    train_dataset = T1w2MraDataset(os.path.join(args.data_dir, "train", "T1W"),
                                   os.path.join(args.data_dir, "train", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype)
    valid_dataset = T1w2MraDataset(os.path.join(args.data_dir, "valid", "T1W"),
                                   os.path.join(args.data_dir, "valid", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False)

    # Print current memory usage
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    print(f"Current memory usage: {current_memory / (1024**3)} GB")

    # def model
    model = UNet(1, 1)
    model.to(device)

    # def optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs

    # load checkpoint if exists
    if os.path.exists("model_checkpoint.pth"):
        print("Model checkpoint found, loading")
        checkpoint = torch.load("model_checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f'Loaded model, starting from epoch {start_epoch}')
    else:
        start_epoch = 0

    print(f'Starting training for {num_epochs} epochs')
    # training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_dataloader, perceptual_loss.get_loss,
                           optimizer, device)
        val_loss, val_acc = validate(model, valid_dataloader,
                                     perceptual_loss.get_loss, device)

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}")
        logging.info(f"Epoch {epoch+1}, Loss: {train_loss}, "
                     f"Val Loss: {val_loss}")

        # save model checkpoint
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "model_checkpoint.pth")

        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping")
            logging.info("Early stopping at epoch {epoch+1}")
            break
