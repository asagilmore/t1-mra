import argparse
import os
import logging
import multiprocessing

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.nn import MSELoss

from T1mra_dataset import T1w2MraDataset
from PerceptualLoss import PerceptualLoss, VGG16FeatureExtractor
from UNet import UNet


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device).float(), masks.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss/len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    return val_loss / len(loader), correct / len(loader.dataset)


if __name__ == "__main__":

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dir "
                        "containing training data. "
                        "Should have train, valid, test subdirectories")
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
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # check cpu count
    if args.num_workers == -1:
        num_workers = multiprocessing.cpu_count() - 1

    # def datasets/dataloaders
    train_dataset = T1w2MraDataset(os.path.join(args.data_dir, "train", "T1W"),
                                   os.path.join(args.data_dir, "train", "MRA"),
                                   transform=train_transform,
                                   num_workers=num_workers)
    valid_dataset = T1w2MraDataset(os.path.join(args.data_dir, "valid", "T1W"),
                                   os.path.join(args.data_dir, "valid", "MRA"),
                                   transform=train_transform,
                                   num_workers=num_workers)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False)

    # def model
    model = UNet(1, 1)
    model.to(device)

    # def optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs

    # load checkpoint if exists
    if os.path.exists("model_checkpoint.pth"):
        checkpoint = torch.load("model_checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_dataloader, perceptual_loss,
                           optimizer, device)
        val_loss, val_acc = validate(model, valid_dataloader, perceptual_loss,
                                     device)

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}, "
              "Val Acc: {val_acc}")
        logging.info(f"Epoch {epoch+1}, Loss: {train_loss}, "
                     "Val Loss: {val_loss}, Val Acc: {val_acc}")

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
