import argparse
import os
import logging

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.nn import MSELoss

from T1mra_dataset import T1w2MraDataset
from UNet import UNet
from PerceptualLoss import VGG16FeatureExtractor, PerceptualLoss



if __name__ == "__main__":

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data"
                                                                    "should have train, valid, test subdirectories")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()


    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")cd

    # setup perceptual loss
    feature_extractor = VGG16FeatureExtractor()
    feature_extractor.to(device)
    perceptual_loss = PerceptualLoss(feature_extractor, MSELoss)

    # def transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    # def datasets/dataloaders
    train_dataset = T1w2MraDataset(os.path.join(args.train_dir, "train", "T1W"),
                                   os.path.join(args.train_dir, "train", "MRA"),
                                   transform=train_transform)
    valid_dataset = T1w2MraDataset(os.path.join(args.valid_dir, "valid", "T1W"),
                                   os.path.join(args.valid_dir, "valid", "MRA"),
                                   transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

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
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = perceptual_loss.get_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

        torch.