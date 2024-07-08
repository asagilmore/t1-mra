import argparse
import os
import logging

from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from T1mra_dataset import T1w2MraDataset
from PerceptualLoss import PerceptualLoss, VGG16FeatureExtractor
from Critic import Critic, gradient_penalty
from UNet import UNet
from train_utils import RandomRotation90

if __name__ == "__main__":

    writer = SummaryWriter()

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dir "
                        "containing training data. "
                        "Must have train, valid, test subdirectories")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch"
                        "size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number "
                        "of epochs")
    parser.add_argument("--preload_dtype", type=str, default="float32",)
    args = parser.parse_args()

    train_logger = logging.getLogger('train_logger')
    train_logger.setLevel(logging.INFO)
    train_file_handler = logging.FileHandler('training.log')
    train_file_handler.setLevel(logging.INFO)
    train_file_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    train_logger.addHandler(train_file_handler)

    # Setup for the second logger (e.g., for validation logs)
    validation_logger = logging.getLogger('validation_logger')
    validation_logger.setLevel(logging.INFO)
    validation_file_handler = logging.FileHandler('validation.log')
    validation_file_handler.setLevel(logging.INFO)
    validation_file_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    validation_logger.addHandler(validation_file_handler)

    # Example usage
    train_logger.info("Started Logging train")
    validation_logger.info("Started Logging validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_logger.info(f"Using device: {device}")

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        RandomRotation90(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    # transform to be used on outputs of generator
    critic_transform = v2.Compose([
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    # setup perceptual loss for identity loss
    feature_extractor = VGG16FeatureExtractor()
    feature_extractor.to(device)
    perceptual_loss = PerceptualLoss(feature_extractor, MSELoss)

    print(f'Loading datasets from {args.data_dir}')
    train_dataset = T1w2MraDataset(os.path.join(args.data_dir, "train", "T1W"),
                                   os.path.join(args.data_dir, "train", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype)
    valid_dataset = T1w2MraDataset(os.path.join(args.data_dir, "valid", "T1W"),
                                   os.path.join(args.data_dir, "valid", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype)

    train_gen_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
    train_critic_dataloader = DataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False)

    generator_model = UNet(1, 1).to(device)
    critic_model = Critic(2).to(device)

    # WGAN values from paper w/o batch_size
    learning_rate = 1e-4
    b1 = 0.5
    b2 = 0.999
    lambda_gp = 10

    # identity loss weight
    identity_weight = 0.1

    gen_optimizer = optim.Adam(generator_model.parameters(), lr=learning_rate,
                               betas=(b1, b2))
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=learning_rate,
                                  betas=(b1, b2))

    if not os.path.exists("model_checkpoint.pth"):
        epoch = 0
    else:
        checkpoint = torch.load("model_checkpoint.pth")
        generator_model.load_state_dict(checkpoint['gen_model_state_dict'])
        critic_model.load_state_dict(checkpoint['critic_model_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        critic_optimizer.load_state_dict(
                         checkpoint['critic_optimizer_state_dict'])
        epoch = checkpoint['epoch']

    for epoch in range(args.num_epochs):

        for gen_batch, critic_batch in zip(train_gen_dataloader,
                                           train_critic_dataloader):
            # Train critic
            generator_model.eval()
            critic_model.train()
            real_inputs, real_masks = gen_batch
            real_inputs = real_inputs.to(device)
            real_masks = real_masks.to(device)

            fake_masks = generator_model(real_inputs)
            fake_masks = critic_transform(fake_masks)

            real_concat = torch.cat([real_inputs, real_masks], dim=1)
            fake_concat = torch.cat([real_inputs, fake_masks], dim=1)

            real_validity = critic_model(real_concat)
            fake_validity = critic_model(fake_concat)

            critic_gp = gradient_penalty(critic_model, real_concat,
                                         fake_concat)

            critic_gp = lambda_gp * critic_gp
            critic_loss = -torch.mean(real_validity) + torch.mean(
                        fake_validity) + critic_gp

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Train generator
            generator_model.train()
            critic_model.eval()

            real_inputs, real_masks = critic_batch
            real_inputs = real_inputs.to(device)
            real_masks = real_masks.to(device)

            gen_optimizer.zero_grad()
            fake_masks = generator_model(real_inputs)
            fake_concat = torch.cat([real_inputs, fake_masks], dim=1)
            critic_validity = critic_model(fake_concat)
            gen_critic_loss = -torch.mean(critic_validity)

            second_output = generator_model(fake_masks)
            identity_loss = perceptual_loss.get_loss(second_output, real_masks)
            gen_loss = gen_critic_loss + (identity_weight * identity_loss)
            gen_loss.backward()
            gen_optimizer.step()

            print(f"Epoch {epoch} Gen Loss: {gen_loss.item()} Critic Loss: "
                  f"{critic_loss.item()} Identity Loss: "
                  f"{identity_loss.item()}")

            train_logger.info(f"Epoch {epoch} Gen Loss: {gen_loss.item()} "
                              f"Critic Loss: {critic_loss.item()} "
                              f"Identity Loss: {identity_loss.item()}")

    generator_model.eval()
    critic_model.eval()

    torch.save({
        'epoch': epoch+1,
        'gen_model_state_dict': generator_model.state_dict(),
        'critic_model_state_dict': critic_model.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict()
    }, "model_checkpoint.pth")

    running_critic_loss = 0.0
    running_gen_loss = 0.0
    running_identity_loss = 0.0
    for image, mask in valid_dataloader:
        image = image.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            fake_mask = generator_model(image)
            real_concat = torch.cat([image, mask], dim=1)
            fake_concat = torch.cat([image, fake_mask], dim=1)

            real_validity = critic_model(real_concat)
            fake_validity = critic_model(fake_concat)

            critic_loss = -torch.mean(real_validity) + torch.mean(
                                                     fake_validity)

            perceptual_loss = perceptual_loss.get_loss(fake_mask, mask)

            second_output = generator_model(fake_mask)
            gen_identity_loss = perceptual_loss.get_loss(second_output, mask)

            gen_critic_loss = -torch.mean(fake_validity)

            running_critic_loss += critic_loss.item()
            running_gen_loss += gen_critic_loss.item()
            running_identity_loss += gen_identity_loss.item()

    running_critic_loss /= len(valid_dataloader)
    running_gen_loss /= len(valid_dataloader)
    running_identity_loss /= len(valid_dataloader)

    validation_logger.info(f"Gen Critic Loss: {running_gen_loss} "
                           f"Gen Ident Loss: {running_identity_loss} "
                           f"Critic Loss: {running_critic_loss} ")
