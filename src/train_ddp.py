import argparse
import os

from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms.v2 as v2
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from T1mra_dataset import T1w2MraDataset
from PerceptualLoss import PerceptualLoss, VGG16FeatureExtractor
from UNet import UNet
from train_utils import tensorboard_write


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_transforms():
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

    return train_transform


def get_dataloader(data_dir, batch_size, num_workers, preload_dtype,
                   transforms, world_size, rank):

    dataset = T1w2MraDataset(os.path.join(data_dir, "T1W"),
                             os.path.join(data_dir, "MRA"),
                             transform=transforms,
                             preload_dtype=preload_dtype)
    sample = DistributedSampler(dataset, num_replicas=world_size,
                                rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample)

    return dataloader


def validate(rank, model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, target in val_loader:
            inputs, target = inputs.to(rank), target.to(rank)
            output = model(inputs)
            loss = criterion(output, target)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    loss_tensor = torch.tensor([total_loss, total_samples],
                               dtype=torch.float32,
                               device=rank)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_samples = loss_tensor.tolist()
    avg_loss = total_loss / total_samples

    return avg_loss


def train(rank, world_size, epochs, batch_size,
          data_dir, lr, num_workers, preload_dtype,
          start_epoch=0):

    setup(rank, world_size)

    # def model
    model = UNet(1, 1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # get dataloaders
    train_dataloader = get_dataloader(os.path.join(data_dir, "train"),
                                      batch_size, num_workers, preload_dtype,
                                      get_transforms(), world_size, rank)
    valid_dataloader = get_dataloader(os.path.join(data_dir, "valid"),
                                      batch_size, num_workers, preload_dtype,
                                      get_transforms(), world_size, rank)

    # setup perceptual loss
    feature_extractor = VGG16FeatureExtractor()
    feature_extractor.to(rank)
    perceptual_loss = PerceptualLoss(feature_extractor, MSELoss)

    criterion = perceptual_loss.get_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    if rank == 0:
        writer = SummaryWriter()

    for epoch in range(start_epoch, epochs):
        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for inputs, target in train_dataloader:
            inputs, target = inputs.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        loss = running_loss / len(train_dataloader)
        val_loss = validate(rank, ddp_model, valid_dataloader, criterion)
        if rank == 0:
            scheduler.step(val_loss)
            print(f"Epoch {epoch}, Loss: {loss}"
                  f"Val Loss: {val_loss}")

        for param_group in optimizer.param_groups:
            lr_tensor = torch.tensor(param_group['lr'],
                                     device=rank)
            dist.broadcast(lr_tensor, src=0)
            param_group['lr'] = lr_tensor.item()

        if rank == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler
                else None,
            }, "model_checkpoint.pth")

            tensorboard_write(writer, rank, loss, val_loss, epoch+1,
                              model, valid_dataloader,
                              num_images=10,
                              adam_optim=optimizer)


if __name__ == "__main__":

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dir "
                        "containing training data. "
                        "Must have train, valid, test subdirectories")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch"
                        "size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number "
                        "of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number "
                        "of workers for dataloader")
    parser.add_argument("--preload_dtype", type=str, default="float32",)
    args = parser.parse_args()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        raise RuntimeError("No GPUs available"
                           "cuda is required for DDP")

    train_args = (world_size, args.num_epochs, args.batch_size,
                  args.data_dir, args.lr, args.num_workers,
                  args.preload_dtype)
    print(f"Running DDP with {world_size} GPUs, "
          f"Training for {args.num_epochs} epochs")
    mp.spawn(train, args=train_args, nprocs=world_size, join=True)
