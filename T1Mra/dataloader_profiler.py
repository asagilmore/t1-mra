import time
import argparse
from os.path import join as pjoin

from .T1mra_dataset import T1w2MraDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4, help="Number of "
                        "workers for data loading")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch "
                        "size for training")
    parser.add_argument("--num_batches", type=int, default=10, help="Number "
                        "of batches to process")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to "
                        "the data directory")

    args = parser.parse_args()

    print("Profiling DataLoader performance"
          f" with {args.num_workers} workers and batch size {args.batch_size}"
          f" on {args.num_batches} batches")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    start_time = time.time()
    test_dataset = T1w2MraDataset(pjoin(args.data_dir, "T1W"),
                                  pjoin(args.data_dir, "MRA"),
                                  test_transform)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print("Time to create Dataset, "
          f"preload into mem: {elapsed_time:.4f} seconds")

    print(f'Length of dataset: {len(test_dataset)}')

    start_time = time.time()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print("Time to create DataLoader: "
          f"{elapsed_time:.4f} seconds")

    # warm up
    for i, _ in enumerate(test_dataloader):
        if i >= 5:
            break

    start_time = time.time()
    for i, _ in enumerate(test_dataloader):
        if i >= args.num_batches:
            break
    end_time = time.time()

    elapsed_time = end_time - start_time
    time_per_batch = elapsed_time / args.num_batches
    print(f"Time per batch: {time_per_batch:.4f} seconds")
    print(f"Total time for all batches: {elapsed_time:.4f} seconds")

