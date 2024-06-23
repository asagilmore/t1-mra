from t1mra_dataset import T1w2MraDataset
import torch
from torchvision import transforms


if __name__ == "__main__":
    mri_path = "/Users/asagilmore/src/t1-to-mra/data/raw-mini/T1W"
    mra_path = "/Users/asagilmore/src/t1-to-mra/data/raw-mini/MRA"

    dataset = T1w2MraDataset(mri_path, mra_path, slice_axis=2, transform=transforms.ToTensor())

    assert dataset is not None, "Dataset initialization failed."

    print(f'Total number of samples: {len(dataset)}')

    # first_item = dataset[0]
    # assert first_item is not None, "Accessing the first item failed."
    # assert isinstance(first_item, torch.Tensor), "The dataset item is not a torch.Tensor."


    # assert first_item.dtype == torch.float32, "The transformation to tensor failed."


    # 5. Iteration Test
    try:
        for item in dataset:
            pass
        print("Iteration test passed.")
    except Exception as e:
        print(f"Iteration test failed with error: {e}")