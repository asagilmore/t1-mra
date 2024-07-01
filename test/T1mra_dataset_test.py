import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             'src')))
from T1mra_dataset import T1w2MraDataset  # noqa: E402
from torchvision import transforms  # noqa: E402


def test_T1w2MraDataset():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset_len = T1w2MraDataset("./test_dataset_len/T1w",
                                      "./test_dataset_len/MRA",
                                      test_transform)
    assert len(test_dataset_len) == 100

    test_dataset = T1w2MraDataset("./test_dataset/T1w",
                                  "./test_dataset/MRA",
                                  test_transform)

    assert 'IXI055' in test_dataset.id_list
    assert 'IXI060' not in test_dataset.id_list

    assert test_dataset[0]
    assert test_dataset[len(test_dataset) - 1]
    assert not test_dataset[len(test_dataset)]

    mri, mra = test_dataset[0]
    assert mri.shape == mra.shape
    assert mri.shape == (512, 512)
