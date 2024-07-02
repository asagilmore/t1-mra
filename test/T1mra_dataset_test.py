import sys
import os

import pytest
import torch
import torchvision.transforms.v2 as v2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             'src')))
from T1mra_dataset import T1w2MraDataset  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))


def test_T1w2MraDataset():
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset_len = T1w2MraDataset(os.path.join(current_dir,
                                                   'test_dataset_len', 'T1W'),
                                      os.path.join(current_dir,
                                                   'test_dataset_len', 'MRA'),
                                      test_transform)
    assert len(test_dataset_len) == 100

    test_dataset = T1w2MraDataset(os.path.join(current_dir, 'test_data',
                                               'T1W'),
                                  os.path.join(current_dir, 'test_data',
                                               'MRA'),
                                  test_transform)

    assert len(test_dataset.scan_list)
    assert len(test_dataset) == 400

    assert test_dataset[0]
    assert test_dataset[len(test_dataset) - 1]

    with pytest.raises(IndexError):
        assert not test_dataset[len(test_dataset)]

    mri, mra = test_dataset[0]
    assert mri.shape == mra.shape
    assert mri.shape == (1, 512, 512)

    test_dataset_len = T1w2MraDataset(os.path.join(current_dir,
                                                   'test_dataset_len', 'T1W'),
                                      os.path.join(current_dir,
                                                   'test_dataset_len', 'MRA'),
                                      test_transform, slice_width=5)
    # length should account for padding
    assert len(test_dataset_len) == (100-4)
    mri, mra = test_dataset_len[0]
    assert mri.shape == (5, 512, 512)
    assert mra.shape == (1, 512, 512)

    # check last index
    mri, mra = test_dataset_len[len(test_dataset_len) - 1]
    assert mri.shape == (5, 512, 512)
    assert mra.shape == (1, 512, 512)

    test_dataset_len = T1w2MraDataset(os.path.join(current_dir,
                                                   'test_dataset_len', 'T1W'),
                                      os.path.join(current_dir,
                                                   'test_dataset_len', 'MRA'),
                                      test_transform, slice_width=5,
                                      width_labels=True)
    # length should account for padding
    assert len(test_dataset_len) == (100-4)
    mri, mra = test_dataset_len[0]
    assert mri.shape == (5, 512, 512)
    assert mra.shape == (5, 512, 512)
    assert mri.shape == mra.shape

