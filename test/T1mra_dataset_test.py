import sys
import os

import pytest
import torch
import torchvision.transforms.v2 as v2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             'src')))
from T1mra_dataset import T1w2MraDataset, T1w2MraDataset_scans  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def test_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])


@pytest.fixture(scope='module')
def test_dataset(test_transform):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'MRA'),
                          test_transform)


@pytest.fixture(scope='module')
def matched_dataset(test_transform):
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'T1W'),
                          test_transform)


@pytest.fixture(scope='module')
def test_dataset_3d(test_transform):
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'MRA'),
                          test_transform,
                          slice_width=5)


@pytest.fixture(scope='module')
def matched_dataset_3d(test_transform):
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'T1W'),
                          test_transform,
                          slice_width=5)


@pytest.fixture(scope='module')
def test_dataset_3d_width_label(test_transform):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'MRA'),
                          test_transform,
                          slice_width=5,
                          width_labels=True)


@pytest.fixture(scope='module')
def matched_dataset_3d_width_label(test_transform):
    return T1w2MraDataset(os.path.join(current_dir, 'test_data', 'T1W'),
                          os.path.join(current_dir, 'test_data', 'T1W'),
                          test_transform,
                          slice_width=5,
                          width_labels=True)

@pytest.fixture(scope='module')
def test_dataset_scans(test_transform):
    return T1w2MraDataset_scans(os.path.join(current_dir, 'test_data', 'T1W'),
                                os.path.join(current_dir, 'test_data', 'MRA'),
                                test_transform)


@pytest.fixture(scope='module')
def test_dataset_scans_matched(test_transform):
    return T1w2MraDataset_scans(os.path.join(current_dir, 'test_data', 'T1W'),
                                os.path.join(current_dir, 'test_data', 'T1W'),
                                test_transform)


def test_dataset_len(test_dataset):
    assert len(test_dataset.scan_list)
    assert len(test_dataset) == 400


def test_dataset_3d_len(test_dataset_3d):
    assert len(test_dataset_3d.scan_list)
    assert len(test_dataset_3d) == 384


def test_dataset_3d_width_label_len(test_dataset_3d_width_label):
    assert len(test_dataset_3d_width_label.scan_list)
    assert len(test_dataset_3d_width_label) == 384


def test_first_last(test_dataset):
    assert test_dataset[0]
    assert test_dataset[len(test_dataset) - 1]

    with pytest.raises(IndexError):
        assert not test_dataset[len(test_dataset)]


def test_first_last_3d(test_dataset_3d):
    assert test_dataset_3d[0]
    assert test_dataset_3d[len(test_dataset_3d) - 1]

    with pytest.raises(IndexError):
        assert not test_dataset_3d[len(test_dataset_3d)]


def test_first_last_3d_width_label(test_dataset_3d_width_label):
    assert test_dataset_3d_width_label[0]
    assert test_dataset_3d_width_label[len(test_dataset_3d_width_label) - 1]

    with pytest.raises(IndexError):
        assert not test_dataset_3d_width_label[len(test_dataset_3d_width_label)]


def test_shape(test_dataset):
    indexs = [0, (len(test_dataset) // 2), len(test_dataset) - 1]
    for i in indexs:
        mri, mra = test_dataset[i]
        assert mri.shape == mra.shape
        assert mri.shape == (1, 512, 512)


def test_shape_3d(test_dataset_3d):
    indexs = [0, (len(test_dataset_3d) // 2), len(test_dataset_3d) - 1]
    for i in indexs:
        mri, mra = test_dataset_3d[i]
        assert mri.shape == (5, 512, 512)
        assert mra.shape == (1, 512, 512)


def test_shape_3d_width_label(test_dataset_3d_width_label):
    indexs = [0, (len(test_dataset_3d_width_label) // 2),
              len(test_dataset_3d_width_label) - 1]
    for i in indexs:
        mri, mra = test_dataset_3d_width_label[i]
        assert mri.shape == mra.shape
        assert mri.shape == (5, 512, 512)


def test_matched(matched_dataset):
    indexs = [0, (len(matched_dataset) // 2),
              len(matched_dataset) - 1]
    for i in indexs:
        mri, mra = matched_dataset[i]
        assert torch.allclose(mri, mra, atol=1e-6)


def test_matched_3d(matched_dataset_3d):
    indexs = [0, (len(matched_dataset_3d) // 2),
              len(matched_dataset_3d) - 1, 101]
    for i in indexs:
        mri, mra = matched_dataset_3d[i]
        middle_index = mri.shape[0] // 2
        mri_middle = mri[middle_index:middle_index + 1, :, :]
        assert torch.allclose(mri_middle, mra, atol=1e-6)


def test_matched_3d_width_label(matched_dataset_3d_width_label):
    indexs = [0, (len(matched_dataset_3d_width_label) // 2),
              len(matched_dataset_3d_width_label) - 1, 101]
    for i in indexs:
        mri, mra = matched_dataset_3d_width_label[i]
        assert torch.allclose(mri, mra, atol=1e-6)


def test_first_last_scans(test_dataset_scans):
    assert test_dataset_scans[0]
    assert test_dataset_scans[len(test_dataset_scans) - 1]

    with pytest.raises(IndexError):
        assert not test_dataset_scans[len(test_dataset_scans)]


def test_shape_scans(test_dataset_scans):
    for i in range(len(test_dataset_scans)):
        mri, mra = test_dataset_scans[i]
        assert mri.shape == mra.shape
        assert mri.shape == (100, 512, 512)


def test_matched_scans(test_dataset_scans_matched):
    for i in range(len(test_dataset_scans_matched)):
        mri, mra = test_dataset_scans_matched[i]
        assert torch.allclose(mri, mra, atol=1e-6)
