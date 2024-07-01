import os
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset
import nibabel as nib
from tqdm import tqdm
import numpy as np

from misc_utils import get_matched_ids


class T1w2MraDataset(Dataset):
    '''
    Dataset class for T1w to MRA data

    Parameters
    ----------
    mri_dir : str
        Path to the directory containing the T1w MRI images
    mra_dir : str
        Path to the directory containing the MRA images
    slice_axis : int or string, optional
        Axis along which to slice the 3D MRI images. Default is "2",
        an int specifies axes as follows:
        0 = Sagittal (x-axis), 1 = Coronal (y-axis), 2 = Axial (z-axis)
    transform : callable, optional
        Transform to apply to the data
    split_char : str, optional
        Character to split the file names UID, default is "-"
    '''
    def __init__(self, mri_dir, mra_dir, transform, slice_axis=2,
                 split_char="-", preload_dtype=np.float16):
        self.mri_dir = mri_dir
        self.mra_dir = mra_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.split_char = split_char
        self.preload_dtype = preload_dtype

        self.shape_frequencies = {}
        self.mri_paths = [os.path.join(mri_dir, filename) for filename in
                          sorted(os.listdir(mri_dir))]
        self.mra_paths = [os.path.join(mra_dir, filename) for filename in
                          sorted(os.listdir(mra_dir))]

        self.scan_list = self._load_scan_list()

    def __len__(self):
        if self.scan_list:
            return self.scan_list[-1].get("last_index") + 1
        else:
            return 0

    def __getitem__(self, idx):
        # handle negative indexing
        if idx < 0:
            idx = len(self) + idx

        scan_to_use = None
        for scan in self.scan_list:
            if idx <= scan.get("last_index") and idx >= scan.get("first_index"):
                scan_to_use = scan
                break

        if scan_to_use is None:
            raise IndexError(f"Index {idx} out of range for dataset")

        slice_index = scan_to_use.get("first_index") - idx

        return self._get_slices(scan_to_use, slice_index)

    def _get_slices(self, scan_object, slice_idx):

        mri_scan = scan_object.get("mri")
        mra_scan = scan_object.get("mra")

        if self.slice_axis == 0:
            mri_slice = mri_scan[slice_idx, :, :]
            mra_slice = mra_scan[slice_idx, :, :]
        elif self.slice_axis == 1:
            mri_slice = mri_scan[:, slice_idx, :]
            mra_slice = mra_scan[:, slice_idx, :]
        elif self.slice_axis == 2:
            mri_slice = mri_scan[:, :, slice_idx]
            mra_slice = mra_scan[:, :, slice_idx]

        mri_slice = self.transform(mri_slice)
        mra_slice = self.transform(mra_slice)

        return mri_slice, mra_slice

    def _update_shape_frequencies(self, shape):
        if shape not in self.shape_frequencies:
            self.shape_frequencies[shape] = 1
        else:
            self.shape_frequencies[shape] += 1

    def _load_scan(self, id):
        matching_mri = [path for path in self.mri_paths if id in path]
        matching_mra = [path for path in self.mra_paths if id in path]

        if len(matching_mri) == 1 and len(matching_mra) == 1:
            mri_scan = nib.load(matching_mri[0]).get_fdata(
                                                    dtype=self.preload_dtype)
            mra_scan = nib.load(matching_mra[0]).get_fdata(
                                                    dtype=self.preload_dtype)

            # update shape frequencies
            self._update_shape_frequencies(mri_scan.shape)
            self._update_shape_frequencies(mra_scan.shape)

            mri_slices = self._get_num_slices(matching_mri[0])
            mra_slices = self._get_num_slices(matching_mra[0])

            if mri_slices != mra_slices:
                raise ValueError(f"ID {id} has {mri_slices} MRI slices "
                                 f"and {mra_slices} MRA slices. They "
                                 "should be equal.")
            else:
                slices = mri_slices

        else:
            raise ValueError("Multiple scans found for ID "
                             f"there should be only one MRA & T1 for {id}")

        return {'mri': mri_scan, 'mra': mra_scan, 'slices': slices}

    def _load_scan_list(self):
        ids = get_matched_ids([self.mri_dir, self.mra_dir],
                              split_char=self.split_char)

        # mutlthreading starts here
        with ThreadPoolExecutor() as executor:
            result_list = list(tqdm(executor.map(self._load_scan, ids),
                                    total=len(ids)))
            # we now have a list as follows:
            # [{'mri': mri_scan, 'mra': mra_scan, 'slices': slices}, ...]

        # now we count up the slices and add the first and last index
        scan_list = []
        slices = 0
        for i, result in enumerate(result_list):
            first_index = slices
            slices += result.get('slices')
            last_index = slices - 1
            scan_list.append({'mri': result.get('mri'), 'mra': result.get('mra'),
                              'last_index': last_index,
                              'first_index': first_index})

        return scan_list

    def _get_num_slices(self, scan):
        '''
        Returns the number of slices for the MRI image input
        '''
        # Check if scan is a filepath (string), then load it; otherwise, use it directly
        if isinstance(scan, str):
            scan_data = nib.load(scan).get_fdata()
        else:
            scan_data = scan

        # Handle the 'all' case or a specific axis
        if self.slice_axis == "all":
            shape = scan_data.shape
            return sum(shape)
        else:
            return scan_data.shape[self.slice_axis]
