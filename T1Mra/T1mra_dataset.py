from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from misc_utils import get_matched_ids
import os


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
        Axis along which to slice the 3D MRI images. Default is "all",
        an int specifies axes as follows:
        0 = Sagittal (x-axis), 1 = Coronal (y-axis), 2 = Axial (z-axis)
    transform : callable, optional
        Transform to apply to the data
    split_char : str, optional
        Character to split the file names UID, default is "-"
    '''
    def __init__(self, mri_dir, mra_dir, transform, slice_axis=2,
                 split_char="-"):
        self.mri_dir = mri_dir
        self.mra_dir = mra_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.split_char = split_char

        self.mri_paths = [os.path.join(mri_dir, filename) for filename in
                          sorted(os.listdir(mri_dir))]
        self.mra_paths = [os.path.join(mra_dir, filename) for filename in
                          sorted(os.listdir(mra_dir))]

        self.id_list = self._get_id_list()

    def __len__(self):
        return self.id_list[-1].get("total_running_slices")

    def __getitem__(self, idx):
        # start at best guess for file index
        file_index = idx // self.id_list[0].get("total_running_slices")

        while file_index <= len(self):
            running_slices_at_file = self.id_list[file_index].get("total_running_slices")
            if running_slices_at_file > idx:
                # if we overshot, go back
                file_index -= 1
            else:
                # check if current file contains slice
                last_file_end = self.id_list[file_index - 1].get("total_running_slices")
                if last_file_end <= idx:
                    # we undershot, go forward
                    file_index += 1
                else:
                    # we found the right file
                    return self._get_slices(file_index, last_file_end + idx)

        raise IndexError("Index out of range, could not find slice")

    def _get_slices(self, file_idx, slice_idx):
        file = self.id_list[file_idx]
        mri_path = file.get("mri_path")
        mra_path = file.get("mra_path")

        mri_mmap = nib.load(mri_path, mmap=True)
        mra_mmap = nib.load(mra_path, mmap=True)

        mri_slice = mri_mmap.dataobj[:, :, slice_idx]
        mra_slice = mra_mmap.dataobj[:, :, slice_idx]

        mri_slice = self.transform(mri_slice)
        mra_slice = self.transform(mra_slice)

        return mri_slice, mra_slice

    def _get_id_list(self):

        ids = get_matched_ids([self.mri_dir, self.mra_dir],
                              split_char=self.split_char)
        id_list = []
        slices = 0
        for i, id in enumerate(ids):
            matching_mri = [path for path in self.mri_paths if id in path]
            matching_mra = [path for path in self.mra_paths if id in path]

            if len(matching_mri) == 1 and len(matching_mra) == 1:
                slices += self._get_num_slices(matching_mri[0])
                id_list.append({"mri_path": matching_mri[0],
                                "mra_path": matching_mra[0],
                                "total_running_slices": slices})
            else:
                raise ValueError(f"ID {id} has {len(matching_mri)} MRI images "
                                 f"and {len(matching_mra)} MRA images. There "
                                 f"should be exactly one of each.")
        return id_list

    def _get_num_slices(self, filepath):
        '''
        Returns the number of slices for the MRI image input
        '''
        if self.slice_axis == "all":
            shape = nib.load(filepath).get_fdata().shape()
            return sum(shape)
        else:
            return nib.load(filepath).get_fdata().shape[self.slice_axis]

    def _get_shape(self):
        '''
        Returns the shape of the MRI images
        '''
        key = next(iter(self.id_dict))
        path = self.id_dict[key]["mri_path"]
        return nib.load(path).get_fdata().shape()
