from torch.utils.data import Dataset
import nibabel as nib
from misc_utils import get_matched_ids
import os
import torch


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
    as_tensor : bool, optional
        Whether to preload scans as tensors,
        or numpy float arrays, default False
    '''
    def __init__(self, mri_dir, mra_dir, transform, slice_axis=2,
                 split_char="-", as_tensor=False):
        self.mri_dir = mri_dir
        self.mra_dir = mra_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.split_char = split_char

        self.mri_paths = [os.path.join(mri_dir, filename) for filename in
                          sorted(os.listdir(mri_dir))]
        self.mra_paths = [os.path.join(mra_dir, filename) for filename in
                          sorted(os.listdir(mra_dir))]

        self.scan_list = self._load_scan_list(as_tensor=as_tensor)

    def __len__(self):
        return self.id_list[-1].get("total_running_slices")

    def __getitem__(self, idx):
        # start at best guess for file index
        file_index = idx // self.id_list[0].get("total_running_slices")

        while file_index <= len(self):
            running_slices_at_file = self.id_list[file_index].get(
                                           "total_running_slices")
            if running_slices_at_file > idx:
                # if we overshot, go back
                file_index -= 1
            else:
                # check if current file contains slice
                last_file_end = self.id_list[file_index - 1].get(
                                          "total_running_slices")
                if last_file_end <= idx:
                    # we undershot, go forward
                    file_index += 1
                else:
                    # we found the right file
                    return self._get_slices(file_index, last_file_end + idx)

        raise IndexError("Index out of range, could not find slice")

    def _get_slices(self, file_idx, slice_idx):
        mri_scan = self.scan_list[file_idx].get("mri")
        mra_scan = self.scan_list[file_idx].get("mra")

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

    def _load_scan_list(self, as_tensor=False):
        '''
        Preloader for scans

        Returns a list of dictionaries containing the loaded mri scans,
        aswell as the slice count

        Parameters
        ----------
        as_tensor : bool, optional
            Whether to return the scans as tensors,
            or numpy float arrays, default False
        '''
        ids = get_matched_ids([self.mri_dir, self.mra_dir],
                              split_char=self.split_char)
        scan_list = []
        slices = 0
        for i, id in enumerate(ids):
            matching_mri = [path for path in self.mri_paths if id in path]
            matching_mra = [path for path in self.mra_paths if id in path]

            if len(matching_mri) == 1 and len(matching_mra) == 1:
                slices += self._get_num_slices(matching_mri[0])
                mri_scan = nib.load(matching_mri[0]).get_fdata()
                mra_scan = nib.load(matching_mra[0]).get_fdata()

                mri_slices = self._get_num_slices(matching_mri[0])
                mra_slices = self._get_num_slices(matching_mra[0])
                if mri_slices != mra_slices:
                    raise ValueError(f"ID {id} has {mri_slices} MRI slices "
                                     f"and {mra_slices} MRA slices. They "
                                     f"should be equal.")
                else:
                    slices = mri_slices

                if as_tensor:
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")

                    mri_scan = mri_scan.to(device)
                    mra_scan = mra_scan.to(device)

                scan_list.append({"mri": mri_scan, "mra": mra_scan,
                                  "total_running_slices": slices})
            else:
                raise ValueError(f"ID {id} has {len(matching_mri)} MRI images "
                                 f"and {len(matching_mra)} MRA images. There "
                                 f"should be exactly one of each.")

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
