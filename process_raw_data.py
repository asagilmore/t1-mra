# resample to t1w space using nibable. Source: https://neuroimaging-data-science.org/content/005-nipy/003-transforms.html
import os
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm
import argparse


def get_matched_ids(dirs, split_char="-"):
    '''
    returns a sorted set of all ids that exist in all given dirs
    '''
    files = [os.listdir(dir) for dir in dirs]
    file_ids = [[file.split(split_char)[0] for file in file_list] for
                file_list in files]
    sets = [set(file_id) for file_id in file_ids]
    matched = set.intersection(*sets)
    return sorted(matched)


def process_data(T1_dir, MRA_dir, T1_out_dir, MRA_out_dir):

    T1_files = [file for file in os.listdir(T1_dir) if not file.startswith('.')]
    MRA_files = [file for file in os.listdir(MRA_dir) if not file.startswith('.')]

    matched_ids = get_matched_ids([T1_dir, MRA_dir])

    for id in tqdm(matched_ids):
        T1_file = [file for file in T1_files if file.startswith(id)]
        MRA_file = [file for file in MRA_files if file.startswith(id)]

        if len(T1_file) == 1 and len(MRA_file) == 1:
            T1_file = T1_file[0]
            MRA_file = MRA_file[0]

            T1_img = nib.load(os.path.join(T1_dir, T1_file))
            MRA_img = nib.load(os.path.join(MRA_dir, MRA_file))

            ## upsample mra to t1w resolution
            T1W_resampled = resample_from_to(T1_img, MRA_img,mode='nearest')

            mra_mask = MRA_img.get_fdata() > 0

            nib.save(T1W_resampled, os.path.join(T1_out_dir, f"{id}-T1W-resampled.nii.gz"))
            nib.save(MRA_img, os.path.join(MRA_out_dir, f"{id}-MRA.nii.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--T1_dir", type=str, help="Path to T1 directory")
    parser.add_argument("--MRA_dir", type=str, help="Path to MRA directory")
    parser.add_argument("--T1_out_dir", type=str, help="Path to T1 output directory")
    parser.add_argument("--MRA_out_dir", type=str, help="Path to MRA output directory")
    args = parser.parse_args()

    process_data(args.T1_dir, args.MRA_dir, args.T1_out_dir, args.MRA_out_dir)