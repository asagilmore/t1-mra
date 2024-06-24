# resample to t1w space using nibable. Source: https://neuroimaging-data-science.org/content/005-nipy/003-transforms.html
import os
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm


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


T1_dir = "/Users/asagilmore/src/t1-mra/raw-data/T1W"
MRA_dir = "/Users/asagilmore/src/t1-mra/raw-data/MRA"

T1_files = [file for file in os.listdir(T1_dir) if not file.startswith('.')]
MRA_files = [file for file in os.listdir(MRA_dir) if not file.startswith('.')]

T1_out_dir = "/Users/asagilmore/src/t1-mra/processed-data/T1W"
MRA_out_dir = "/Users/asagilmore/src/t1-mra/processed-data/MRA"

matched_ids = get_matched_ids([T1_dir, MRA_dir])

for id in tqdm(matched_ids):
    T1_file = [file for file in T1_files if file.startswith(id)]
    MRA_file = [file for file in MRA_files if file.startswith(id)]

    if len(T1_file) == 1 and len(MRA_file) == 1:
        T1_file = T1_file[0]
        MRA_file = MRA_file[0]

        T1_img = nib.load(os.path.join(T1_dir, T1_file))
        MRA_img = nib.load(os.path.join(MRA_dir, MRA_file))

        print(f"t1w shape: {T1_img.shape}")
        print(f"mra shape: {MRA_img.shape}")
        ## upsample mra to t1w resolution
        T1W_resampled = resample_from_to(T1_img, MRA_img,mode='nearest')

        mra_mask = MRA_img.get_fdata() > 0
        print(f"T1W resamp shape: {T1W_resampled.shape}")
        print(f"mra mask shape: {mra_mask.shape}")

        nib.save(T1W_resampled, os.path.join(T1_out_dir, f"{id}-T1W-resampled.nii.gz"))
        nib.save(MRA_img, os.path.join(MRA_out_dir, f"{id}-MRA.nii.gz"))
