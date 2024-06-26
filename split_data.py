import argparse
import os
import shutil
from random import sample
from misc_utils import get_matched_ids, get_filepath_from_id


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training, testing, and validation folders.")
    parser.add_argument("--valid-percent", type=float, required=True, help="Percentage of data for validation set.")
    parser.add_argument("--test-percent", type=float, required=True, help="Percentage of data for test set.")
    parser.add_argument("--mra-dir", type=str, required=True, help="Directory containing MRA images.")
    parser.add_argument("--t1-dir", type=str, required=True, help="Directory containing T1 images.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for split datasets.")
    return parser.parse_args()


def split_data(valid_percent, test_percent, mra_dir, t1_dir, outdir):
    # Ensure output directories exist
    for category in ["train", "test", "valid"]:
        os.makedirs(os.path.join(outdir, category, "MRA"), exist_ok=True)
        os.makedirs(os.path.join(outdir, category, "T1W"), exist_ok=True)

    # List files in each directory
    mra_files = os.listdir(mra_dir)
    t1_files = os.listdir(t1_dir)

    if len(mra_files) != len(t1_files):
        raise ValueError("Number of files in MRA and T1 directories do not match.")

    # Calculate number of files for each set
    total_ids = get_matched_ids([mra_dir, t1_dir])
    valid_count = int(len(total_ids) * valid_percent)
    test_count = int(len(total_ids) * test_percent)
    train_count = len(total_ids) - (valid_count + test_count)

    # Randomly select files for each set
    valid_ids = sample(total_ids, valid_count)
    test_ids = sample(list(set(total_ids) - set(valid_ids)), test_count)
    train_ids = list(set(total_ids) - set(valid_ids) - set(test_ids))

    for id in valid_ids:
        mra_to_copy = [get_filepath_from_id(mra_dir, id) for id in valid_ids]
        t1_to_copy = [get_filepath_from_id(t1_dir, id) for id in valid_ids]
        mra_out_dir = os.path.join(outdir, "valid","MRA")
        t1_out_dir = os.path.join(outdir, "valid","T1W")
        for mra_file, t1_file in zip(mra_to_copy, t1_to_copy):
            shutil.copy(mra_file, mra_out_dir)
            shutil.copy(t1_file, t1_out_dir)

    for id in test_ids:
        mra_to_copy = [get_filepath_from_id(mra_dir, id) for id in test_ids]
        t1_to_copy = [get_filepath_from_id(t1_dir, id) for id in test_ids]
        mra_out_dir = os.path.join(outdir, "test","MRA")
        t1_out_dir = os.path.join(outdir, "test","T1W")
        for mra_file, t1_file in zip(mra_to_copy, t1_to_copy):
            shutil.copy(mra_file, mra_out_dir)
            shutil.copy(t1_file, t1_out_dir)
    for id in train_ids:
        mra_to_copy = [get_filepath_from_id(mra_dir, id) for id in train_ids]
        t1_to_copy = [get_filepath_from_id(t1_dir, id) for id in train_ids]
        mra_out_dir = os.path.join(outdir, "train","MRA")
        t1_out_dir = os.path.join(outdir, "train","T1W")
        for mra_file, t1_file in zip(mra_to_copy, t1_to_copy):
            shutil.copy(mra_file, mra_out_dir)
            shutil.copy(t1_file, t1_out_dir)


if __name__ == "__main__":
    args = parse_args()
    split_data(args.valid_percent, args.test_percent, args.mra_dir, args.t1_dir, args.outdir)