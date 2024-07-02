import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             'src')))

from misc_utils import get_matched_ids, get_filepath_from_id  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_matched_ids():
    dirs = [
        os.path.join(current_dir, "./test_matched_ids", "MRA"),
        os.path.join(current_dir, "./test_matched_ids", "T1W")
    ]

    should_contain = ['IXI055', 'IXI056', 'IXI057', 'IXI058']
    should_not_contain = ['IXI059', 'IXI060']
    matched_ids = get_matched_ids(dirs)
    for id in should_contain:
        assert id in matched_ids

    for id in should_not_contain:
        assert id not in matched_ids


def test_get_filepath_from_id():
    dir = os.path.join(current_dir, "test_data", "MRA")
    id = "IXI055"
    filepath = get_filepath_from_id(dir, id)
    assert filepath == os.path.join(current_dir, "test_data", "MRA",
                                    "IXI055-MRA.nii.gz")
