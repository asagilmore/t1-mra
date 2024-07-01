import os


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


def get_filepath_from_id(dir, id):
    dir_files = os.listdir(dir)
    for file in dir_files:
        if id in file:
            return os.path.join(dir, file)
