import math
import os
import numpy as np


def get_all_file_paths(base):
    """ Get a list containing all file paths from folder base.

    Args:
        base:
            Folder name. For example: "./train/"

    Returns:
        A list containing all file paths.
        For example: ["./train/0.npy", "./train/data/0.npy", "./train/label/0.npy"]
    """
    results = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            results.append(os.path.join(root, f))
    return np.array(results)


def load_data_and_labels(data_dir_path, label_dir_path, label_index=()):
    """ Import data and label by dir paths.

    Args:
        data_dir_path:
            Data file folder name. For example: "./train/data"
        label_dir_path:
            Label file folder name. For example: "./train/label"
        label_index:
            Receive a tuple. Indicate label index which will be picked for training. Default: ()
        input_format:
            Receive ".npy" or ".txt".

    Returns:
        Two list containing datas and labels.
    """
    data, labels = [], []
    for data_path in get_all_file_paths(data_dir_path):
        data.append(np.loadtxt(data_path))

    for label_path in get_all_file_paths(label_dir_path):
        labels.append([np.loadtxt(label_path)[i] for i in label_index]
                      if len(label_index) != 0 else np.loadtxt(label_path))

    return np.array(data), np.array(labels)
