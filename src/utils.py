import h5py
import os
import git
import torch.nn as nn
import torch
import numpy as np
from math import floor, ceil

def git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return git_root


def get_model(vel_path=os.path.join(git_root(os.getcwd()), "data")):
    """
    Downloads and returns a velocity model
    """

    if not os.path.exists(vel_path):
        os.makedirs(vel_path)

    vel_model='parihaka_model_high-freq.h5'
    vel_file = os.path.join(vel_path, vel_model)

    if not os.path.isfile(vel_file):
        os.system("wget https://www.dropbox.com/s/eouo2awl156vc94/"
                  "parihaka_model_high-freq.h5 -O" + vel_file)

    spacing = (25.0, 12.5)

    dm = np.transpose(h5py.File(vel_file, 'r')['dm'][...])[1:, :]
    shape = list(dm.shape)
    dm = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)

    return dm, spacing, shape
