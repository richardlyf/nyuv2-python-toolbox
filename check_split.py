#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from nyuv2 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", "-d", dest="data_path", default='dataset', help="path to data files")
parser.add_argument("--split_path", "-s", dest="split_path", help="path to the split files")
args = parser.parse_args()

DATASET_DIR = Path(args.data_path)

def main():
    archive = np.load(DATASET_DIR / "nyu_archive.npy", allow_pickle=True).item()
    fpath = os.path.join(args.split_path, "{}_files.txt")
    split_paths = [fpath.format(e) for e in ["train", "val", "test"]]
    for split_path in split_paths:
        print("Checking ", split_path)
        with open(split_path, 'r') as f:
            for line in tqdm(f):
                folder, frame_id = line.split()
                frame_id = int(frame_id)
                frames = archive[folder]
                for i in [-1, 0, 1]:
                    image_name = frames[frame_id + i][1][:-3] + '.jpg'
                    full_image_path = DATASET_DIR / folder / image_name
                    if not os.path.exists(full_image_path):
                        print(full_image_path, " doesn't exist. frame_id: ", frame_id, " i: ", i)


if __name__ == '__main__':
    main()
    