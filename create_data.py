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
parser.add_argument("--data_path", "-d", dest="data_path", default='dataset', help="path to labeled nyu data .mat file")
parser.add_argument("--output_path", "-o", dest="output_path", required=True, help="where the output data is located")
args = parser.parse_args()

DATASET_DIR = Path(args.data_path)
OUTPUT_DIR = Path(args.output_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'image', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'depth', exist_ok=True)
    labeled_dataset = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')

    test_file = []
    frame_index = 0
    for i in tqdm(range(len(labeled_dataset))):
        color, depth = labeled_dataset[i]
        image_name = "{:3d}.jpg".format(i)
        depth_name = "{:3d}.npy".format(i)
        plt.imsave(OUTPUT_DIR / 'image' / image_name, color)
        np.save(OUTPUT_DIR / 'depth' / depth_name, depth, allow_pickle=True)
        frame_index += 1
        test_file.append("{} {}".format('image', i))

    print("Saving split files")
    with open('test_files.txt', 'w') as f:
        for item in test_file:
            f.write("%s\n" % item)
    print("Done")
    labeled_dataset.close()

if __name__ == '__main__':
    main()
