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

def load_train_test_split():
    file = sio.loadmat(DATASET_DIR / 'splits.mat')
    train_idxs = file['trainNdxs']
    test_idxs = file['testNdxs']
    return train_idxs, test_idxs

def create_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'train/image', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'train/depth', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'test/image', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'test/depth', exist_ok=True)

def create_data(dataset, data_type, indices):
    frame_index = 0
    for id in tqdm(indices):
        color, depth = dataset[id.item()]
        image_name = "{:3d}.jpg".format(frame_index)
        plt.imsave(OUTPUT_DIR / data_type / 'image' / image_name, color)
        plt.imsave(OUTPUT_DIR / data_type / 'depth' / image_name, depth)
        frame_index += 1
        
if __name__ == '__main__':
    create_output_dirs()
    labeled_dataset = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')
    train_idxs, test_idxs = load_train_test_split()
    create_data(labeled_dataset, 'train', train_idxs)
    create_data(labeled_dataset, 'test', test_idxs)
    labeled_dataset.close()