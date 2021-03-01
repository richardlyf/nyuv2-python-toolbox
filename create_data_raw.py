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
parser.add_argument("--data_path", "-d", dest="data_path", default='dataset', help="path to raw nyu data zip files")
args = parser.parse_args()

DATASET_DIR = Path(args.data_path)

def create_data_split_files(data_split, mode):
    print("Saving split files for mode: ", mode)
    with open(mode + '_files.txt', 'w') as f:
        for item in data_split:
            f.write("%s\n" % item)
    print("Done")

def main():
    # Archive is used to find image name using frame id without going through zip files
    archive = {}
    data_split = []
    # Extract all color images from the zip files
    for zip_path in DATASET_DIR.glob('*.zip'):
        raw_archive = RawDatasetArchive(zip_path)
        folder = os.path.basename(str(zip_path))[:-4] 
        archive[folder] = raw_archive.frames
        for frame_id, frame in tqdm(enumerate(raw_archive)):
            depth_path, color_path = Path('.') / frame[0], Path('.') / frame[1]
            color_full_path = DATASET_DIR / folder / color_path
            image_name = str(color_full_path)[:-3] + 'jpg'
            # only save color image if it's not already created
            if not os.path.exists(image_name):
                # only extract color image
                if not color_full_path.exists():
                    raw_archive.extract_frame([frame[1]], DATASET_DIR / folder)
                color = load_color_image(color_full_path)    
                # (480, 640, 3)
                # save color image and remove the unziped image file
                os.system('rm ' + str(color_full_path))
                if color is not None:
                    plt.imsave(image_name, color)
                else:
                    archive[folder].remove(frame)
                    continue
                
            # Ignore the first and last frame when counting for data split
            if frame_id != 0 and frame_id != len(raw_archive) - 1:
                data_split.append("{} {}".format(folder, frame_id))

    np.save("nyu_archive", archive, allow_pickle=True)
    # Break data into train, val, and test files
    total_frames_len = len(data_split)
    num_val = int(total_frames_len * 0.2)
    num_test = int(total_frames_len * 0.1)
    val_split = data_split[:num_val]
    test_split = data_split[num_val: num_val + num_test]
    train_split = data_split[num_val + num_test:]
    np.random.shuffle(val_split)
    np.random.shuffle(test_split)
    np.random.shuffle(train_split)
    create_data_split_files(val_split, "val")
    create_data_split_files(test_split, "test")
    create_data_split_files(train_split, "train")

if __name__ == '__main__':
    main()
    