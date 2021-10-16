import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    origin_path = os.path.join(data_dir, "waymo")
    # train
    directory = "train"
    path = os.path.join(data_dir, directory)
    try:
        os.mkdir(path)
        print("Directory '%s' created" %directory)
    except:
        print("Directory '%s' already present" %directory)
    
    file = open('train.txt', 'r')
    for line in file:
        origin = origin_path + '/' + line[:-1]
        destination = data_dir + 'train/' + line[:-1]
        #os.rename(origin, destination)   # move the file by renaming the path --> Did not work due to access right
        shutil.copy(origin, destination)  # let's create a copy
        
    # val
    directory = "val"
    path = os.path.join(data_dir, directory)
    try:
        os.mkdir(path)
        print("Directory '%s' created" %directory)
    except:
        print("Directory '%s' already present" %directory)
    
    file = open('validation.txt', 'r')
    for line in file:
        # move the file by renaming the path
        origin = origin_path + '/' + line[:-1]
        destination = data_dir + 'val/' + line[:-1]
        #os.rename(origin, destination)   # move the file by renaming the path --> Did not work due to access right
        shutil.copy(origin, destination)  # let's create a copy
        
    # test
    directory = "test"
    path = os.path.join(data_dir, directory)
    try:
        os.mkdir(path)
        print("Directory '%s' created" %directory)
    except:
        print("Directory '%s' already present" %directory)
    
    file = open('test.txt', 'r')
    for line in file:
        # move the file by renaming the path
        origin = origin_path + '/'  + line[:-1]
        destination = data_dir + 'test/' + line[:-1]
        # os.rename(origin, destination)  # move the file by renaming the path --> Did not work due to access right
        shutil.copy(origin, destination)  # let's create a copy
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('\nCreating splits...')
    split(args.data_dir)