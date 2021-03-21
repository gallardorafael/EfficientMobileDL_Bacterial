"""
Code to create the train_set folder and the test_set folder
of a dataset.

Params:
    input_dir: Directory with the original images, i.e the vanilla version
        of the DIBaS dataset.
    output_dir: Directory to save the splits.
    train_size: Percentage of the samples assigned to the training set.
"""

import argparse
import os
import shutil
import random

# Arguments
parser = argparse.ArgumentParser(description='Options for the split')
parser.add_argument('--input_dir', default='./')
parser.add_argument('--output_dir', default='./')
parser.add_argument('--train_size', default=70)
args = parser.parse_args()

# Function to get every pair of image's path and labels.
def path_label(path_to_folder):
    data = list()
    for filename in os.listdir(path_to_folder):
        label = filename.split("_")[0]
        img = os.path.join(path_to_folder, filename)
        data.append((img, label))

    return data

# Function that receives pairs of paths and labels and
# return train and test splits of the tuples.
def train_test_paths(path_to_folder, train_percent):
    data = path_label(path_to_folder)
    print("Examples of specie:", len(data))
    train_size = round(len(data) * (train_percent / 100))
    # The original list if shuffled
    random.shuffle(data)
    # Creating the splits
    train_paths = data[:train_size]
    print("Train set size:",len(train_paths))
    res_paths =  data[train_size:]
    mid_index = len(res_paths) // 2
    val_paths = res_paths[:mid_index]
    print("Validation set size:",len(val_paths))
    test_paths = res_paths[mid_index:]
    print("Test set size:",len(test_paths))

    return train_paths, val_paths, test_paths

# Function that create the dirs with their files
def create_trainset_dirs(orig, dest, ts):
    try:
        os.mkdir(dest)
    except:
        pass
    try:
        os.mkdir(os.path.join(dest, "train"))
        os.mkdir(os.path.join(dest, "val"))
        os.mkdir(os.path.join(dest, "test"))
    except OSError as error:
        pass

    train_dest = os.path.join(dest, "train")
    val_dest = os.path.join(dest, "val")
    test_dest = os.path.join(dest, "test")

    for specie in os.listdir(orig):
        print("\t\t\tProcessing specie:", specie)
        # Origin path
        orig_specie_path = os.path.join(orig, specie)
        specie_train, specie_val, specie_test = train_test_paths(orig_specie_path, ts)

        # Creating path if not exist
        try:
            os.mkdir(os.path.join(train_dest, specie))
            os.mkdir(os.path.join(val_dest, specie))
            os.mkdir(os.path.join(test_dest, specie))
        except OSError as error:
            pass

        # Copiying train set
        print("Creating train split...")
        for i, example in enumerate(specie_train):
            filename = example[1] + "_" + str(i) + ".tif"
            dest_path = os.path.join(train_dest, specie)
            file_path = os.path.join(dest_path, filename)
            shutil.copyfile(example[0], file_path)

        # Copiying validation set
        print("Creating validation split...")
        for i, example in enumerate(specie_val):
            filename = example[1] + "_" + str(i) + ".tif"
            dest_path = os.path.join(val_dest, specie)
            file_path = os.path.join(dest_path, filename)
            shutil.copyfile(example[0], file_path)

        # Copying test set
        print("Creating test split...")
        for i, example in enumerate(specie_test):
            filename = example[1] + "_" + str(i) + ".tif"
            dest_path = os.path.join(test_dest, specie)
            file_path = os.path.join(dest_path, filename)
            shutil.copyfile(example[0], file_path)

create_trainset_dirs(args.input_dir, args.output_dir, int(args.train_size))
