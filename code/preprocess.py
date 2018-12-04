import numpy as np
import os
import sys
# from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib

from NIIReader.Augmentation import *

data_path = './../dataset/processed'
train_path = './../dataset/processed/train'
val_path = './../dataset/processed/val'

def dataset_preprocess():
    ## Generate images
    image_paths, label_paths = getPaths()
    if len(os.listdir(data_path)) != 2 and len(os.listdir(data_path)) != 6722:
        dataAugmentation(image_paths, label_paths)


    ## Separate Train and Val data
    all_data_names = os.listdir(data_path)
    print('Total data number is', len(all_data_names))
    train_data_names = all_data_names[:int(len(all_data_names)*0.8)]
    print('Training data number is', len(train_data_names))
    val_data_names = all_data_names[int(len(all_data_names)*0.8):]
    print('Val data number is', len(val_data_names))

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        for f in train_data_names: 
            in_path = os.path.join(data_path, f)
            out_path = os.path.join(train_path, f)
            os.system('mv' + ' ' + in_path + ' ' + out_path)


    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        for f in val_data_names: 
            in_path = os.path.join(data_path, f)
            out_path = os.path.join(val_path, f)
            os.system('mv' + ' ' + in_path + ' ' + out_path)
    
def main():
    dataset_preprocess()
    
if __name__ == '__main__':
    main()