#!/usr/bin/python

import os
import sys
import math
import random
import numpy as np
import cv2

# Import Mask RCNN
ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils


class BrainsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Brain"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_CHANNEL_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = False
    
    # Optimizing GPU Memory
    # Link: https://github.com/matterport/Mask_RCNN/wiki
    
#    BACKBONE = "resnet50"
#    model.train(..., layers='heads', ...)
#    IMAGE_MIN_DIM = 800
#    IMAGE_MAX_DIM = 1024
#    GPU_COUNT = 1
#    IMAGES_PER_GPU = 2
#    TRAIN_ROIS_PER_IMAGE = 200
#    MAX_GT_INSTANCES = 100
#    IMAGE_RESIZE_MODE = "crop"
#    IMAGE_MIN_DIM = 512
#    IMAGE_MAX_DIM = 512


class BrainsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_brains(self, data_folder):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # TODO: Update with 10 classes
        self.add_class("brains", 0, "0")
        self.add_class("brains", 1, "1")
        self.add_class("brains", 2, "2")
        self.add_class("brains", 3, "3")
        self.add_class("brains", 4, "4")
        self.add_class("brains", 5, "5")
        self.add_class("brains", 6, "6")
        self.add_class("brains", 7, "7")
        self.add_class("brains", 8, "8")
        self.add_class("brains", 9, "9")
        self.add_class("brains", 10, "10")

        # Add images
        img_paths = [os.path.join(data_folder, x) for x in os.listdir(data_folder)]
        for i in range(len(img_paths)):
            self.add_image('brains', image_id = i, path = img_paths[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        path = self.image_info[image_id]['path']
        image = np.load(path)[0]
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "brains":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        seg = np.load(info['path'])[1]
        class_list = np.unique(seg).astype(int)
        class_ids = []
        mask = np.zeros([seg.shape[0], seg.shape[1], len(class_list)])
        for i in class_list:
            each_class = np.ones([seg.shape[0], seg.shape[1]])*i
            mask[:, :, i] = (seg == each_class).astype(int)
            class_ids.append(i)

        # TODO: What is this??? Never change for brain dataset. Could be bug in the future.
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(len(class_ids)-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Transform to array
        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)

    # def load_mask(self, image_id):
    #     """Generate instance masks for shapes of the given image ID.
    #     Returns:
    #     masks: A bool array of shape [height, width, instance count] with
    #         one mask per instance.
    #     class_ids: a 1D array of class IDs of the instance masks.
    #     """
    #     info = self.image_info[image_id]
    #     seg = np.load(info['path'])[1]
    #     mask = np.zeros([seg.shape[0], seg.shape[1], len(self.class_info)])
    #     class_ids = []
    #     for i in range(len(self.class_info)):
    #         each_class = np.ones([seg.shape[0], seg.shape[1]]) * i
    #         if np.sum(seg == each_class) != 0:
    #             mask[:, :, i:i+1] = (seg == each_class).astype(int)
    #             class_ids.append(i)

    #     # TODO: What is this??? Never change for brain dataset. Could be bug in the future.
    #     # Handle occlusions
    #     occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    #     count = len(class_ids)
    #     for i in range(count-2, -1, -1):
    #         mask[:, :, i] = mask[:, :, i] * occlusion
    #         occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

    #     # Transform to array
    #     class_ids = np.array(class_ids)
    #     #np.save('/Users/dodo/Downloads/mask.npy', mask)
    #     #np.save('/Users/dodo/Downloads/class_ids.npy', class_ids)
    #     return mask.astype(np.bool), class_ids.astype(np.int32)


def main():
    from preprocess import dataset_preprocess
    dataset_preprocess()

    from brain import BrainsConfig, BrainsDataset
    config = BrainsConfig()
    config.display()

    # Dataset path
    train_data_path = './../dataset/processed/train'
    val_data_path = './../dataset/processed/val'

    # Training dataset
    dataset_train = BrainsDataset()
    dataset_train.load_brains(train_data_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BrainsDataset()
    dataset_val.load_brains(val_data_path)
    dataset_val.prepare()

if __name__ == '__main__':
    main()
