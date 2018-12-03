#!/usr/bin/python

import os
import sys
import math
import random
import numpy as np
import cv2


# Import Mask RCNN
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn.config import config
from mrcnn import utils


class BrainConfig(Config):
    NAME = 'brains'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    # Ground-truth labels: 0, 9, 10
    # Other segment labels: 1 to 8
    NUM_CLASSES = 1 + 8

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class BrainsDataset(utils.Dataset):

    def load_brains(self):
        self.add_class("brains", 0, "background")
        self.add_class("brains", 1, "Cortical gray matter")
        self.add_class("brains", 2, "Basal ganglia")
        self.add_class("brains", 3, "White matter")
        self.add_class("brains", 4, "White matter lesions")
        self.add_class("brains", 5, "Cerebrospinal fluid")
        self.add_class("brains", 6, "Ventricles")
        self.add_class("brains", 7, "Cerebellum")
        self.add_class("brains", 8, "Brain stem")
        self.add_class("brains", 9, "Infarction")
        self.add_class("brains", 10, "Other")

    def load_image(self, image_id):
        pass

    def load_mask(self, image_id):
        pass



