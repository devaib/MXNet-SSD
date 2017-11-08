from __future__ import print_function, absolute_import
import os
import numpy as np
from .imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2

class Kitti(Imdb):
    """
    Implementation of Imdb for KITTI dataset

    Parameters:
    image_set : str
        set to be used, can be train, val
    kitti_path : str
        path of KITTI dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, kitti_path, shuffle=False, is_train=False,
                 names='kitti.names'):
        super(Kitti, self).__init__('kitti' + '_' + image_set)
        self.image_set = image_set
        self.kitti = kitti_path
        self.data_path = os.path.join(kitti_path, 'data_object_image_2', 'training', 'image_2')
        self.extension = '.png'
        self.is_train = is_train
        self.classes = self._load_class_names(names,
                                              os.path.join(os.path.dirname(__file__), 'names'))
        # self.config = {'use_difficult': True}
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()

    @property
    def cache_path(self):
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, )

    def _load_image_labels(self):
        pass










