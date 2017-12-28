from __future__ import print_function, absolute_import
import os
import numpy as np
from .imdb import Imdb
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
    def __init__(self, image_set, kitti_path, shuffle=False, suffix='', is_train=False,
                 names='kitti.names'):
        super(Kitti, self).__init__('kitti' + '_' + image_set)
        self.image_set = image_set
        self.kitti_path = kitti_path
        self.data_path = os.path.join(self.kitti_path, 'data_object_image_2', 'training')
        self.extension = '.png'
        self.is_train = is_train
        self.suffix = suffix
        self.classes = self._load_class_names(names,
                                              os.path.join(os.path.dirname(__file__), 'names'))
        # TODO: consider DontCare
        self.class_dict = {'Car': 0,
                           'Van': 0,
                           'Truck': 0}
        '''
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8}
        '''
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
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index : int
            index of a specific image

        Returns:
        ----------
        full path of this image
        """
        assert self.image_set is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        # image_file = os.path.join(self.data_path, 'image_2'+self.suffix, name + self.extension)
        image_file = os.path.join(self.data_path, 'image_2', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index : int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index : str
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.kitti_path, 'data_object_label_2', 'training', 'label_2' + self.suffix, index + '.txt')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _image_size_from_index(self, index):
        """
        given image index, return its size

        Parameters:
        ----------
        index : str
            index of a specific image

        Returns:
        ---------
        image size in format [width, height]
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        image_size_file = os.path.join(self.data_path, 'image_size', index + '.txt')
        with open(image_size_file) as f:
            line = f.readlines()[0]
            image_size = line.split()
        return image_size

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_object x 5] tensor
        """
        temp = []

        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            size = self._image_size_from_index(idx)
            width = float(size[0])
            height = float(size[1])
            label = []

            with open(label_file) as f:
                lines = f.readlines()

                for line in lines:
                    words = line.split()
                    class_name = words[0]
                    # ignore other classes
                    if class_name not in self.class_dict.keys():
                        continue
                    cls_id = self.class_dict[class_name]
                    xmin = float(words[4]) / width
                    ymin = float(words[5]) / height
                    xmax = float(words[6]) / width
                    ymax = float(words[7]) / height
                    label.append([cls_id, xmin, ymin, xmax, ymax])
                temp.append(np.array(label))

        return temp














