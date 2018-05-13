from __future__ import print_function
import os
import numpy as np
from .imdb import Imdb
# from evaluate.eval_voc import voc_eval
import cv2
import json


class CaltechPedestrian(Imdb):
    """
    Implementation of Imdb for Caltech Pedestrian dataset
    Parameters:
    ----------
    caltech_path : str
        path of Caltech Pedestrian dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, imageset, caltech_path, shuffle=False, is_train=False):
        super(CaltechPedestrian, self).__init__('caltech_pedestrain')
        self.annotation_path = "/home/binghao/workspace/MXNet-SSD/data/caltech-pedestrian-dataset-converter"
        self.annotation_folder = "data"
        self.annotation_name = "annotations.json"
        self.annotation_file = os.path.join(self.annotation_path, self.annotation_folder, self.annotation_name)

        # todo: figure out train and val sets
        self.trainset = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
        self.valset = ['set06', 'set07', 'set08', 'set09', 'set10']
        if (imageset == 'train'):
            self.imageset = self.trainset
        elif (imageset == 'val'):
            self.imageset = self.valset

        # self.caltech_path = caltech_path
        self.caltech_path = self.annotation_path
        self.data_path = os.path.join(self.caltech_path, 'data', 'images')
        self.extension = '.png'
        self.is_train = is_train

        # TODO: figure out if necessary
        self.classes = ['person', '', '', '',
                        '', '', '', '',
                        '', '', '', '',
                        '', '', '', '',
                        '', '', '', '']

        # todo: figure out other parameters
        self.config = {'padding': 30}

        self.num_classes =len(self.classes)
        if self.is_train:
            [self.image_set_index, self.labels] = self._load_image_index_labels(shuffle)
        else:
            # self.image_set_index = self._load_image_set_index(shuffle)
            # TODO: reconstruct evaluation with caltech_eval()
            [self.image_set_index, self.labels] = self._load_image_index_labels(shuffle=False)
        self.num_images = len(self.image_set_index)

    @property
    def cache_path(self):
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle):
        image_set_index = []
        with open(self.annotation_file) as f:
            annotations = json.load(f)
            for setname in annotations.keys():
                if setname not in self.imageset:
                    continue
                for videoname in annotations[setname].keys():
                    for imagename in annotations[setname][videoname]['frames'].keys():
                        # actually image name
                        image_set_index.append(setname + "_" + videoname + "_" + imagename)
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def _load_image_index_labels(self, shuffle):
        # todo: make sure of the image size
        width = 640
        height = 480
        temp = []
        image_set_index = []
        max_objects = 0
        lbl = 'person'
        h_min = 50
        v_min = 0.65
        bnds = [5, 5, 635, 475]
        with open(self.annotation_file) as f:
            annotations = json.load(f)
            for setname in annotations.keys():
                if setname not in self.imageset:
                    continue
                # if self.is_train == False and setname != 'set06':
                    # continue
                for videoname in annotations[setname].keys():
                    # if self.is_train == False and float(videoname[1:]) > 2.0:
                    # if self.is_train == False and videoname != 'V006':
                    #     continue
                    for imagename in annotations[setname][videoname]['frames'].keys():
                        # if self.is_train == False and float(imagename) > 10:
                        #     continue
                        image = annotations[setname][videoname]['frames'][imagename]
                        # actually image name
                        image_set_index.append(setname + "_" + videoname + "_" + imagename)
                        label = []
                        for detection in image:
                            # check label
                            cls_name = detection['lbl']
                            if not cls_name == 'person':
                                continue

                            # check height
                            coord = detection['pos']
                            if coord[3] < h_min:
                                continue

                            # check visiblity ratio
                            coord_v = detection['posv']
                            if (type(coord_v) is not list) or (type(coord) is not list):
                                continue
                            if not all(0 == v for v in coord_v):
                                if (coord_v[2] * coord_v[3] / coord[2] / coord[3]) < v_min:
                                    continue

                            # check bounds
                            if coord[0] < bnds[0]:
                                continue
                            if coord[1] < bnds[1]:
                                continue
                            if (coord[0] + coord[2]) > bnds[2]:
                                continue
                            if (coord[1] + coord[3]) > bnds[3]:
                                continue

                            cls_id = 0

                            # Filter objects (with width <= 20 or width > 20)
                            #if float(coord[2]) <= 20:
                            #    continue
                            # Filter occluded objects with occlusion over 65%

                            xmin = float(coord[0]) / width
                            ymin = float(coord[1]) / height
                            xmax = float(coord[0] + coord[2]) / width
                            ymax = float(coord[1] + coord[3]) / height
                            label.append([cls_id, xmin, ymin, xmax, ymax])
                        temp.append(np.array(label))
                        # max_objects = max(max_objects, len(label))
        # assert max_objects > 0, "No objects found for any of the images"
        # assert max_objects <= self.config['padding'], "# obj exceed padding"
        # self.padding = self.config['padding']
        # labels = []
        # for label in temp:
        #     # todo: # detection on certain image not correct & some image has no detections
        #     label = np.lib.pad(label, ((0, self.padding - label.shape[0]), (0, 0)), \
        #                        'constant', constant_values=(-1, -1))
        #     labels.append(label)
        # labels = np.array(labels)
        # assert len(image_set_index) == len(labels), "image_set_index and label size not equal"
        if shuffle:
            combined = list(zip(image_set_index, temp))
            np.random.shuffle(combined)
            image_set_index, temp = zip(*combined)
        return [image_set_index, temp]

    def image_path_from_index(self, index):
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        assert self.image_set_index is not None, "Dataset not initialized"
        pos = self.labels[index]
        return pos

    def evaluate_detections(self, detections):
        result_dir = os.path.join(self.caltech_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        res_file_folder = os.path.join(self.caltech_path, 'results')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        #TODO: reconstruct evaluation
        # self.write_pascal_results(detections)
        self.do_python_eval(detections)

    def get_result_file_template(self):
        res_file_folder = os.path.join(self.annotation_path, 'results')
        filename = 'det_' + 'set06' + '_' + 'video012' + '.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        # for cls_ind, cls in enumerate(self.classes):
        cls_ind = 0
        cls = self.classes[cls_ind]
        print('Writing {} caltech results file'.format(cls))
        filename = self.get_result_file_template().format(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_set_index):
                dets = all_boxes[im_ind]
                if dets.shape[0] < 1:
                    continue
                h, w = self._get_imsize(self.image_path_from_index(im_ind))
                for k in range(dets.shape[0]):
                    if (int(dets[k, 0]) == cls_ind):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, 1],
                                       int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                       int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))

    def caltech_ap(self, rec, prec, use_07_metric=False):
        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap += p / 11.
        else:
            # append sentinel values at both ends
            mrec = np.concatenate([0.], rec, [1.])
            mpre = np.concatenate([0.], prec, [0.])

            # compute precision integration ladder
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # look for recall value changes
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # sum (\delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def do_python_eval(self, all_boxes):
        cls_ind = 0
        cls = self.classes[cls_ind]
        detections = []
        for im_ind, index in enumerate(self.image_set_index):
            dets = all_boxes[im_ind]
            if dets.shape[0] < 1:
                continue
            h, w = self._get_imsize(self.image_path_from_index(im_ind))
            for k in range(dets.shape[0]):
                if (int(dets[k, 0]) == cls_ind):
                    detection = [index, dets[k, 1],
                                    int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                    int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1]
                    detections.append(detection)

        # ground truth
        class_recs = {}
        npos = 0
        classindex = 0
        #TODO: figure out if necessary to keep difficulty
        difficulty = 0
        for ind, image_name in enumerate(self.image_set_index):
            objects = [obj for obj in self.labels[ind] if obj[0] == classindex]
            bbox = np.array([obj[1:] for obj in objects])
            difficult = np.array([difficulty for x in objects]).astype(np.bool)
            det = [False] * len(objects)
            npos = npos + sum(~difficult)

            class_recs[image_name] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

        # detections
        image_ids = [detection[0] for detection in detections]
        confidence = np.array([float(detection[1]) for detection in detections])
        bbox = np.array([[float(x) for x in detection[2:]] for detection in detections])

        # sort by confidence
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

        # go down detections and mark true positives and false positives
        ovthresh = 0.5
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            r = class_recs[image_ids[d]]
            h, w = self._get_imsize(os.path.join(self.data_path, image_ids[d]) + self.extension)
            bb = bbox[d, :].astype(float)
            ovmax = -np.inf
            bbgt = r['bbox'].astype(float)
            bbgt[:,0] *= w
            bbgt[:,1] *= h
            bbgt[:,2] *= w
            bbgt[:,3] *= h

            if bbgt.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                       (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not r['difficult'][jmax]:
                    if not r['det'][jmax]:
                        tp[d] = 1.
                        r['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.caltech_ap(rec, prec, use_07_metric=True)

        print ('AP for {} = {:.4f}'.format(cls, ap))
        print ('Mean AP = {:.4f}'.format(np.mean(ap)))


    def _get_imsize(self, im_name):
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
