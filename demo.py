import argparse
import tools.find_mxnet
import mxnet as mx
import numpy as np
import os
import sys
import glob

sys.path.append('/usr/local/lib/python2.7/site-packages')
from detect.detector import Detector
from symbol.symbol_factory import get_symbol

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='resnet50',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'),
                        type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=512,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--mode', dest='mode', type=int, default=-1,
                       help='running mode of this file')
    args = parser.parse_args()
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # customized
    #args.network = 'resnet101'
    args.network = 'resnetsub101_two_shared'
    #imgpath = './data/kitti/data_object_image_2/training/image_2/'
    # imgpath = './data/caltech-pedestrian-dataset-converter/data/images/'

    if args.mode == 2:
        mode = 1
    else:
        # 0 -   show detections (demo)
        # 1 -   record detections in to_file
        # 2 -   record anchors (don't visualize detection result,
        #       default anchor box position, no shifts or transformations)
        # 3 -   same as 1, but on Caltech Pedestrian
        mode = 3

    if mode == 0:
        #imgnames = ['006667', '001671', '005589', '001264', '003507', '004370']
        imgnames = ['set00_V001_1783', 'set07_V002_1809','set07_V003_1341']
        #set00_V001_1783
        #set07_V002_1809
        #set07_V003_1341
    elif mode == 1:
        val_path = './data/kitti/data_object_image_2/training/val.txt'
        with open(val_path) as f:
            imgnames = [idx.rstrip() for idx in f.readlines()]

    # Caltech
    elif mode == 3:
        caltechPath = './data/caltech-pedestrian-dataset-converter/data/test-images/'
        sets = ['set' + setNum for setNum in ['06', '07', '08', '09', '10']]            # set06 - set10
        videos = ['V' + videoNum for videoNum in [str(x).zfill(3) for x in range(20)]]  # V000  - V019

        #s = sets[0]
        #v = videos[0]
        #imgpath = os.path.join(caltechPath, s, v)
        for s in sets:
            for v in videos:
                val_path = caltechPath + s + v
                imgpath = os.path.join(caltechPath, s, v)
                if not os.path.exists(imgpath):
                    continue
                print("{} {}".format(s, v))

                image_list = []
                for f in os.listdir(imgpath):
                    if f.endswith(".png"):
                        image_list.append(f)

                # Sorting
                import re

                def atoi(text):
                    return int(text) if text.isdigit() else text

                def natural_keys(text):
                    return [atoi(c) for c in re.split('(\d+)', text)]

                image_list.sort(key=natural_keys)

                args.images = ','.join([os.path.join(imgpath, fn) for fn in image_list])

                toFilePath = './data/caltechUSA-two-stream/results/{}'.format(s, v)
                if not os.path.exists(toFilePath):
                    os.makedirs(toFilePath)
                to_file = os.path.join(toFilePath, '{}.txt'.format(v))    # skip layer defined in multibox_detection.cu

                #if mode == 0 or mode == 1:
                #    ext = '.png'
                #    args.images = ', '.join([imgpath + s + ext for s in imgnames])

                args.dir = None
                args.ext = None
                args.epoch = 10
                args.prefix = os.path.join(os.getcwd(), 'model', 'resnet101', 'resnet-101-two-stream-caltech')
                args.data_shape = [480, 640]
                args.mean_r = 123
                args.mean_g = 117
                args.mean_b = 104
                args.thresh = 0.10
                args.nms = 0.45
                #args.force_nms = True
                args.force_nms = False
                args.show_timer = True
                args.deploy_net = False
                args.class_names = 'Person'  # 'Car, Van, Truck, Pedestrian, Persion_sitting, Cyclist, Tram, Misc, DontCare'
                if mode == 0 or mode == 1 or mode == 3:
                    args.cpu = False
                    args.gpu_id = 0
                if mode == 2:
                    args.cpu = True
                    ctx = mx.cpu()

                #print("Generating detection results for {} images".format(len(imgnames)))

                # parse image list
                image_list = [i.strip() for i in args.images.split(',')]
                assert len(image_list) > 0, "No valid image specified to detect"

                network = None if args.deploy_net else args.network
                class_names = parse_class_names(args.class_names)
                if args.prefix.endswith('_'):
                    prefix = args.prefix + args.network + '_' + str(args.data_shape)
                else:
                    prefix = args.prefix

                if "detector" not in locals():
                    detector = get_detector(network, prefix, args.epoch,
                                            args.data_shape,
                                            (args.mean_r, args.mean_g, args.mean_b),
                                            ctx, len(class_names), args.nms_thresh, args.force_nms)
                # run detection demo
                if mode == 0:
                    detector.detect_and_visualize(image_list, args.dir, args.extension,
                                                  class_names, args.thresh, args.show_timer)
                # keep records of valid detection
                elif mode == 1:
                    detector.detect_and_record(image_list, to_file, args.dir, args.extension,
                                               class_names, args.thresh, args.show_timer)
                # keep records of scores and position for each anchor box (in cpu mode),
                # multibox_detecion.cc should be compiled with DEBUG defined in config.mk
                elif mode == 2:
                    detector.detect_and_record_anchors(image_list, args.dir, args.extension,
                                                       class_names, args.thresh, args.show_timer)

                # for Caltech
                elif mode == 3:
                    detector.detect_and_record(image_list, to_file, args.dir, args.extension,
                                               class_names, args.thresh, args.show_timer)
