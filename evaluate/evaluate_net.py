from __future__ import print_function
import os
import sys
import importlib
import tools.find_mxnet
import mxnet as mx
from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric
import logging
from symbol.symbol_factory import get_symbol

def evaluate_net(net, path_imgrec, num_classes, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None,
                 voc07_metric=False,
                 use_second_network=False,
                 net1=None, path_imgrec1=None, epoch1=None, model_prefix1=None, data_shape1=None):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    elif isinstance(data_shape, list):
        data_shape = (3, data_shape[0], data_shape[1])
    assert len(data_shape) == 3 and data_shape[0] == 3
    # model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape,
                              path_imglist=path_imglist, **cfg.valid)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if net is None:
        net = load_net
    else:
        net = get_symbol(net, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names)

    # run evaluation
    if not use_second_network:
        results = mod.score(eval_iter, metric, num_batch=None)
        for k, v in results:
            print("{}: {}".format(k, v))
    else:
        logging.basicConfig()
        logger1 = logging.getLogger()
        logger1.setLevel(logging.INFO)

        # load sub network
        if isinstance(data_shape1, int):
            data_shape1 = (3, data_shape1, data_shape1)
        elif isinstance(data_shape1, list):
            data_shape1 = (3, data_shape1[0], data_shape1[1])
        assert len(data_shape1) == 3 and data_shape1[0] == 3

        # iterator
        eval_iter1 = DetRecordIter(path_imgrec1, batch_size, data_shape1,
                                   path_imglist=path_imglist, **cfg.valid)
        # model params
        load_net1, args1, auxs1 = mx.model.load_checkpoint(model_prefix1, epoch1)
        # network
        if net1 is None:
            net1 = load_net1
        else:
            net1 = net
        if 'label' not in net1.list_arguments():
            label1 = mx.sym.Variable(name='label')
            net1 = mx.sym.Group([net1, label1])

        # init module
        mod1 = mx.mod.Module(net1, label_names=('label',), logger=logger1, context=ctx,
                            fixed_param_names=net1.list_arguments())
        mod1.bind(data_shapes=eval_iter1.provide_data, label_shapes=eval_iter1.provide_label)
        mod1.set_params(args1, auxs1, allow_missing=False, force_init=True)

        if voc07_metric:
            metric1 = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
        else:
            metric1 = MApMetric(ovp_thresh, use_difficult, class_names)

        filepath = '/home/binghao//workspace/MXNet-SSD/matlab/kitti/outputs/ssd/'
        filepath1 = '/home/binghao//workspace/MXNet-SSD/matlab/kitti/outputs/ssd_central/'
        mod.score_m(filepath, eval_iter, metric, num_batch=None)
        mod1.score_m(filepath1, eval_iter1, metric1, num_batch=None)












