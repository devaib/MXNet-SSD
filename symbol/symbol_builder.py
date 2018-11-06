from __future__ import absolute_import
import mxnet as mx
from .common import multi_layer_feature, multibox_layer, multibox_layer2, multi_layer_feature_concat


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)

def get_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out

def get_symbol_train_concat(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    # first stream
    label = mx.sym.Variable('label')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature_concat(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter, f_layers=['_plus12', '_plus15', '', ''], stream='origin')

    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # second stream
    label2 = mx.sym.Variable('label2')
    #body = import_module(network).get_symbol(num_classes, **kwargs)
    layers2 = multi_layer_feature_concat(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter, f_layers=['_plus28', '_plus31', '', ''], stream='scaled')

    loc_preds2, cls_preds2, anchor_boxes2 = multibox_layer2(layers2, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp2 = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes2, label2, cls_preds2], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target2")
    loc_target2 = tmp2[0]
    loc_target_mask2 = tmp2[1]
    cls_target2 = tmp2[2]

    cls_prob2 = mx.symbol.SoftmaxOutput(data=cls_preds2, label=cls_target2, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob2")
    loc_loss_2 = mx.symbol.smooth_l1(name="loc_loss_2", \
        data=loc_target_mask2 * (loc_preds2 - loc_target2), scalar=1.0)
    loc_loss2 = mx.symbol.MakeLoss(loc_loss_2, grad_scale=1., \
        normalization='valid', name="loc_loss2")

    # monitoring training status
    cls_label2 = mx.symbol.MakeLoss(data=cls_target2, grad_scale=0, name="cls_label2")
    det2 = mx.contrib.symbol.MultiBoxDetection(*[cls_prob2, loc_preds2, anchor_boxes2], \
        name="detection2", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det2 = mx.symbol.MakeLoss(data=det2, grad_scale=0, name="det_out2")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det, cls_prob2, loc_loss2, cls_label2, det2])
    return out


def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out


def get_symbol_concat(network, num_classes, from_layers, num_filters, sizes, ratios,
                      strides, pads, normalizations=-1, steps=[], min_filter=128,
                      nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    body = import_module(network).get_symbol(num_classes, **kwargs)

    # first stream
    layers = multi_layer_feature_concat(body, from_layers, num_filters, strides, pads,
                                        min_filter=min_filter, f_layers=['_plus12', '_plus15', '', ''], stream='origin')
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out1 = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    # second stream
    layers2 = multi_layer_feature_concat(body, from_layers, num_filters, strides, pads,
                                         min_filter=min_filter, f_layers=['_plus28', '_plus31', '', ''], stream='scaled')
    loc_preds2, cls_preds2, anchor_boxes2 = multibox_layer2(layers2, \
                                                        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
                                                        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    cls_prob2 = mx.symbol.SoftmaxActivation(data=cls_preds2, mode='channel', \
        name='cls_prob2')
    out2 = mx.contrib.symbol.MultiBoxDetection(*[cls_prob2, loc_preds2, anchor_boxes2], \
        name="detection2", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    out = mx.symbol.Group([out1, out2])
    return out



# customized for debug
def get_symbol_m(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    body = import_module(network).get_symbol(num_classes, **kwargs)
    # layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
    #     min_filter=min_filter)
    #
    # loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
    #     num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
    #     num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    #
    # cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
    #     name='cls_prob')
    # out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return body
