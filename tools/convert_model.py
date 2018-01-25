"""
Copy ResNet101 model to both sub-network streams
"""

import os
import find_mxnet
import mxnet as mx

# pretrained ResNet
pretrained = os.path.join(os.getcwd(),'..', 'model', 'resnet101', 'resnet-101')
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained, epoch)

# pretrained SSD_customized
pretrained_customized = os.path.join(os.getcwd(), '..', 'model', 'resnet101', 'resnet-101_customized')
epoch_customized = 120
sym_customized, arg_params_customized, aux_params_customized = mx.model.load_checkpoint(pretrained_customized,
                                                                                        epoch_customized)

# pretrained SSD_small
pretrained_small = os.path.join(os.getcwd(), '..', 'model', 'resnet101', 'resnet-101_small')
epoch_small = 120
sym_small, arg_params_small, aux_params_small = mx.model.load_checkpoint(pretrained_small, epoch_small)

# two-stream SSD
pretrained_sub = os.path.join(os.getcwd(),'..', 'model', 'resnet101', 'resnet-101-sub')
epoch_sub = 1
sym_sub, arg_params_sub, aux_params_sub = mx.model.load_checkpoint(pretrained_sub, epoch_sub)

# copy params to sub-network
new_arg_params = {}
arg_params_customized.pop('_plus12_cls_pred_conv_bias')
arg_params_customized.pop('_plus12_cls_pred_conv_weight')
arg_params_customized.pop('_plus12_loc_pred_conv_bias')
arg_params_customized.pop('_plus12_loc_pred_conv_weight')
arg_params_small.pop('bn_data_beta')
arg_params_small.pop('bn_data_gamma')
new_arg_params['_plus45_cls_pred_conv_bias'] = arg_params_small['_plus12_cls_pred_conv_bias']
new_arg_params['_plus45_cls_pred_conv_weight'] = arg_params_small['_plus12_cls_pred_conv_weight']
new_arg_params['_plus45_loc_pred_conv_bias'] = arg_params_small['_plus12_loc_pred_conv_bias']
new_arg_params['_plus45_loc_pred_conv_weight'] = arg_params_small['_plus12_loc_pred_conv_weight']
arg_params_small.pop('_plus12_cls_pred_conv_bias')
arg_params_small.pop('_plus12_cls_pred_conv_weight')
arg_params_small.pop('_plus12_loc_pred_conv_bias')
arg_params_small.pop('_plus12_loc_pred_conv_weight')
for k, v in arg_params_customized.iteritems():
    new_arg_params[k] = v
for k, v in arg_params_small.iteritems():
    new_k = 'sub_' + k
    new_arg_params[new_k] = v


print 0
