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

# pretrained SSD_large
pretrained_large = os.path.join(os.getcwd(), '..', 'model', 'resnet101', 'resnet-101_large')
epoch_large = 120
sym_large, arg_params_large, aux_params_large = mx.model.load_checkpoint(pretrained_large, epoch_large)

print len(arg_params_customized)
print len(arg_params_large)

large_set = []
customized_set = []
for k, _ in arg_params_large.iteritems():
    large_set.append(k)
for k, _ in arg_params_customized.iteritems():
    customized_set.append(k)
intersect = list(set(large_set) & set(customized_set))
unique_large = list(set(large_set) - set(customized_set))
unique_customized = list(set(customized_set) - set(large_set))

# two-stream SSD
pretrained_sub = os.path.join(os.getcwd(),'..', 'model', 'resnet101', 'resnet-101-sub')
epoch_sub = 17
sym_sub, arg_params_sub, aux_params_sub = mx.model.load_checkpoint(pretrained_sub, epoch_sub)

"""
# copy params to sub-network
# customized and small
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
"""
# copy params to sub-network
# large and small
new_arg_params = {}

# large multi_feat 1 -> 2, 2 -> 3
new_arg_params['multi_feat_2_conv_1x1_conv_bias'] = arg_params_large['multi_feat_1_conv_1x1_conv_bias']
new_arg_params['multi_feat_2_conv_1x1_conv_weight'] = arg_params_large['multi_feat_1_conv_1x1_conv_weight']
new_arg_params['multi_feat_2_conv_3x3_conv_bias'] = arg_params_large['multi_feat_1_conv_3x3_conv_bias']
new_arg_params['multi_feat_2_conv_3x3_conv_weight'] = arg_params_large['multi_feat_1_conv_3x3_conv_weight']
new_arg_params['multi_feat_2_conv_3x3_relu_loc_pred_conv_bias'] = arg_params_large['multi_feat_1_conv_3x3_relu_loc_pred_conv_bias']
new_arg_params['multi_feat_2_conv_3x3_relu_loc_pred_conv_weight'] = arg_params_large['multi_feat_1_conv_3x3_relu_loc_pred_conv_weight']
new_arg_params['multi_feat_2_conv_3x3_relu_cls_pred_conv_bias'] = arg_params_large['multi_feat_1_conv_3x3_relu_cls_pred_conv_bias']
new_arg_params['multi_feat_2_conv_3x3_relu_cls_pred_conv_weight'] = arg_params_large['multi_feat_1_conv_3x3_relu_cls_pred_conv_weight']

new_arg_params['multi_feat_3_conv_1x1_conv_bias'] = arg_params_large['multi_feat_2_conv_1x1_conv_bias']
new_arg_params['multi_feat_3_conv_1x1_conv_weight'] = arg_params_large['multi_feat_2_conv_1x1_conv_weight']
new_arg_params['multi_feat_3_conv_3x3_conv_bias'] = arg_params_large['multi_feat_2_conv_3x3_conv_bias']
new_arg_params['multi_feat_3_conv_3x3_conv_weight'] = arg_params_large['multi_feat_2_conv_3x3_conv_weight']
new_arg_params['multi_feat_3_conv_3x3_relu_loc_pred_conv_bias'] = arg_params_large['multi_feat_2_conv_3x3_relu_loc_pred_conv_bias']
new_arg_params['multi_feat_3_conv_3x3_relu_loc_pred_conv_weight'] = arg_params_large['multi_feat_2_conv_3x3_relu_loc_pred_conv_weight']
new_arg_params['multi_feat_3_conv_3x3_relu_cls_pred_conv_bias'] = arg_params_large['multi_feat_2_conv_3x3_relu_cls_pred_conv_bias']
new_arg_params['multi_feat_3_conv_3x3_relu_cls_pred_conv_weight'] = arg_params_large['multi_feat_2_conv_3x3_relu_cls_pred_conv_weight']

arg_params_large.pop('multi_feat_1_conv_1x1_conv_bias')
arg_params_large.pop('multi_feat_1_conv_1x1_conv_weight')
arg_params_large.pop('multi_feat_1_conv_3x3_conv_bias')
arg_params_large.pop('multi_feat_1_conv_3x3_conv_weight')
arg_params_large.pop('multi_feat_1_conv_3x3_relu_loc_pred_conv_bias')
arg_params_large.pop('multi_feat_1_conv_3x3_relu_loc_pred_conv_weight')
arg_params_large.pop('multi_feat_1_conv_3x3_relu_cls_pred_conv_bias')
arg_params_large.pop('multi_feat_1_conv_3x3_relu_cls_pred_conv_weight')

arg_params_large.pop('multi_feat_2_conv_1x1_conv_bias')
arg_params_large.pop('multi_feat_2_conv_1x1_conv_weight')
arg_params_large.pop('multi_feat_2_conv_3x3_conv_bias')
arg_params_large.pop('multi_feat_2_conv_3x3_conv_weight')
arg_params_large.pop('multi_feat_2_conv_3x3_relu_loc_pred_conv_bias')
arg_params_large.pop('multi_feat_2_conv_3x3_relu_loc_pred_conv_weight')
arg_params_large.pop('multi_feat_2_conv_3x3_relu_cls_pred_conv_bias')
arg_params_large.pop('multi_feat_2_conv_3x3_relu_cls_pred_conv_weight')

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
for k, v in arg_params_large.iteritems():
    new_arg_params[k] = v
for k, v in arg_params_small.iteritems():
    new_k = 'sub_' + k
    new_arg_params[new_k] = v

# validation
sub_set = []
for k, _ in arg_params_sub.iteritems():
    sub_set.append(k)
new_set = []
for k, _ in new_arg_params.iteritems():
    new_set.append(k)
sub_new = list(set(sub_set) - set(new_set))
new_sub = list(set(new_set) - set(sub_set))

print 0
