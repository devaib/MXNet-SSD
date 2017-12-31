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

# two-stream SSD
pretrained_sub = os.path.join(os.getcwd(),'..', 'model', 'resnet101', 'resnet-101-sub')
epoch_sub = 1
sym_sub, arg_params_sub, aux_params_sub = mx.model.load_checkpoint(pretrained_sub, epoch_sub)

# copy params to sub-network
new_arg_params = {}
for k, v in arg_params.iteritems():
    new_k = 'sub_' + k
    new_arg_params[k] = v
    new_arg_params[new_k] = v

# save a model to mymodel-symbol.json and mymodel-0100.params
# prefix = 'mymodel'
# iteration = 100
# model.save(prefix, iteration)

print 0
