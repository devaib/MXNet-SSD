import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mxnet as mx


def residual_unit(arg_params, data, stride, dim_match, name):

    gamma = arg_params['{}_bn1_gamma'.format(name)]
    beta = arg_params['{}_bn1_beta'.format(name)]
    moving_mean = aux_params['{}_bn1_moving_mean'.format(name)]
    moving_var = aux_params['{}_bn1_moving_var'.format(name)]
    bn1 = mx.nd.BatchNorm(data=data, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var)
    act1 = mx.nd.Activation(data=bn1, act_type='relu')
    weight = arg_params['{}_conv1_weight'.format(name)]
    conv1 = mx.nd.Convolution(data=act1, weight=weight, no_bias=True, kernel=weight.shape[2:],
                              stride=(1,1), num_filter=weight.shape[0])
    gamma = arg_params['{}_bn2_gamma'.format(name)]
    beta = arg_params['{}_bn2_beta'.format(name)]
    moving_mean = aux_params['{}_bn2_moving_mean'.format(name)]
    moving_var = aux_params['{}_bn2_moving_var'.format(name)]
    bn2 = mx.nd.BatchNorm(data=conv1, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var)
    act2 = mx.nd.Activation(data=bn2, act_type='relu')
    weight = arg_params['{}_conv2_weight'.format(name)]
    conv2 = mx.nd.Convolution(data=act2, weight=weight, no_bias=True, kernel=weight.shape[2:], pad=(1, 1),
                              stride=stride, num_filter=weight.shape[0])
    gamma = arg_params['{}_bn3_gamma'.format(name)]
    beta = arg_params['{}_bn3_beta'.format(name)]
    moving_mean = aux_params['{}_bn3_moving_mean'.format(name)]
    moving_var = aux_params['{}_bn3_moving_var'.format(name)]
    bn3 = mx.nd.BatchNorm(data=conv2, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var)
    act3 = mx.nd.Activation(data=bn3, act_type='relu')
    weight = arg_params['{}_conv3_weight'.format(name)]
    conv3 = mx.nd.Convolution(data=act3, weight=weight, no_bias=True, kernel=weight.shape[2:],
                              stride=(1,1), num_filter=weight.shape[0])

    if dim_match:
        shortcut = data
    else:
        weight = arg_params['{}_sc_weight'.format(name)]
        shortcut = mx.nd.Convolution(data=act1, weight=weight, kernel=weight.shape[2:], num_filter=weight.shape[0],
                                     stride=stride, no_bias=True)

    return conv3 + shortcut


image_path = '/home/binghao/workspace/MXNet-SSD/data/caltech-pedestrian-dataset-converter/data/test-images/'
#image_name = image_path + 'set10/V010/1500.png'
image_name = image_path + 'set10/V010/720.png'
image = cv2.imread(image_name)
image = np.transpose(image, (2, 0, 1))
image = mx.nd.array(image)
input = image.reshape((1,) + image.shape).astype('float32')

model_path = os.path.join(os.getcwd(), '.', 'model', 'resnet50', 'resnet-50-Caltech_all-two_stream_w_four_layers', 'resnet-50')
epoch = 6
sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)

mean_rgb = mx.nd.array([123, 177, 104])
mean_rgb = mean_rgb.reshape((1, 3, 1, 1))

gamma = arg_params['bn_data_gamma']
beta = arg_params['bn_data_beta']
moving_mean = aux_params['bn_data_moving_mean']
moving_var = aux_params['bn_data_moving_var']
bn = mx.nd.BatchNorm(data=input-mean_rgb, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var)
weight = arg_params['conv0_weight']
bias = mx.nd.zeros(weight.shape[0],)
conv0 = mx.nd.Convolution(data=bn, weight=weight, no_bias=True, kernel=weight.shape[2:], num_filter=weight.shape[0])
gamma = arg_params['bn0_gamma']
beta = arg_params['bn0_beta']
moving_mean = aux_params['bn0_moving_mean']
moving_var = aux_params['bn0_moving_var']
bn0 = mx.nd.BatchNorm(data=conv0, gamma=gamma, beta=beta, eps=2e-5, moving_mean=moving_mean, moving_var=moving_var)
relu = mx.nd.Activation(data=bn0, act_type='relu', name='relu0')
maxpool = mx.nd.Pooling(data=relu, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

# stage1
body = residual_unit(arg_params, maxpool, stride=(1, 1), dim_match=False, name='stage{}_unit{}'.format(1, 1))
body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(1, 2))
body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(1, 3))

# stage2
body = residual_unit(arg_params, body, stride=(2, 2), dim_match=False, name='stage{}_unit{}'.format(2, 1))
body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(2, 2))
body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(2, 3))
body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(2, 4))

# stage3
#body = residual_unit(arg_params, body, stride=(2, 2), dim_match=False, name='stage{}_unit{}'.format(3, 1))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(3, 2))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(3, 3))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(3, 4))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(3, 5))

# stage4
#body = residual_unit(arg_params, body, stride=(2, 2), dim_match=False, name='stage{}_unit{}'.format(4, 1))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(4, 2))
#body = residual_unit(arg_params, body, stride=(1, 1), dim_match=True, name='stage{}_unit{}'.format(4, 3))


save_path = '/home/binghao/pic4/small'

for i in range(body.shape[2]):
    out_img = body[0][i].asnumpy()
    #plt.imshow(out_img, cmap='gray')
    #plt.show()
    mpimg.imsave(os.path.join(save_path, 'ch{}.png'.format(i)), out_img)

