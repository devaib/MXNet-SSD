"""Presets for various network configurations"""
from __future__ import absolute_import
import logging
from . import symbol_builder

def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
                [.75, .8216], [.9, .9721]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in
                [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'inceptionv3':
        from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_two_stream_w_four_layers':
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnetsub'
        from_layers = ['_plus28', '_plus12', '_plus15', '', '']
        num_filters = [-1, -1, -1, 512, 256]
        strides = [-1, -1, -1, 2, 2]
        pads = [-1, -1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .141], [.2,.272], [.37, .447], [.54, .619]]
        ratios = [[1,2,.5], [1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_four_layers':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_customized':
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_customized_last_three_layers':
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus15', '', '']
        num_filters = [-1, 512, 256]
        strides = [-1, 2, 2]
        pads = [-1, 1, 1]
        sizes = [[.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_customized_first_layer':
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12']
        num_filters = [-1]
        strides = [-1]
        pads = [-1]
        sizes = [[.03, .0548]]
        ratios = [[1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50_two_stream':
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnetsub'
        from_layers = ['_plus28', '_plus15', '', '']  # 31-(15-12)=28
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_two_stream':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnetsub'
        from_layers = ['_plus62', '_plus32', '', '']  # 45 - 3 - 4
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnetsub101_two_shared':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnetsub_two_shared'
        from_layers = ['_plus38', '_plus15', '', '']  # 45 - 3 - 4
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnetsub101_one_shared':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnetsub_one_shared'
        from_layers = ['_plus42', '_plus15', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnetsub101_test':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnetsub'
        from_layers = ['_plus45', '_plus15', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_test_last_three_layer':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus32', '', '']
        num_filters = [-1, 512, 256]
        strides = [-1, 2, 2]
        pads = [-1, 1, 1]
        sizes = [[.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_test':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.03, .0548], [.1, .1732], [.3, .3873], [.5, .5916]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_test_one_layer':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29']
        num_filters = [-1]
        strides = [-1]
        pads = [-1]
        sizes = [[.03, .0548]]
        ratios = [[1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '', '', '']
        #from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        '''
            min_ratio = 20, max_ratio = 90,
            step = (max_ratio - min_ratio) / (#layer - 2) = (90 - 20) / (6 - 2) = 17,
            ratio = range(min_ratio, max_ratio, step) = [20, 37, 54, 71, 88],
            min_sizes = ratio / 100 = [.2, .37, .54, .71, .88],
            min_sizes = [.1] + min_size = [.1, .2, .37, .54, .71, .88],
            max_sizes = sqrt( min_size * (min_size+step/100) ) = [.141, .272, .447, .619, .79, .961]
        '''
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_w_feature_layer1':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12']
        num_filters = [-1]
        strides = [-1]
        pads = [-1]
        sizes = [[.1, .141]]
        ratios = [[1,2,.5]]
        nomalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_w_feature_layer2':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15']
        num_filters = [-1, -1]
        strides = [-1, -1]
        pads = [-1, -1]
        sizes = [[.1, .141], [.2, .272]]
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1./3]]
        nomalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_w_feature_layer3':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '']
        num_filters = [-1, -1, 512]
        strides = [-1, -1, 2]
        pads = [-1, -1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_w_feature_layer4':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '']
        num_filters = [-1, -1, 512, 256]
        strides = [-1, -1, 2, 2]
        pads = [-1, -1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101_w_feature_layer5':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '']
        num_filters = [-1, -1, 512, 256, 256]
        strides = [-1, -1, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobilenet':
        from_layers = ['activation22', 'activation26', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_symbol_train(network, data_shape, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol_train(**config)

def get_symbol(network, data_shape, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol(**config)

# for debug
def get_symbol_m(network, data_shape, **kwargs):
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol_m(**config)
