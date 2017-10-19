from __future__ import print_function, absolute_import
import os.path as osp
import find_mxnet
import mxnet as mx
import argparse
from symbol import symbol_factory
import json



parser = argparse.ArgumentParser(description='network visualization')
parser.add_argument('--network', type=str, default='vgg16_reduced',
                    help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=300,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

args.network = 'mobilenet'
args.num_classes = 20
args.data_shape = 224
args.train = True


if not args.train:
    net = symbol_factory.get_symbol(args.network, args.data_shape, num_classes=args.num_classes)
    a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape)}, \
        node_attrs={"shape":'rect', "fixedsize":'false'})
    filename = "ssd_" + args.network + '_' + str(args.data_shape)
    a.render(osp.join(osp.dirname(__file__), filename))
else:
    net = symbol_factory.get_symbol_train(args.network, args.data_shape, num_classes=args.num_classes)
    # print(net.tojson())
    with open("{}_network.txt".format(args.network), "w") as outfile:
        parsed = json.loads(net.tojson())
        json.dump(parsed, outfile, indent=4, sort_keys=True)
