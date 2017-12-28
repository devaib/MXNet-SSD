import argparse
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.find_mxnet
import mxnet as mx
import numpy as np
from evaluate.eval_metric import MApMetric, VOC07MApMetric


filepath = '/home/binghao/workspace/MXNet-SSD/matlab/kitti/outputs/ssd/'
filepath1 = '/home/binghao/workspace/MXNet-SSD/matlab/kitti/outputs/ssd_central/'
filepath2 = '/home/binghao/workspace/MXNet-SSD/matlab/kitti/outputs/ssd_small/'
val_file = '../data/kitti/rec/val.lst'
val_central_file = '../data/kitti/rec/val_central.lst'
csv_file = './mAP.csv'
num_anchors = 19692
overlap_thresh = 0.5
use_difficult = False
class_names = ['Car']
metric = VOC07MApMetric(overlap_thresh, use_difficult, class_names)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on two networks')
    parser.add_argument('--margin', dest='margin', help='margin from central boundaries',
                        default=0.14, type=float)
    parser.add_argument('--overlap_thresh_central', dest='over_thresh',
                        help='overlap threshold of predtions from two netwoks', default=0.5, type=float)
    args = parser.parse_args()
    return args


def iou(x, ys):
    """
    Calculate intersection-over-union overlap
    Params:
    ----------
    x : numpy.array
        single box [xmin, ymin ,xmax, ymax]
    ys : numpy.array
        multiple box [[xmin, ymin, xmax, ymax], [...], ]
    Returns:
    -----------
    numpy.array
        [iou1, iou2, ...], size == ys.shape[0]
    """
    ixmin = np.maximum(ys[:, 0], x[0])
    iymin = np.maximum(ys[:, 1], x[1])
    ixmax = np.minimum(ys[:, 2], x[2])
    iymax = np.minimum(ys[:, 3], x[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
          (ys[:, 3] - ys[:, 1]) - inters
    ious = inters / uni
    ious[uni < 1e-12] = 0  # in case bad boxes
    return ious


if __name__ == '__main__':
    args = parse_args()

    # # get image list from val.lst and val_central.lst
    # vals = []
    # vals_central = []
    # with open(os.path.join(os.getcwd(), val_file)) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         vals.append(line[-11:-5])
    # with open(os.path.join(os.getcwd(), val_central_file)) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         vals_central.append(line[-11:-5])
    #
    # assert len(vals) >= len(vals_central), "images from vals less than vals_central"
    #
    # final_pred = -1 * mx.nd.ones((1, num_anchors, 6), mx.gpu())
    # for i in range(0, len(vals)):
    #     imgname = vals[i]
    #     pred_file = filepath+'preds'+str(i)+'.ndarray'
    #     pred = mx.nd.load(pred_file)
    #     final_pred[0][0:num_anchors] = pred[0][0]
    #
    #     # index in vals_central.lst
    #     if imgname in vals_central:
    #         j = vals_central.index(imgname)
    #         pred_central_file = filepath1+'preds'+str(j)+'.ndarray'
    #         pred_central = mx.nd.load(pred_central_file)
    #
    #         # transform bbs to [x_center, y_center, ww, hh] and project them back to original image
    #         clsid = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=0, end=1)
    #         score = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=1, end=2)
    #         xmins = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=2, end=3)
    #         ymins = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=3, end=4)
    #         xmaxs = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=4, end=5)
    #         ymaxs = mx.nd.slice_axis(pred_central[0][0], axis=1, begin=5, end=6)
    #         a = mx.nd.ones((num_anchors, 1))
    #         x_centers = (xmins + xmaxs) / 2 / 2 + 0.25 * mx.nd.ones((num_anchors, 1), mx.gpu())
    #         y_centers = (ymins + ymaxs) / 2 / 2 + 0.25 * mx.nd.ones((num_anchors, 1), mx.gpu())
    #         wws = (xmaxs - xmins) / 2 / 2
    #         hhs = (ymaxs - ymins) / 2 / 2
    #         xmins_o = x_centers - wws
    #         ymins_o = y_centers - hhs
    #         xmaxs_o = x_centers + wws
    #         ymaxs_o = y_centers + hhs
    #         pred_central_o = mx.nd.concat(clsid, score, xmins_o, ymins_o, xmaxs_o, ymaxs_o, dim=1)
    #
    #         # merge detecions from pred and pred_central_o
    #         cid = -1
    #         p = final_pred[0].asnumpy()
    #         indices = np.where(p[:, 0].astype(int) != cid)[0]
    #         dets = p[indices]
    #
    #         p = pred_central_o.asnumpy()
    #         indices = np.where(p[:, 0].astype(int) != cid)[0]
    #         dets_central = p[indices]
    #         # constrains
    #         #dets_central = dets_central[dets_central[:, 1] > 0.5]
    #         dets_central = dets_central[dets_central[:, 2] > (0.25 + args.margin)]
    #         dets_central = dets_central[dets_central[:, 3] > (0.25 + args.margin)]
    #         dets_central = dets_central[dets_central[:, 4] < (0.75 - args.margin)]
    #         dets_central = dets_central[dets_central[:, 5] < (0.75 - args.margin)]
    #         # dets_central = dets_central[(dets_central[:, 5] - dets_central[:,3]) < 0.3]
    #
    #         # remove overlap from dets
    #         remove_indices = []
    #         for k, det in enumerate(dets):
    #             ious = iou(det[2:], dets_central[:, 2:])
    #             for v_ious in ious:
    #                 if v_ious > args.over_thresh:
    #                     remove_indices.append(k)
    #                     break
    #         dets = np.delete(dets, remove_indices, axis=0)
    #         finals = np.vstack((dets, dets_central))
    #         finals = mx.nd.array([finals])
    #         labels = [pred[1]]
    #         preds = [finals, pred[1]]
    #         metric.update(labels, preds)

    # to combine ssd anad ssd_small_object
    vals = []
    with open(os.path.join(os.getcwd(), val_file)) as f:
        lines = f.readlines()
        for line in lines:
            vals.append(line[-11:-5])
    for i in range(0, len(vals)):
        imgname = vals[i]
        pred_file = filepath+'preds'+str(i)+'.ndarray'
        pred = mx.nd.load(pred_file)
        dets = pred[0][0].asnumpy()

        pred_small_file = filepath2 + 'preds' + str(i) + '.ndarray'
        pred_small = mx.nd.load(pred_small_file)
        dets_small = pred_small[0][0].asnumpy()

        finals = np.vstack((dets, dets_small))
        finals = mx.nd.array([finals])
        labels = [pred[1]]
        preds = [finals, pred[1]]
        metric.update(labels, preds)

    results = metric.get_name_value()

    for k, v in results:
        print("{}: {}".format(k, v))

    # save results to csv
    with open(csv_file, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow([args.margin, args.over_thresh] + [v for k, v in results])



