from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
from collections import defaultdict
from six.moves import xrange
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all, im_detect_raw_masks
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

sys.path.append(os.path.realpath('./Detectron.pytorch'))
sys.path.append(os.path.realpath('./utils'))
from stream import VideoInputStream

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def process_detections(cls_boxes, cls_segms, height, width):
    boxes = []
    scores = []
    masks = []

    box_list = []

    for b in cls_boxes:
        if len(b) > 0:
            for bidx in range(len(b)):
                box = b[bidx, :4]
                box = [ float(box[1])/height, float(box[0])/width,
                        float(box[3])/height, float(box[2])/width ]
                box_list.append(box)

    score_list = [ b[:, 4] for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.array(box_list)
        scores = np.concatenate(score_list)
    else:
        boxes = np.array([])
        scores = np.array([])

    class_ids = []
    for j in range(len(cls_boxes)):
        class_ids += [j] * len(cls_boxes[j])

    for j in range(len(class_ids)):
        segms = cls_segms[j]
        cls = class_ids[j]
        masks.append(segms[cls, :, :].copy())

    return boxes, class_ids, scores, masks

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--input_video_path',
        help='path to the input video stream')
    parser.add_argument(
        '--output_path',
        help='path to save the detection results',
        default="infer_outputs")
    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.input_video_path
    assert args.output_path

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    s = VideoInputStream(args.input_video_path)
    frame_id = 0

    frame_detections ={}

    for im in s:
        print('frame_id', frame_id)
        assert im is not None

        timers = defaultdict(Timer)
        cls_boxes, cls_segms, _ = im_detect_raw_masks(maskRCNN, im, timers=timers)

        if cls_segms is not None:
            print(im.shape, len(cls_boxes), len(cls_segms))
        boxes, class_ids, scores, masks = \
                process_detections(cls_boxes, cls_segms, s.height, s.width)

        frame_detections[frame_id] = [boxes, class_ids, scores, masks]
        frame_id = frame_id + 1

    np.save(args.output_path, frame_detections)

if __name__ == '__main__':
    main()
