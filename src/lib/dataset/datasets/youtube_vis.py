from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset

class youtube_vis(GenericDataset):
    num_categories = 40
    default_resolution = [352, 640]
    class_name = ['person','giant_panda','lizard','parrot','skateboard','sedan',
                  'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
                  'train','horse','turtle','bear','motorbike','giraffe','leopard',
                  'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
                  'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
                  'tennis_racket']
    max_objs = 50
    cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    def __init__(self, opt, split):
        self.dataset_version = opt.dataset_version
        print('Using Youtube-VIS')
        data_dir = os.path.join(opt.data_dir, 'youtube_vis')

        if opt.dataset_version in ['train', 'val']:
            ann_file = '{}.json'.format(opt.dataset_version)
        img_dir = os.path.join(data_dir, '{}/JPEGImages/'.format(opt.dataset_version))

        print('ann_file', ann_file)
        ann_path = os.path.join(data_dir, 'annotations', ann_file)

        self.images = None
        # load image list and coco
        super(youtube_vis, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded Youtube-VIS {} {} {} samples'.format(
            self.dataset_version, split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

    def run_eval(self, results, save_dir):
        print('Finised')

