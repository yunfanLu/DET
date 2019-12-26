# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 20:41
# @Author  : yunfan


from __future__ import absolute_import
from __future__ import division

import json
import os
import warnings
import numpy as np
import mxnet as mx
import pandas
from gluoncv.data.base import VisionDataset

class MoonLabelMeDetection(VisionDataset):

    CLASSES = ['hole',]

    def __init__(self, root, train=True, depth=False, transform=None):
        super(MoonLabelMeDetection, self).__init__(root)
        self.depth = depth
        self.train = train
        self.items = self.load_items(root)
        self.label_cache = [None for i in range(len(self.items))]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dom_path, dem_path, csv_path = self.items[idx]
        if self.label_cache[idx] is None:
            self.label_cache[idx] = self.load_label(csv_path)
        label = self.label_cache[idx]
        img = self.load_img(dom_path, dem_path)
        if self.transform is not None:
            return self.transform(img, label)
        return img, label

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def load_img(self, dem_path, dom_path):
        dom = mx.image.imread(dom_path, 1)
        if self.depth:
            dem = mx.image.imread(dem_path, 1)
            dom.combine(dem)
        return dom

    def load_label(self, csv_path):
        label_y1_x1_y2_x2 = pandas.read_csv(csv_path)
        label = []
        for index, row in label_y1_x1_y2_x2.iterrows():
            label.append([int(row.y1), int(row.x1), int(row.y2), int(row.x2), 0, 0])
        return np.array(label)

    def load_items(self, root):
        folders = os.listdir(root)
        train_load_folders = []
        test_load_folders = []
        for i, folder in enumerate(folders):
            idx = i % 10
            if idx > 3:
                train_load_folders.append(folder)
            else:
                test_load_folders.append(folder)
        if self.train:
            folders = train_load_folders
        else:
            folders = test_load_folders

        items = []
        for folder in folders:
            path = os.path.join(root, folder)
            files = os.listdir(path)
            for file in files:
                name, ftype = file.split('.')
                if (ftype == 'tif') and ('DOM' in name):
                    dom_id = name
                else:
                    continue

                dom_file = file
                dem_file = f'{dom_id.replace("DOM", "DEM")}.tif'
                csv_file = f'{dom_id}_label.csv'
                dom_path, dem_path, csv_path = [os.path.join(path, file) for file in [dom_file, dem_file, csv_file]]
                if os.path.isfile(dom_path) and os.path.isfile(dem_path) and os.path.isfile(csv_path):
                    items.append((dom_path, dem_path, csv_path))

        return items