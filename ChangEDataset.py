# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 20:41
# @Author  : yunfan


from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import mxnet as mx
import pandas
from gluoncv.data.base import VisionDataset


class ChangEDET(VisionDataset):
    CLASSES = ['hole']

    def __init__(self, root, train=True, depth=False, transform=None, label_count_limit=100):
        super(ChangEDET, self).__init__(root)
        self.depth = depth
        self.train = train
        self.items = self.load_items(root)
        self.label_cache = [None for i in range(len(self.items))]
        self._transform = transform
        self._label_limit = label_count_limit
        print(f'Load {len(self.items)} images')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dom_path, dem_path, csv_path = self.items[idx]
        img = self.load_img(dom_path, dem_path)
        w, h = img.shape[0], img.shape[1]
        # if self.label_cache[idx] is None:
        #     self.label_cache[idx] = self.load_label(csv_path, w, h)
        # label = self.label_cache[idx]
        label, label_count = self.load_label(csv_path, w, h)
        if label_count == 0:
            return self.__getitem__((idx + 1) % len(self))
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        return 1

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        # assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        # assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        # assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        # assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)
        if (0 <= xmin < width) and (0 <= ymin < height) and (xmin < xmax <= width) \
            and ( ymin < ymax <= height) and ((xmax - xmin) >= 10) and ((ymax - ymin) >= 10):
            return True
        return False

    def load_img(self, dom_path, dem_path):
        dom = mx.image.imread(dom_path, 1)
        if self.depth:
            dem = mx.image.imread(dem_path, 1)
            dom.combine(dem)
        return dom

    def load_label(self, csv_path, w, h):
        label_y1_x1_y2_x2 = pandas.read_csv(csv_path)
        label = []
        for index, row in label_y1_x1_y2_x2.iterrows():
            x1, y1, x2, y2 = int(row.y1), int(row.x1), int(row.y2), int(row.x2)
            if self._validate_label(x1, y1, x2, y2, w, h) == False:
                # print("Invalid label at {},".format(csv_path))
                # print(f"\tx1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, w {w}, h {h}")
                continue
            label.append([x1, y1, x2, y2, 0, 0])
        # print(csv_path, len(label))
        if (len(label) >= self._label_limit):
            label = label[:self._label_limit]
        return np.array(label), len(label)

    def load_items(self, root):
        folders = os.listdir(root)
        train_load_folders = []
        test_load_folders = []
        for i, folder in enumerate(folders):
            idx = i % 10
            if idx > 2:
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
                dom_path, dem_path, csv_path = [os.path.join(path, file) for file in
                                                [dom_file, dem_file, csv_file]]
                if os.path.isfile(dom_path) and os.path.isfile(dem_path) and os.path.isfile(csv_path):
                    items.append((dom_path, dem_path, csv_path))

        return items
