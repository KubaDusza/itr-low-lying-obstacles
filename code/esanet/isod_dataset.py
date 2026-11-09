# -*- coding: utf-8 -*-
"""ISOD as an ESANet dataset, so the published RGB-D model can be trained on it unchanged.

Dropped into the ESANet clone as src/datasets/isod/pytorch_dataset.py by setup.sh. It
follows the NYUv2 dataset in that repository exactly; only the directory names, the class
list and the depth statistics differ.

Three labels, in ESANet's convention where class 0 is void and is dropped from the loss and
from the metric:

  0  void       walls, furniture, and anything else labelled not-drivable that is not one of
                our floor objects. The same ignore region eval.py uses, so the two pipelines
                score the same pixels.
  1  floor      ISOD's drivable area.
  2  obstacle   the objects lying on the floor, from prep_gt.py.
"""
import json
import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase


class ISODBase:
    SPLITS = ['train', 'valid', 'test']
    SPLIT_FILELIST_FILENAMES = {s: f'{s}.txt' for s in SPLITS}
    SPLIT_DIRS = {s: s for s in SPLITS}

    DEPTH_DIR = 'depth'
    DEPTH_RAW_DIR = 'depth'          # ISOD ships one depth image; 0 means no reading
    RGB_DIR = 'rgb'
    LABELS_DIR_FMT = 'labels_{:d}'

    N_CLASSES = [2]
    CLASS_NAMES_2 = ['void', 'floor', 'obstacle']
    CLASS_COLORS_2 = [(0, 0, 0), (60, 60, 60), (255, 40, 40)]


class ISOD(ISODBase, DatasetBase):
    def __init__(self, data_dir=None, n_classes=2, split='train',
                 depth_mode='raw', with_input_orig=False):
        super(ISOD, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['d455']

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir), data_dir
            self._data_dir = data_dir
            fp = os.path.join(self._data_dir,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
            with open(os.path.join(self._data_dir, 'depth_stats.json')) as f:
                st = json.load(f)
            self._depth_mean, self._depth_std = st['mean_mm'], st['std_mm']
        else:
            print(f'Loaded {self.__class__.__name__} dataset without files')
            self._depth_mean, self._depth_std = 0.0, 1.0

        self._class_names = getattr(self, f'CLASS_NAMES_{self._n_classes}')
        self._class_colors = np.array(
            getattr(self, f'CLASS_COLORS_{self._n_classes}'), dtype='uint8')

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, directory, filename):
        fp = os.path.join(self._data_dir, self.SPLIT_DIRS[self._split],
                          directory, f'{filename}.png')
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(fp)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._filenames[idx])

    def load_depth(self, idx):
        return self._load(self.DEPTH_DIR, self._filenames[idx])

    def load_label(self, idx):
        return self._load(self.LABELS_DIR_FMT.format(self._n_classes),
                          self._filenames[idx])

    def __len__(self):
        return len(self._filenames)
