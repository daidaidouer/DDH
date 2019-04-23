# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import cv2
import numpy as np

class database(object):
    def __init__(self, path):
        self.lines = open(path, 'r').readlines()
        self.n_samples = len(self.lines)
        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0

    def data(self, index):
        ret_img = []
        ret_label = []
        if self._status:
            for i in index:
                ret_img.append(self._img[i])
                ret_label.append(self._label[i])
            return np.asarray(ret_img), np.asarray(ret_label)
        else:
            for i in index:
                self._img[i] = cv2.resize(cv2.imread(self.lines[i].strip().split()[0]), (256, 256))
                self._label[i] = [int(j) for j in self.lines[i].strip().split()[1:]]
                self._load_num += 1 - self._load[i]
                self._load[i] = 1
                ret_img.append(self._img[i])
                ret_label.append(self._label[i])
            if self._load_num == self.n_samples:
                self._status = 1
            return np.asarray(ret_img), np.asarray(ret_label)

    def get_labels(self):
        for i in range(self.n_samples):
            self._label[i] = [int(j) for j in self.lines[i].strip().split()[1:]]
        return np.asarray(self._label)

def import_train(config):
    return database(config['img_tr'])

def import_validation(config):
    return database(config['img_db']), database(config['img_te'])

