#!/usr/bin/env python3
#
# This script is used for downloading the datasets used by the examples.
# The datasets used in examples are: iris, glass, and segment.
# The following table describes some informations related to the three datasets.

# === data ==== N ==== K ==== D ==
# |  iris   |  150  |  3  |   4  |
# ================================
# |  glass  |  214  |  6  |   9  |
# ================================
# | segment |  2310 |  7  |  19  |
# ================================

# Note: N is the number of examples, including both training and testing set;
#       K is the number of classes;
#       D is the input feature dimension, i.e. the length of feature vector.

import os
import urllib.request
import random


IRIS_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale'
IRIS_SIZE = 150

GLASS_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale'
GLASS_SIZE = 214

SEGMENT_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/segment.scale'
SEGMENT_SIZE = 2310

TARGET_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    print('downloading iris ...')
    rows = list(urllib.request.urlopen(IRIS_URL))
    selected = random.sample(rows, IRIS_SIZE)
    with open(TARGET_PATH + '/iris.txt', 'wb') as f:
        for row in selected:
            f.write(row)

    print('downloading glass ...')
    rows = list(urllib.request.urlopen(GLASS_URL))
    selected = random.sample(rows, GLASS_SIZE)
    with open(TARGET_PATH + '/glass.txt', 'wb') as f:
        for row in selected:
            f.write(row)

    print('downloading segment ...')
    rows = list(urllib.request.urlopen(SEGMENT_URL))
    selected = random.sample(rows, SEGMENT_SIZE)
    with open(TARGET_PATH + '/segment.txt', 'wb') as f:
        for row in selected:
            f.write(row)


if __name__ == '__main__':
    main()
