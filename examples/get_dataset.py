#!/usr/bin/env python3
#
# This script is used for downloading the dataset used by the examples.
# Dataset used: Statlog / Letter (in libsvm format)

import os
import urllib.request
import random


DATASET_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale'
DATASET_SIZE = 1000
TARGET_PATH = os.path.dirname(os.path.realpath(__file__)) + '/dataset.txt'


def main():
    rows = list(urllib.request.urlopen(DATASET_URL))
    selected = random.sample(rows, DATASET_SIZE)
    with open(TARGET_PATH, 'wb') as f:
        for row in selected:
            f.write(row)


if __name__ == '__main__': main()
