#!/usr/bin/env python3
#
# This script is used for downloading the dataset used by the examples.
# Dataset used: UCI / Pima Indians Diabetes (in libsvm format)

import os
import urllib.request


DATASET_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes'
TARGET_PATH = os.path.dirname(os.path.realpath(__file__)) + '/dataset.txt'


def main():
    urllib.request.urlretrieve(DATASET_URL, TARGET_PATH)


if __name__ == '__main__': main()
