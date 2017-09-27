#!/usr/bin/env python3
#
# The script is used for downloading the three datasets used by the examples.
#
# All three datasets come from LIBSVM website, and are stored in LIBSVM format.
# For more details, please refer to the following link:
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
#
# Besides, all three datasets are for binary classification, since most of the
# state-of-the-arts active learning algorithms (query strategies) are only
# suitable for binary classification.
#
# The following table describes some informations about
# the three datasets: australian, diabetes, and heart.
#
# +------------+-----+----+
# |   dataset  |  N  |  D |
# +============+=====+====+
# | australian | 690 | 14 |
# +------------+-----+----+
# |  diabetes  | 768 |  8 |
# +------------+-----+----+
# |    heart   | 270 | 13 |
# +------------+-----+----+
#
# N is the number of samples, and D is the dimension of the input feature.
# labels y \in {-1, +1}

import os
import six.moves.urllib as urllib
import random


AUS_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale'
AUS_SIZE = 690

DB_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale'
DB_SIZE = 768

HT_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale'
HT_SIZE = 270

TARGET_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    print('downloading australian ...')
    rows = list(urllib.request.urlopen(AUS_URL))
    selected = random.sample(rows, AUS_SIZE)
    with open(os.path.join(TARGET_PATH, 'australian.txt'), 'wb') as f:
        for row in selected:
            f.write(row)
    print('australian downloaded successfully !\n')

    print('downloading diabetes ...')
    rows = list(urllib.request.urlopen(DB_URL))
    selected = random.sample(rows, DB_SIZE)
    with open(os.path.join(TARGET_PATH, 'diabetes.txt'), 'wb') as f:
        for row in selected:
            f.write(row)
    print('diabetes downloaded successfully !\n')

    print('downloading heart ...')
    rows = list(urllib.request.urlopen(HT_URL))
    selected = random.sample(rows, HT_SIZE)
    with open(os.path.join(TARGET_PATH, 'heart.txt'), 'wb') as f:
        for row in selected:
            f.write(row)
    print('heart downloaded successfully !')


if __name__ == '__main__':
    main()
