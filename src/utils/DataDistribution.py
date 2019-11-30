# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import pickle


print('Loading data...')
train_path = str(Path('../../data/NTC_dataset').joinpath('ided_train.pkl'))
test_path = str(Path('../../data/NTC_dataset').joinpath('ided_test.pkl'))
dev_path = str(Path('../../data/NTC_dataset').joinpath('ided_dev.pkl'))
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)
with open(dev_path, 'rb') as f:
    dev_data = pickle.load(f)

X_train, y_train = train_data[0], train_data[1]
X_test, y_test = test_data[0], test_data[1]
X_dev, y_dev = dev_data[0], dev_data[1]


def count_freq(_tag, _data):
    unique_elements, counts_elements = np.unique(_data, return_counts=True)
    print("Frequency of unique values of the said array: {}".format(_tag))
    sum_counts = np.sum(counts_elements)
    print(np.asarray((unique_elements, counts_elements)))
    print(np.round(counts_elements / sum_counts * 100, 2))

count_freq('train', y_train)
print("----------")
count_freq('test', y_test)
print("----------")
count_freq('dev', y_dev)


'''
Frequency of unique values of the said array: train
[[      0       1       2       3]
 [  50890   27216    6701 2102725]]
[ 2.33  1.24  0.31 96.12]
----------
Frequency of unique values of the said array: test
[[     0      1      2      3]
 [ 19064  10372   2890 819131]]
[ 2.24  1.22  0.34 96.2 ]
----------
Frequency of unique values of the said array: dev
[[     0      1      2      3]
 [ 10121   5532   1850 437920]]
[ 2.22  1.21  0.41 96.16]
'''


