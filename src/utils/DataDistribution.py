# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import pickle


print('Loading data...')
# train_path = str(Path('../../data/NTC_dataset').joinpath('ided_train2_bert.pkl'))
# test_path = str(Path('../../data/NTC_dataset').joinpath('ided_test_bert.pkl'))
# dev_path = str(Path('../../data/NTC_dataset').joinpath('ided_dev_bert.pkl'))
train_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('ided_train_bccwj_bert.pkl'))
test_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('ided_test_bccwj_bert.pkl'))
dev_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('ided_dev_bccwj_bert.pkl'))

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

# count_freq('train', y_train)
# print("----------")
# count_freq('test', y_test)
# print("----------")
# count_freq('dev', y_dev)

print('Loading data...')
# train_path = str(Path('../../data/NTC_dataset').joinpath('listed_train2.pkl'))
# test_path = str(Path('../../data/NTC_dataset').joinpath('listed_test.pkl'))
# dev_path = str(Path('../../data/NTC_dataset').joinpath('listed_dev.pkl'))
# train_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_train_bccwj.pkl'))
# test_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_test_bccwj.pkl'))
# dev_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_dev_bccwj.pkl'))
train_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_train_bccwj_bert.pkl'))
test_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_test_bccwj_bert.pkl'))
dev_path = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_dev_bccwj_bert.pkl'))

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)
with open(dev_path, 'rb') as f:
    dev_data = pickle.load(f)


def get_property(_data):
    sent_lens = {}
    cnt = 0
    sent_id = _data[0][6]
    sent_len = 0
    for item in _data:
        if item[6] > sent_id:
            cnt += 1
            sent_id = item[6]
            if sent_len in sent_lens.keys():
                sent_lens[sent_len] += 1
            else:
                sent_lens[sent_len] = 1
        sent_len = item[7]
    cnt += 1
    if sent_len in sent_lens.keys():
        sent_lens[sent_len] += 1
    else:
        sent_lens[sent_len] = 1
    return {'details': sent_lens, 'total': cnt}

print('total number of sentences train: {}'.format(get_property(train_data)['total']))
print('total number of sentences test: {}'.format(get_property(test_data)['total']))
print('total number of sentences dev: {}'.format(get_property(dev_data)['total']))

'''
* ntc / ntc with BERT
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
total number of sentences train: 62489
total number of sentences test: 23981
total number of sentences dev: 12724

* bccwj
Frequency of unique values of the said array: train
[[      0       1       2       3]
 [  57982   32065   14091 2858121]]
[ 1.96  1.08  0.48 96.48]
----------
Frequency of unique values of the said array: test
[[     0      1      2      3]
 [ 10626   5883   2504 542973]]
[ 1.89  1.05  0.45 96.62]
----------
Frequency of unique values of the said array: dev
[[     0      1      2      3]
 [ 11093   6110   2931 525829]]
[ 2.03  1.12  0.54 96.31]
total number of sentences train: 78268
total number of sentences test: 14582
total number of sentences dev: 15430


* bccwj with BERT
Frequency of unique values of the said array: train
[[      0       1       2       3]
 [  57958   32049   14089 2779895]]
[ 2.01  1.11  0.49 96.39]
----------
Frequency of unique values of the said array: test
[[     0      1      2      3]
 [ 10619   5873   2503 528409]]
[ 1.94  1.07  0.46 96.53]
----------
Frequency of unique values of the said array: dev
[[     0      1      2      3]
 [ 11079   6106   2929 510419]]
[ 2.09  1.15  0.55 96.21]
total number of sentences train: 78223
total number of sentences test: 14550
total number of sentences dev: 15426
'''


