# -*- coding: utf-8 -*-
from pathlib import Path
import pickle
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from .KerasUtils import *


def load_vocab(_type='original'):
    if _type == 'reload_fasttext':
        _, vocab, _ = load_fasttext()
        train_vocab = vocab
        test_vocab = vocab
        dev_vocab = vocab
    else:
        train_vocab = str(Path('../data/NTC_dataset').joinpath('key_ids_train.pkl'))
        test_vocab = str(Path('../data/NTC_dataset').joinpath('key_ids_test.pkl'))
        dev_vocab = str(Path('../data/NTC_dataset').joinpath('key_ids_dev.pkl'))
        with open(train_vocab, 'rb') as f:
            train_vocab = pickle.load(f)
        with open(test_vocab, 'rb') as f:
            test_vocab = pickle.load(f)
        with open(dev_vocab, 'rb') as f:
            dev_vocab = pickle.load(f)
    return train_vocab, test_vocab, dev_vocab


def get_id(_vocab, _data):
    if _data in _vocab.keys():
        return _vocab[_data]
    else:
        return max(_vocab.values()) + 1


def load_data(with_bccwj=False):
    if with_bccwj:
        train_path = str(Path('../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_train.pkl'))
        test_path = str(Path('../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_test.pkl'))
        dev_path = str(Path('../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc').joinpath('listed_dev.pkl'))
    else:
        train_path = str(Path('../data/NTC_dataset').joinpath('listed_train.pkl'))
        test_path = str(Path('../data/NTC_dataset').joinpath('listed_test.pkl'))
        dev_path = str(Path('../data/NTC_dataset').joinpath('listed_dev.pkl'))

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    with open(dev_path, 'rb') as f:
        dev_data = pickle.load(f)
    return train_data, test_data, dev_data


def preprocessing(_base_data,  _base_vocab):
    # __base_data = pd.DataFrame(_base_data)
    X_arg = []
    X_pred = []
    y_ga = []
    y_ni = []
    y_wo = []
    y_all = []
    y_property = []
    _columns = 0
    _X_arg = []
    _X_pred = []
    _y_ga = []
    _y_ni = []
    _y_wo = []
    _y_all = []
    _y_property = []
    for __base_data in _base_data:
        if __base_data[5] == _columns:
            _X_arg.append(get_id(_base_vocab, __base_data[2]))
            _X_pred.append(get_id(_base_vocab, __base_data[1]))
            if __base_data[0] == 0:
                _y_ga.extend([1])
                _y_ni.extend([0])
                _y_wo.extend([0])
                _y_all.extend([1])
            elif __base_data[0] == 1:
                _y_ga.extend([0])
                _y_ni.extend([0])
                _y_wo.extend([1])
                _y_all.extend([1])
            elif __base_data[0] == 2:
                _y_ga.extend([0])
                _y_ni.extend([1])
                _y_wo.extend([0])
                _y_all.extend([1])
            else:
                _y_ga.extend([0])
                _y_ni.extend([0])
                _y_wo.extend([0])
                _y_all.extend([0])
            _y_property.append(__base_data[3])
        else:
            X_arg.append(_X_arg)
            X_pred.append(_X_pred)
            y_ga.append(_y_ga)
            y_ni.append(_y_ni)
            y_wo.append(_y_wo)
            y_all.append(_y_all)
            y_property.append(_y_property)
            _columns = __base_data[5]
            _X_arg = []
            _X_pred = []
            _y_ga = []
            _y_ni = []
            _y_wo = []
            _y_all = []
            _y_property = []
            _X_arg.append(get_id(_base_vocab, __base_data[2]))
            _X_pred.append(get_id(_base_vocab, __base_data[1]))
            if __base_data[0] == 0:
                _y_ga.extend([1])
                _y_ni.extend([0])
                _y_wo.extend([0])
                _y_all.extend([1])
            elif __base_data[0] == 1:
                _y_ga.extend([0])
                _y_ni.extend([0])
                _y_wo.extend([1])
                _y_all.extend([1])
            elif __base_data[0] == 2:
                _y_ga.extend([0])
                _y_ni.extend([1])
                _y_wo.extend([0])
                _y_all.extend([1])
            else:
                _y_ga.extend([0])
                _y_ni.extend([0])
                _y_wo.extend([0])
                _y_all.extend([0])
            _y_property.append(__base_data[3])
    X_arg.append(_X_arg)
    X_pred.append(_X_pred)
    y_ga.append(_y_ga)
    y_ni.append(_y_ni)
    y_wo.append(_y_wo)
    y_all.append(_y_all)
    y_property.append(_y_property)
    return X_arg, X_pred, {'ga': y_ga, 'ni': y_ni, 'wo': y_wo, 'all': y_all},  y_property


def categorizing(_data):
    num_classes = 4
    _y = []
    for _item in _data:
        _y.append(to_categorical(_item, num_classes))
    return _y


def one_hot_argmax(_data):
    index = np.argmax(_data)
    array = np.zeros_like(_data, dtype='int32')
    array[index] = 1
    return array


def clip(_data):
    return np.clip(np.round(_data), 0, 1)


# 0:が格, 1:を格, 2:に格, 3:Else
def change_label_to_sequence(_ga_lists, _ni_lists, _wo_lists, is_gold=False, method='clip'):
    rets = []
    for _ga_list, _ni_list, _wo_list in zip(_ga_lists, _ni_lists, _wo_lists):
        ret = []

        if not is_gold:
            if method == 'argmax':
                _ga_list = one_hot_argmax(_ga_list)
                _ni_list = one_hot_argmax(_ni_list)
                _wo_list = one_hot_argmax(_wo_list)
            elif method == 'clip':
                _ga_list = clip(_ga_list)
                _ni_list = clip(_ni_list)
                _wo_list = clip(_wo_list)

        _ga_list = _ga_list[:-1]
        _ni_list = _ni_list[:-1]
        _wo_list = _wo_list[:-1]

        for _ga, _ni, _wo in zip(_ga_list, _ni_list, _wo_list):
            if _ga == 0:
                if _ni == 0:
                    if _wo == 0:
                        ret.append(3)
                    elif _wo == 1:
                        ret.append(1)
                elif _ni == 1:
                    if _wo == 0:
                        ret.append(2)
                    elif _wo == 1:
                        ret.append([2, 1])
            elif _ga == 1:
                if _ni == 0:
                    if _wo == 0:
                        ret.append(0)
                    elif _wo == 1:
                        ret.append([0, 1])
                elif _ni == 1:
                    if _wo == 0:
                        ret.append([0, 2])
                    elif _wo == 1:
                        ret.append([0, 1, 2])
        rets.append(ret)
    return rets
