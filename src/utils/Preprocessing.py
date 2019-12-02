# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle
from itertools import chain
import numpy as np

id_string = 0
id_type = 1
id_relation = 2
id_distance = 3
id_class = 4


def load_data(_filename, with_bccwj=False):
    if with_bccwj:
        path_to_data = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/raw', _filename).resolve())
    else:
        path_to_data = str(Path('../../data/NTC_dataset', _filename).resolve())
    with open(path_to_data, mode='rb') as f:
        ret = pickle.load(f)
    return ret


def save_data(_filename, _data, with_bccwj=False):
    if with_bccwj:
        path_to_data = str(Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc', _filename).resolve())
    else:
        path_to_data = str(Path('../../data/NTC_dataset').joinpath(_filename))
    with open(path_to_data, mode='wb') as f:
        pickle.dump(_data, f)


def save_sample(_filename, _data, with_bccwj=False):
    if with_bccwj:
        path_to_data = Path('../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc', _filename).resolve()
    else:
        path_to_data = Path('../../data/NTC_dataset').joinpath(_filename)
    with path_to_data.open(mode='w', encoding='utf-8') as f:
        for items in _data:
            if items[6] == 101:
                break
            f.write(', '.join(map(str, items)))
            f.write('\n')


def enlist_feature_and_label(_data):
    ret = []

    ret.extend(
        Parallel(n_jobs=-1)(
            [
                delayed(make_pairs)(i, item)
                for i, item in enumerate(_data)
            ]
        )
    )

    ret = list(chain.from_iterable(ret))
    sorted_ret = sorted(ret, key=lambda x: (x[5], x[6]))

    return sorted_ret


def enlist_feature_and_label_rework(_data, ignore_length=None):
    ret = []

    # for i, item in enumerate(_data):
    #     items = make_pairs_rework(i, item, ignore_length=ignore_length)
    #     ret.extend(items)

    ret.extend(
        Parallel(n_jobs=-1)(
            [
                delayed(make_pairs_rework)(i, item, ignore_length=ignore_length)
                for i, item in enumerate(_data)
            ]
        )
    )

    ret = list(chain.from_iterable(ret))
    sorted_ret = sorted(ret, key=lambda x: (x[6], x[7]))

    return sorted_ret


def enlist_for_bert(_data):
    ret = []

    # for i, item in enumerate(_data):
    #     items = make_pairs_bert(i, item)
    #     ret.extend(items)

    ret.extend(
        Parallel(n_jobs=-1)(
            [
                delayed(make_pairs_bert)(i, item)
                for i, item in enumerate(_data)
            ]
        )
    )

    ret = list(chain.from_iterable(ret))
    sorted_ret = sorted(ret, key=lambda x: (x[8], x[9]))

    return sorted_ret


def enlist_property(_data):
    ret = []

    # ret.extend(
    #     Parallel(n_jobs=-1)(
    #         [
    #             delayed(make_pairs)(i, item)
    #             for i, item in enumerate(_data)
    #         ]
    #     )
    # )

    for i, item in enumerate(_data):
        ret.extend([item[3]])

    return ret


def make_pairs(_id, _item):
    target_str = _item[id_string][0]
    _pairs = []
    for t, candidate_info in enumerate(_item[id_string][2]):
        target_type = _item[id_type][t]
        target_relation = _item[id_relation][t]
        target_distance = _item[id_distance][t]
        candidate_str = candidate_info[0]
        pair = [target_type, target_str, candidate_str, target_relation, target_distance, _id, t]
        _pairs.append(pair)
    return _pairs


def make_pairs_rework(_id, _item, ignore_length=None):
    target_str = _item[id_string][0]
    _pairs = []
    for t, candidate_info in enumerate(_item[id_string][2]):
        if ignore_length is not None and len(_item[id_type]) >= ignore_length:
            continue
        target_type = _item[id_type][t]
        target_relation = _item[id_relation][t]
        target_distance = _item[id_distance][t]
        target_class = _item[id_class][t]
        candidate_str = candidate_info[0]
        pair = [target_type, target_str, candidate_str, target_relation, target_distance, target_class, _id, t]
        _pairs.append(pair)
    return _pairs


def make_pairs_bert(_id, _item):
    _pairs = []

    target_sentence = [item[0] for item in _item[id_string][2]]
    target_index = 0
    for t, item in enumerate(_item[id_distance]):
        if item[0] == 0:
            target_index = t
            break
    target_sentence[target_index] = "[PRED] " + target_sentence[target_index]
    target_sentence_base = target_sentence.copy()

    for t in range(0, len(target_sentence_base), 1):
        target_sentence = target_sentence_base.copy()
        if target_sentence[t].find("[PRED]") != -1:
            continue
        target_sentence.insert(t, "[ARG]")
        target_sentence = " ".join(target_sentence)

        target_type = _item[id_type][t]
        target_relation = _item[id_relation][t]
        target_distance_word = _item[id_distance][t][0]
        target_distance_phase = _item[id_distance][t][1]
        target_class = _item[id_class][t]
        candidate_index = t
        pair = [target_type, candidate_index, target_index, target_sentence, target_relation, target_distance_word, target_distance_phase, target_class, _id, t]
        _pairs.append(pair)
    return _pairs


def count_words(_data):
    ret = dict()
    for sentence in _data:
        items = sentence[0][2]
        for item in items:
            if item[0] in ret.keys():
                ret[item[0]] += 1
            else:
                ret[item[0]] = 1
    return ret


def one_hot_ids(_data, _key_ids):
    label = np.empty(len(_data), dtype=np.int32)
    pred_ids = np.empty(len(_data), dtype=np.int32)
    arg_ids = np.empty(len(_data), dtype=np.int32)
    for i, item in enumerate(_data):
        label[i] = item[0]
        pred_ids[i] = one_hot_id(item[1], _key_ids)
        arg_ids[i] = one_hot_id(item[2], _key_ids)

    ret = (np.column_stack((pred_ids, arg_ids)), label)
    print('max label={}'.format(max(label)))

    return ret


def one_hot_item(_item, _key_ids):
    pred = _item[1]
    vec_pred = one_hot_word(pred, _key_ids)
    arg = _item[2]
    vec_arg = one_hot_word(arg, _key_ids)

    vec = vec_pred + vec_arg
    return [_item[0], vec]


def one_hot_id(_word, _key_ids):
    if _word in _key_ids:
        return _key_ids[_word]
    else:
        return len(_key_ids) + 1


def one_hot_word(_word, _key_ids):
    ret = [0] * (len(_key_ids) + 1)
    if _word in _key_ids:
        ret[_key_ids[_word]] = 1
    else:   # 未知語の場合
        ret[-1] = 1
    return ret


def freeze_keys(_keys):
    ret = {}
    for key in _keys:
        if key not in ret.keys():
            ret[key] = len(ret) + 1
    return ret


if __name__ == '__main__':
    with_bccwj = False

    if with_bccwj:
        tags = ["train_bccwj", "dev_bccwj", "test_bccwj"]
    else:
        tags = ['train2', 'test', 'dev']
    # tags = ['train2']
    # tags = ['dev']
    for tag in tags:
        # tag = tag + "_rework_181130"
        # tag = tag + "_190105"
        print('-- {} --'.format(tag))
        print('loading')
        raw_data = load_data('raw_' + tag + '.pkl', with_bccwj=with_bccwj)
        # raw_data = load_data('train/train_event_noun.pkl')

        print('counting')
        freq_of_words = count_words(raw_data)
        save_data('counted_' + tag + '.pkl', freq_of_words, with_bccwj=with_bccwj)

        print('enlisting')
        listed_data = enlist_feature_and_label_rework(raw_data, ignore_length=None)
        save_data('listed_' + tag + '.pkl', listed_data, with_bccwj=with_bccwj)
        save_sample('listed_' + tag + '.txt', listed_data, with_bccwj=with_bccwj)

        # print('enlisting')
        # listed_data = enlist_for_bert(raw_data)
        # save_data('bert_listed_' + tag + '.pkl', listed_data)
        # save_sample('bert_listed_' + tag + '.txt', listed_data)

        property_data = enlist_property(listed_data)
        save_data('property_' + tag + '.pkl', property_data, with_bccwj=with_bccwj)

        keys = freq_of_words.keys()
        key_ids = freeze_keys(keys)
        save_data('key_ids_' + tag + '.pkl', key_ids, with_bccwj=with_bccwj)

        print('set id')
        ided_data = one_hot_ids(listed_data, key_ids)
        save_data('ided_' + tag + '.pkl', ided_data, with_bccwj=with_bccwj)
