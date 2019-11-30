# -*- coding: utf-8 -*-
from pathlib import Path
import datetime
import argparse
import numpy as np
import torch
import sys


def get_save_dir(_tag, _now):
    dir_tag = _tag + "-{0:%Y%m%d-%H%M%S}".format(_now)
    _path = Path('../../results').joinpath(dir_tag)
    _path.mkdir(exist_ok=True)
    return _path, str(_path.resolve())


def get_now():
    return datetime.datetime.now()


def get_pasa():
    return 'pasa'


def get_argparser():
    parser = argparse.ArgumentParser(description='PASA task')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vocab_thresh', type=int, default=-1)
    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--fc1_size', type=int, default=128)
    parser.add_argument('--fc2_size', type=int, default=64)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--hyp', action='store_false')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm', 'pointer', 'combination', 'lstmnd', 'lstmnz', 'slmhatten', 'bertsl', 'bertptr', 'bertslnr', 'bertptrnr'])
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adadelta', 'sgd'])
    parser.add_argument('--dev_size', type=int, default=4)
    parser.add_argument('--clip', type=int, default=4)
    parser.add_argument('--type', type=str, default='sentence', choices=['sentence', 'pair'])
    parser.add_argument('--earlystop', type=int, default=-1)
    parser.add_argument('--printevery', type=int, default=1)
    parser.add_argument('--num_segment', type=int, default=1)
    parser.add_argument('--embed', type=str, default='original', choices=['original', 'w2v', 'fasttext', 'cbow', 'cbow-retrofitting', 'glove', 'glove-retrofitting', 'skipgram', 'skipgram-retrofitting'])
    parser.add_argument('--max_eval', type=int, default=100)
    parser.add_argument('--init_checkpoint', type=str, default='')
    parser.add_argument('--init_trials', type=str, default='')
    parser.add_argument('--trials_key', type=str, default='')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--gpu_watch', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--spreadsheet', action='store_true')
    parser.add_argument('--line', action='store_true')
    parser.add_argument('--add_null_word', action='store_true')
    parser.add_argument('--add_null_weight', action='store_true')
    parser.add_argument('--add_loss_weight', action='store_true')
    parser.add_argument('--decode', type=str, default='global_argmax', choices=['ordered', 'global_argmax', 'no_decoder'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--without_linear', action='store_true')
    parser.add_argument('--num_data', type=int, default=-1)
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--with_bccwj', action='store_true')
    arguments = parser.parse_args()
    # print(arguments)
    return arguments


def get_device_id(args):
    device_id = args.split(',')
    device_id = list(map(int, device_id))
    return device_id


def get_cuda_id(args):
    cuda_id = args.split(',')
    cuda_id = range(len(cuda_id))
    return cuda_id


def get_separated_label(labels):
    rets_ga = []
    rets_ni = []
    rets_wo = []
    for batch in labels:
        ret_ga = [1 if item == 0 else 0 if item != -1 else -1 for item in batch]
        ret_ni = [1 if item == 2 else 0 if item != -1 else -1 for item in batch]
        ret_wo = [1 if item == 1 else 0 if item != -1 else -1 for item in batch]
        rets_ga.append(ret_ga)
        rets_ni.append(ret_ni)
        rets_wo.append(ret_wo)
    return np.array(rets_ga), np.array(rets_ni), np.array(rets_wo)


def concat_labels(ga_labels, ni_labels, wo_labels):
    rets = []
    for ga_label, ni_label, wo_label in zip(ga_labels, ni_labels, wo_labels):
        ret = []
        for ga_item, ni_item, wo_item in zip(ga_label[1:], ni_label[1:], wo_label[1:]):
            if ga_item == 0 and ni_item == 0 and wo_item == 0:
                ret.append(3)
            elif ga_item == 1 and ni_item == 0 and wo_item == 0:
                ret.append(0)
            elif ga_item == 0 and ni_item == 1 and wo_item == 0:
                ret.append(2)
            elif ga_item == 0 and ni_item == 0 and wo_item == 1:
                ret.append(1)
            elif ga_item == 1 and ni_item == 1 and wo_item == 0:
                ret.append([0, 2])
            elif ga_item == 1 and ni_item == 0 and wo_item == 1:
                ret.append([0, 1])
            elif ga_item == 0 and ni_item == 1 and wo_item == 1:
                ret.append([2, 1])
            elif ga_item == 1 and ni_item == 1 and wo_item == 1:
                ret.append([0, 2, 1])
        rets.append(ret)
    return rets


def reveal_prediction(pointers, word_pos, pred_id, type_index):
    pred_index = get_pred_pos(word_pos, pred_id)

    rets = []
    for index, batch in zip(pred_index, pointers):
        ret = []
        for item in batch:
            if item == index:
                ret.append(type_index)
            else:
                ret.append(3)
        rets.append(ret)
    return np.array(rets)


def get_pred_pos(word_pos, pred_id):
    rets = []
    for batch in word_pos:
        for index, item in enumerate(batch):
            if item == pred_id:
                rets.append(index)
    return rets


def add_null_label(label):
    if 1 in label:
        ret = [0] + label
    else:
        ret = [1] + label
    return ret


def get_index(label):
    if 1 in label:
        return label.index(1)
    return None


def get_pointer_label(labels):
    rets_ga = []
    rets_ni = []
    rets_wo = []
    for label in labels:
        # output shape: Sentence_length
        label_ga = [1 if item == 0 else -1 if item == -1 else 0 for item in label]
        label_ni = [1 if item == 2 else -1 if item == -1 else 0 for item in label]
        label_wo = [1 if item == 1 else -1 if item == -1 else 0 for item in label]

        # output shape: Sentence_length+1
        label_ga = add_null_label(label_ga)
        label_ni = add_null_label(label_ni)
        label_wo = add_null_label(label_wo)

        # output shape: 1
        ret_ga = get_index(label_ga)
        ret_ni = get_index(label_ni)
        ret_wo = get_index(label_wo)

        rets_ga.append(ret_ga)
        rets_ni.append(ret_ni)
        rets_wo.append(ret_wo)

    # output shape: Batch, 1
    return np.array(rets_ga), np.array(rets_ni), np.array(rets_wo)


def add_null(items, null_item):
    items = np.hstack([[[null_item]] * items.shape[0], items])
    return items


def add_null_y(ga, ni, wo, null_label):
    ga_np = []
    ni_np = []
    wo_np = []
    for i in range(ga.shape[0]):
        ga_null = get_null_label(ga[i], null_label)
        ga_data = np.hstack([ga_null, ga[i].data])
        ga_data = np.vstack([ga_data, [-1] * (ga[i].shape[1] + 1)])
        ga_np.append(ga_data)
        ni_null = get_null_label(ni[i], null_label)
        ni_data = np.hstack([ni_null, ni[i].data])
        ni_data = np.vstack([ni_data, [-1] * (ni[i].shape[1] + 1)])
        ni_np.append(ni_data)
        wo_null = get_null_label(wo[i], null_label)
        wo_data = np.hstack([wo_null, wo[i].data])
        wo_data = np.vstack([wo_data, [-1] * (wo[i].shape[1] + 1)])
        wo_np.append(wo_data)
    # ga_np = [[np.argmax(item) for item in items] for items in ga_np]
    # ni_np = [[np.argmax(item) for item in items] for items in ni_np]
    # wo_np = [[np.argmax(item) for item in items] for items in wo_np]
    return np.array(ga_np), np.array(ni_np), np.array(wo_np)


def get_null_label(batch, null_label):
    rets = []
    for list in batch:
        ret = null_label
        for index, item in enumerate(list):
            if item == 0:
                pass
            elif item == 1:
                ret = 0
                break
            elif item == -1:
                if index == 0:
                    ret = -1
                break
            else:
                break
        rets.append([ret])
    return rets


def translate_score_and_loss(value):
    return 1. - value


def print_b(string):
    sys.stdout.buffer.write(str(string).encode('utf-8'))
    sys.stdout.buffer.write('\n'.encode('utf-8'))


if __name__ == "__main__":
    labels = [[3,3,0,1,2,3,3,3,3,3], [3,0,1,2,3,3,3,-1,-1,-1]]
    pos = [[9,8,7,6,5,4,3,1,2,1], [6,5,4,3,1,2,1,3,4,5]]
    pred_id = 2
    ret_ga, ret_ni, ret_wo = get_pointer_label(labels)
    print(ret_ga)
    print(ret_ni)
    print(ret_wo)
