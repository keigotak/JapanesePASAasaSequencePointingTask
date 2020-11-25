from pathlib import Path
import pickle
import numpy as np
import random

import sys
sys.path.append('../')
import os
sys.path.append(os.pardir)
from utils.Datasets import get_datasets_in_sentences
from Validation import get_pr_numbers, get_f_score
from utils.Scores import get_pr
from BertWithJumanModel import BertWithJumanModel


def run(path_pkl, path_detail, lengthwise_bin_size=1, positionwise_bin_size=1, with_initial_print=True):
    test_label, test_args, test_preds, _, _, test_word_pos, _, _, _, _, _ = get_datasets_in_sentences('test', with_bccwj=False, with_bert=False)
    np.random.seed(71)
    random.seed(71)
    sequence_index = list(range(len(test_args)))
    random.shuffle(sequence_index)

    sentences = []
    labels = []
    word_pos = []
    for i in sequence_index:
        sentences.extend([''.join(test_args[i])])
        labels.extend([test_label[i].tolist()])
        index = test_word_pos[i].tolist().index(0)
        word_pos.extend([[i for i in range(index, -len(test_word_pos[i]) + index, -1)]])

    if with_initial_print:
        count_sentences = set()
        sentence_length = {}
        bert = BertWithJumanModel(device='cpu')
        for sentence in sentences:
            if sentence not in sentence_length.keys():
                bert_tokens = bert.bert_tokenizer.tokenize(" ".join(bert.juman_tokenizer.tokenize(sentence)))
                sentence_length[sentence] = (len(bert_tokens), bert_tokens)
            count_sentences.add(sentence)
        print('test: {}'.format(len(count_sentences)))
        if arguments.reset_sentences:
            with Path('../../data/NTC_dataset/test_sentences.txt').open('w', encoding='utf-8') as f:
                f.write('\n'.join(['{}, {}, {}, {}, {}, {}, {}'.format(''.join(arg), pred[0], ''.join(map(str, label)), len(arg), sentence_length[''.join(arg)][0], '|'.join(sentence_length[''.join(arg)][1]), '|'.join(arg)) for arg, pred, label in zip(test_args, test_preds, test_label)]))

    print(path_detail)

    with Path(path_pkl).open('rb') as f:
        outputs = pickle.load(f)
    if 'lstm' in path_pkl or 'bertsl' in path_pkl:
        properties = [output[1] for output in outputs]
        index = 2
    else:
        properties = [output[3] for output in outputs]
        index = 4

    with Path(path_detail).open('r', encoding='utf-8') as f:
        detailed_outputs = f.readlines()
    predictions = []
    prediction = []
    i = 0
    for line in detailed_outputs:
        words = line.strip().split(', ')
        prediction.append(int(words[7]))
        if len(prediction) == len(outputs[i][index]):
            predictions.append(prediction.copy())
            prediction = []
            i += 1

    positionwise_all_scores, positionwise_dep_scores, positionwise_zero_scores, positionwise_itr, p_tp, p_fp, p_fn, p_counts = get_f1_with_position_wise(predictions, labels, properties, word_pos, positionwise_bin_size)
    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, l_tp, l_fp, l_fn, l_counts = get_f1_with_sentence_length(predictions, labels, properties, lengthwise_bin_size)

    ret = {
        'position': {'all': positionwise_all_scores, 'dep': positionwise_dep_scores, 'zero': positionwise_zero_scores,
                     'itr': positionwise_itr, 'tp': p_tp, 'fp': p_fp, 'fn': p_fn, 'counts': p_counts},
        'length': {'all': lengthwise_all_scores, 'dep': lengthwise_dep_scores, 'zero': lengthwise_zero_scores,
                     'itr': lengthwise_itr, 'tp': l_tp, 'fp': l_fp, 'fn': l_fn, 'counts': l_counts}
           }

    return ret


def get_f1_with_position_wise(outputs, labels, properties, word_pos, bin_size=1):
    max_word_pos = max(list(map(max, word_pos)))
    min_word_pos = min(list(map(min, word_pos)))

    itr = range(min_word_pos // bin_size, max_word_pos // bin_size + 1)
    tp_histories, fp_histories, fn_histories = {i: np.array([0]*6) for i in itr}, {i: np.array([0]*6) for i in itr}, {i: np.array([0]*6) for i in itr}
    counts = {i: 0 for i in itr}
    for output, label, property, pos in zip(outputs, labels, properties, word_pos):
        for io, il, ip, iw in zip(output, label, property, pos):
            tp_history, fp_history, fn_history = get_pr(io, il, ip)
            tp_histories[iw // bin_size] += tp_history
            fp_histories[iw // bin_size] += fp_history
            fn_histories[iw // bin_size] += fn_history
            counts[iw // bin_size] += 1

    all_scores, dep_scores, zero_scores = {i: 0 for i in itr}, {i: [] for i in itr}, {i: [] for i in itr}
    for i in itr:
        num_tp = tp_histories[i].tolist()
        num_fp = fp_histories[i].tolist()
        num_fn = fn_histories[i].tolist()
        all_scores[i], dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)
        dep_scores[i].append(dep_score)
        zero_scores[i].append(zero_score)
        precisions, recalls, f1s = [], [], []
        num_tn = np.array([0] * len(num_tp))
        for _tp, _fp, _fn, _tn in zip(num_tp, num_fp, num_fn, num_tn):
            precision = 0.0
            if _tp + _fp != 0:
                precision = _tp / (_tp + _fp)
            precisions.append(precision)

            recall = 0.0
            if _tp + _fn != 0:
                recall = _tp / (_tp + _fn)
            recalls.append(recall)

            f1 = 0.0
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        dep_scores[i].extend(f1s[0:3])
        zero_scores[i].extend(f1s[3:6])

    return all_scores, dep_scores, zero_scores, itr, tp_histories, fp_histories, fn_histories, counts


def get_f1_with_sentence_length(outputs, labels, properties, bin_size=1):
    max_sentence_length = max(list(map(len, labels))) + 1
    itr = range(0, max_sentence_length // bin_size + 1)
    tp_histories, fp_histories, fn_histories = {i: np.array([0]*6) for i in itr}, {i: np.array([0]*6) for i in itr}, {i: np.array([0]*6) for i in itr}
    counts = {i: 0 for i in itr}
    for output, label, property in zip(outputs, labels, properties):
        tp_history, fp_history, fn_history = get_f1(output, label, property)
        tp_histories[len(label) // bin_size] += tp_history[0]
        fp_histories[len(label) // bin_size] += fp_history[0]
        fn_histories[len(label) // bin_size] += fn_history[0]
        counts[len(label) // bin_size] += 1

    all_scores, dep_scores, zero_scores = {i: 0 for i in itr}, {i: [] for i in itr}, {i: [] for i in itr}
    for i in itr:
        num_tp = tp_histories[i].tolist()
        num_fp = fp_histories[i].tolist()
        num_fn = fn_histories[i].tolist()
        all_scores[i], dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)
        dep_scores[i].append(dep_score)
        zero_scores[i].append(zero_score)
        precisions, recalls, f1s = [], [], []
        num_tn = np.array([0] * len(num_tp))
        for _tp, _fp, _fn, _tn in zip(num_tp, num_fp, num_fn, num_tn):
            precision = 0.0
            if _tp + _fp != 0:
                precision = _tp / (_tp + _fp)
            precisions.append(precision)

            recall = 0.0
            if _tp + _fn != 0:
                recall = _tp / (_tp + _fn)
            recalls.append(recall)

            f1 = 0.0
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        dep_scores[i].extend(f1s[0:3])
        zero_scores[i].extend(f1s[3:6])

    return all_scores, dep_scores, zero_scores, itr, tp_histories, fp_histories, fn_histories, counts


def get_f1(outputs, labels, properties):
    tp_history, fp_history, fn_history = [], [], []
    tp, fp, fn = get_pr_numbers([outputs], [labels], [properties])
    tp_history.append(tp)
    fp_history.append(fp)
    fn_history.append(fn)
    return tp_history, fp_history, fn_history


def get_files(model_name):
    file = {
        'sl': (
            ['../../results/pasa-lstm-20200329-172255-856357/sl_20200329-172255_model-0_epoch17-f0.8438.h5.pkl',
             '../../results/pasa-lstm-20200329-172256-779451/sl_20200329-172256_model-0_epoch13-f0.8455.h5.pkl',
             '../../results/pasa-lstm-20200329-172259-504777/sl_20200329-172259_model-0_epoch16-f0.8465.h5.pkl',
             '../../results/pasa-lstm-20200329-172333-352525/sl_20200329-172333_model-0_epoch16-f0.8438.h5.pkl',
             '../../results/pasa-lstm-20200329-172320-621931/sl_20200329-172320_model-0_epoch13-f0.8473.h5.pkl'],
            ['../../results/pasa-lstm-20200329-172255-856357/detaillog_lstm_20200329-172255.txt',
             '../../results/pasa-lstm-20200329-172256-779451/detaillog_lstm_20200329-172256.txt',
             '../../results/pasa-lstm-20200329-172259-504777/detaillog_lstm_20200329-172259.txt',
             '../../results/pasa-lstm-20200329-172333-352525/detaillog_lstm_20200329-172333.txt',
             '../../results/pasa-lstm-20200329-172320-621931/detaillog_lstm_20200329-172320.txt']),
        'spn': (
            ['../../results/pasa-pointer-20200328-224527-671480/sp_20200328-224527_model-0_epoch10-f0.8453.h5.pkl',
             '../../results/pasa-pointer-20200328-224523-646336/sp_20200328-224523_model-0_epoch10-f0.8450.h5.pkl',
             '../../results/pasa-pointer-20200328-224545-172444/sp_20200328-224545_model-0_epoch14-f0.8459.h5.pkl',
             '../../results/pasa-pointer-20200328-224538-434833/sp_20200328-224538_model-0_epoch11-f0.8450.h5.pkl',
             '../../results/pasa-pointer-20200328-224530-441394/sp_20200328-224530_model-0_epoch10-f0.8446.h5.pkl'],
            ['../../results/pasa-pointer-20200328-224527-671480/detaillog_pointer_20200328-224527.txt',
             '../../results/pasa-pointer-20200328-224523-646336/detaillog_pointer_20200328-224523.txt',
             '../../results/pasa-pointer-20200328-224545-172444/detaillog_pointer_20200328-224545.txt',
             '../../results/pasa-pointer-20200328-224538-434833/detaillog_pointer_20200328-224538.txt',
             '../../results/pasa-pointer-20200328-224530-441394/detaillog_pointer_20200328-224530.txt']),
        'spg': (
            ['../../results/pasa-pointer-20200328-220620-026555/sp_20200328-220620_model-0_epoch13-f0.8462.h5.pkl',
             '../../results/pasa-pointer-20200328-220701-953235/sp_20200328-220701_model-0_epoch10-f0.8466.h5.pkl',
             '../../results/pasa-pointer-20200328-220650-498845/sp_20200328-220650_model-0_epoch10-f0.8469.h5.pkl',
             '../../results/pasa-pointer-20200328-220618-338695/sp_20200328-220618_model-0_epoch9-f0.8466.h5.pkl',
             '../../results/pasa-pointer-20200328-220642-006275/sp_20200328-220642_model-0_epoch11-f0.8461.h5.pkl'],
            ['../../results/pasa-pointer-20200328-220620-026555/detaillog_pointer_20200328-220620.txt',
             '../../results/pasa-pointer-20200328-220701-953235/detaillog_pointer_20200328-220701.txt',
             '../../results/pasa-pointer-20200328-220650-498845/detaillog_pointer_20200328-220650.txt',
             '../../results/pasa-pointer-20200328-220618-338695/detaillog_pointer_20200328-220618.txt',
             '../../results/pasa-pointer-20200328-220642-006275/detaillog_pointer_20200328-220642.txt']),
        'spl': (
            ['../../results/pasa-pointer-20200329-181219-050793/sp_20200329-181219_model-0_epoch10-f0.8455.h5.pkl',
             '../../results/pasa-pointer-20200329-181242-757471/sp_20200329-181242_model-0_epoch11-f0.8455.h5.pkl',
             '../../results/pasa-pointer-20200329-181255-253679/sp_20200329-181255_model-0_epoch12-f0.8440.h5.pkl',
             '../../results/pasa-pointer-20200329-181329-741718/sp_20200329-181329_model-0_epoch9-f0.8476.h5.pkl',
             '../../results/pasa-pointer-20200329-181405-914906/sp_20200329-181405_model-0_epoch12-f0.8456.h5.pkl'],
            ['../../results/pasa-pointer-20200329-181219-050793/detaillog_pointer_20200329-181219.txt',
             '../../results/pasa-pointer-20200329-181242-757471/detaillog_pointer_20200329-181242.txt',
             '../../results/pasa-pointer-20200329-181255-253679/detaillog_pointer_20200329-181255.txt',
             '../../results/pasa-pointer-20200329-181329-741718/detaillog_pointer_20200329-181329.txt',
             '../../results/pasa-pointer-20200329-181405-914906/detaillog_pointer_20200329-181405.txt']),
        'bsl': (
            ['../../results/pasa-bertsl-20200402-123819-415751/bsl_20200402-123819_model-0_epoch7-f0.8631.h5.pkl',
             '../../results/pasa-bertsl-20200402-123818-814117/bsl_20200402-123818_model-0_epoch7-f0.8650.h5.pkl',
             '../../results/pasa-bertsl-20200402-123820-333582/bsl_20200402-123820_model-0_epoch9-f0.8620.h5.pkl',
             '../../results/pasa-bertsl-20200402-123820-545980/bsl_20200402-123820_model-0_epoch5-f0.8611.h5.pkl',
             '../../results/pasa-bertsl-20200402-201956-237530/bsl_20200402-201956_model-0_epoch11-f0.8647.h5.pkl'],
            ['../../results/pasa-bertsl-20200402-123819-415751/detaillog_bertsl_20200402-123819.txt',
             '../../results/pasa-bertsl-20200402-123818-814117/detaillog_bertsl_20200402-123818.txt',
             '../../results/pasa-bertsl-20200402-123820-333582/detaillog_bertsl_20200402-123820.txt',
             '../../results/pasa-bertsl-20200402-123820-545980/detaillog_bertsl_20200402-123820.txt',
             '../../results/pasa-bertsl-20200402-201956-237530/detaillog_bertsl_20200402-201956.txt']),
        'bspn': (
            ['../../results/pasa-bertptr-20200402-165144-157728/bsp_20200402-165144_model-0_epoch6-f0.8702.h5.pkl',
             '../../results/pasa-bertptr-20200402-165338-628976/bsp_20200402-165338_model-0_epoch10-f0.8703.h5.pkl',
             '../../results/pasa-bertptr-20200402-165557-747882/bsp_20200402-165557_model-0_epoch17-f0.8718.h5.pkl',
             '../../results/pasa-bertptr-20200402-170544-734496/bsp_20200402-170544_model-0_epoch8-f0.8698.h5.pkl',
             '../../results/pasa-bertptr-20200402-170813-804379/bsp_20200402-170813_model-0_epoch10-f0.8706.h5.pkl'],
            ['../../results/pasa-bertptr-20200402-165144-157728/detaillog_bertptr_20200402-165144.txt',
             '../../results/pasa-bertptr-20200402-165338-628976/detaillog_bertptr_20200402-165338.txt',
             '../../results/pasa-bertptr-20200402-165557-747882/detaillog_bertptr_20200402-165557.txt',
             '../../results/pasa-bertptr-20200402-170544-734496/detaillog_bertptr_20200402-170544.txt',
             '../../results/pasa-bertptr-20200402-170813-804379/detaillog_bertptr_20200402-170813.txt']),
        'bspg': (
            ['../../results/pasa-bertptr-20200402-134057-799938/bsp_20200402-134057_model-0_epoch11-f0.8703.h5.pkl',
             '../../results/pasa-bertptr-20200402-134106-778681/bsp_20200402-134106_model-0_epoch10-f0.8707.h5.pkl',
             '../../results/pasa-bertptr-20200402-134057-825245/bsp_20200402-134057_model-0_epoch6-f0.8709.h5.pkl',
             '../../results/pasa-bertptr-20200402-134057-738238/bsp_20200402-134057_model-0_epoch10-f0.8719.h5.pkl',
             '../../results/pasa-bertptr-20200402-134057-896365/bsp_20200402-134057_model-0_epoch10-f0.8709.h5.pkl'],
            ['../../results/pasa-bertptr-20200402-134057-799938/detaillog_bertptr_20200402-134057.txt',
             '../../results/pasa-bertptr-20200402-134106-778681/detaillog_bertptr_20200402-134106.txt',
             '../../results/pasa-bertptr-20200402-134057-825245/detaillog_bertptr_20200402-134057.txt',
             '../../results/pasa-bertptr-20200402-134057-738238/detaillog_bertptr_20200402-134057.txt',
             '../../results/pasa-bertptr-20200402-134057-896365/detaillog_bertptr_20200402-134057.txt']),
        'bspl': (
            ['../../results/pasa-bertptr-20200402-195131-152329/bsp_20200402-195131_model-0_epoch6-f0.8710.h5.pkl',
             '../../results/pasa-bertptr-20200402-195230-748475/bsp_20200402-195230_model-0_epoch10-f0.8705.h5.pkl',
             '../../results/pasa-bertptr-20200402-195441-889702/bsp_20200402-195441_model-0_epoch10-f0.8718.h5.pkl',
             '../../results/pasa-bertptr-20200402-200529-393340/bsp_20200402-200529_model-0_epoch8-f0.8701.h5.pkl',
             '../../results/pasa-bertptr-20200402-200821-141107/bsp_20200402-200821_model-0_epoch10-f0.8707.h5.pkl'],
            ['../../results/pasa-bertptr-20200402-195131-152329/detaillog_bertptr_20200402-195131.txt',
             '../../results/pasa-bertptr-20200402-195230-748475/detaillog_bertptr_20200402-195230.txt',
             '../../results/pasa-bertptr-20200402-195441-889702/detaillog_bertptr_20200402-195441.txt',
             '../../results/pasa-bertptr-20200402-200529-393340/detaillog_bertptr_20200402-200529.txt',
             '../../results/pasa-bertptr-20200402-200821-141107/detaillog_bertptr_20200402-200821.txt'])
    }
    return file[model_name]


def remove_file(_path):
    try:
        os.remove(_path)
    except FileNotFoundError:
        pass


def main(model, arguments):
    files = get_files(model)
    results = {}
    modes = ['length', 'position']
    bin_size = {'length': arguments.length_bin_size, 'position': arguments.position_bin_size}
    with_initial_print = arguments.with_initial_print

    for path_pkl, path_detail in zip(files[0], files[1]):
        results[path_detail] = run(path_pkl, path_detail, lengthwise_bin_size=bin_size['length'], positionwise_bin_size=bin_size['position'], with_initial_print=with_initial_print)
        with_initial_print = False

    for mode in modes:
        file_extention = ''
        if bin_size[mode] != 1:
            file_extention = '-{}'.format(bin_size[mode])
        if arguments.reset_scores:
            remove_file(Path('../../results/ntc-{}wise-fscores-avg{}.txt'.format(mode, file_extention)))
        with Path('../../results/ntc-{}wise-fscores-avg{}.txt'.format(mode, file_extention)).open('a', encoding='utf-8') as f:
            for i in results[files[1][0]][mode]['itr']:
                sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
                count = 0
                tmp_scores = []
                for path_detail in files[1]:
                    tmp_scores.append([results[path_detail][mode]['all'][i]] + results[path_detail][mode]['dep'][i] + results[path_detail][mode]['zero'][i])
                    sum_tp += results[path_detail][mode]['tp'][i]
                    sum_fp += results[path_detail][mode]['fp'][i]
                    sum_fn += results[path_detail][mode]['fn'][i]
                count += results[path_detail][mode]['counts'][i]
                all_score, dep_score, zero_score = get_f_score(sum_tp, sum_fp, sum_fn)
                line = '{}, {}, {}, {}, {}, {}'.format(model,
                                                       i,
                                                       all_score,
                                                       dep_score,
                                                       zero_score,
                                                       count)
                print(line)
                f.write(line + '\n')

        if arguments.reset_scores:
            remove_file(Path('../../results/ntc-{}wise-final-results.txt'.format(mode)))
        with Path('../../results/ntc-{}wise-final-results.txt'.format(mode)).open('a', encoding='utf-8') as f:
            sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
            count = 0
            tmp_scores = []
            for i in results[files[1][0]][mode]['itr']:
                for path_detail in files[1]:
                    tmp_scores.append([results[path_detail][mode]['all'][i]] + results[path_detail][mode]['dep'][i] + results[path_detail][mode]['zero'][i])
                    sum_tp += results[path_detail][mode]['tp'][i]
                    sum_fp += results[path_detail][mode]['fp'][i]
                    sum_fn += results[path_detail][mode]['fn'][i]
                count += results[path_detail][mode]['counts'][i]
            all_score, dep_score, zero_score = get_f_score(sum_tp, sum_fp, sum_fn)
            line = '{}, {}, {}, {}, {}, {}'.format(model,
                                                   i,
                                                   all_score,
                                                   dep_score,
                                                   zero_score,
                                                   count)
            print(line)
            f.write(line + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PASA ntc analysis')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--reset_sentences', action='store_true')
    parser.add_argument('--reset_scores', action='store_true')
    parser.add_argument('--with_initial_print', action='store_true')
    parser.add_argument('--length_bin_size', default=10, type=int)
    parser.add_argument('--position_bin_size', default=4, type=int)
    arguments = parser.parse_args()

    model = arguments.model
    if model == 'all':
        for item in ['sl', 'spg', 'spl', 'spn', 'bsl', 'bspg', 'bspl', 'bspn']:
            main(item, arguments)
            arguments.reset_scores = False
            arguments.with_initial_print = False
    else:
        main(model, arguments)
