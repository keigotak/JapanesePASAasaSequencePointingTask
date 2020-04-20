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
from BertWithJumanModel import BertWithJumanModel


def run(path_pkl, path_detail, bin_size=1, with_initial_print=True):
    test_label, test_args, test_preds, _, _, _, _, _, _, _, _ = get_datasets_in_sentences('test', with_bccwj=False, with_bert=False)
    np.random.seed(71)
    random.seed(71)
    sequence_index = list(range(len(test_args)))
    random.shuffle(sequence_index)

    sentences = []
    labels = []
    for i in sequence_index:
        sentences.extend([''.join(test_args[i])])
        labels.extend([test_label[i].tolist()])

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

    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, tp, fp, fn, counts = get_f1_with_sentence_length(predictions, labels, properties, bin_size)

    return _, _, _, lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, tp, fp, fn, counts


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
            ['../../results/pasa-bertsl-20200419-151541-906623/bsl_20200419-151541_model-0_epoch8-f0.7877.h5.pkl',
             '../../results/pasa-bertsl-20200419-151543-725292/bsl_20200419-151543_model-0_epoch8-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200419-151542-553196/bsl_20200419-151542_model-0_epoch8-f0.7918.h5.pkl',
             '../../results/pasa-bertsl-20200419-151542-129443/bsl_20200419-151542_model-0_epoch6-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200420-101530-395702/bsl_20200420-101530_model-0_epoch9-f0.7894.h5.pkl'],
            ['../../results/pasa-bertsl-20200419-151541-906623/detaillog_bertsl_20200419-151541.txt',
             '../../results/pasa-bertsl-20200419-151543-725292/detaillog_bertsl_20200419-151543.txt',
             '../../results/pasa-bertsl-20200419-151542-553196/detaillog_bertsl_20200419-151542.txt',
             '../../results/pasa-bertsl-20200419-151542-129443/detaillog_bertsl_20200419-151542.txt',
             '../../results/pasa-bertsl-20200420-101530-395702/detaillog_bertsl_20200420-101530.txt']),
        'bspn': (
            ['../../results/pasa-bertptr-20200419-151541-252443/bsp_20200419-151541_model-0_epoch7-f0.7936.h5.pkl',
             '../../results/pasa-bertptr-20200419-151541-522425/bsp_20200419-151541_model-0_epoch9-f0.7937.h5.pkl',
             '../../results/pasa-bertptr-20200419-151542-522073/bsp_20200419-151542_model-0_epoch11-f0.7924.h5.pkl',
             '../../results/pasa-bertptr-20200419-151542-441209/bsp_20200419-151542_model-0_epoch12-f0.7922.h5.pkl',
             '../../results/pasa-bertptr-20200419-151542-348408/bsp_20200419-151542_model-0_epoch11-f0.7945.h5.pkl'],
            ['../../results/pasa-bertptr-20200419-151541-252443/detaillog_bertptr_20200419-151541.txt',
             '../../results/pasa-bertptr-20200419-151541-522425/detaillog_bertptr_20200419-151541.txt',
             '../../results/pasa-bertptr-20200419-151542-522073/detaillog_bertptr_20200419-151542.txt',
             '../../results/pasa-bertptr-20200419-151542-441209/detaillog_bertptr_20200419-151542.txt',
             '../../results/pasa-bertptr-20200419-151542-348408/detaillog_bertptr_20200419-151542.txt']),
        'bspg': (
            ['../../results/pasa-bertptr-20200419-182745-898846/bsp_20200419-182745_model-0_epoch11-f0.7939.h5.pkl',
             '../../results/pasa-bertptr-20200419-182757-754887/bsp_20200419-182757_model-0_epoch9-f0.7943.h5.pkl',
             '../../results/pasa-bertptr-20200419-182747-610892/bsp_20200419-182747_model-0_epoch11-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200419-182806-105741/bsp_20200419-182806_model-0_epoch9-f0.7950.h5.pkl',
             '../../results/pasa-bertptr-20200419-182806-354122/bsp_20200419-182806_model-0_epoch8-f0.7943.h5.pkl'],
            ['../../results/pasa-bertptr-20200419-182745-898846/detaillog_bertptr_20200419-898846.txt',
             '../../results/pasa-bertptr-20200419-182757-754887/detaillog_bertptr_20200419-754887.txt',
             '../../results/pasa-bertptr-20200419-182747-610892/detaillog_bertptr_20200419-610892.txt',
             '../../results/pasa-bertptr-20200419-182806-105741/detaillog_bertptr_20200419-105741.txt',
             '../../results/pasa-bertptr-20200419-182806-354122/detaillog_bertptr_20200419-354122.txt']),
        'bspl': (
            ['../../results/pasa-bertptr-20200419-204136-400021/bsp_20200419-204136_model-0_epoch9-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200419-204241-960472/bsp_20200419-204241_model-0_epoch9-f0.7944.h5.pkl',
             '../../results/pasa-bertptr-20200419-204241-360608/bsp_20200419-204241_model-0_epoch12-f0.7940.h5.pkl',
             '../../results/pasa-bertptr-20200419-204242-504153/bsp_20200419-204242_model-0_epoch8-f0.7924.h5.pkl',
             '../../results/pasa-bertptr-20200419-204241-858968/bsp_20200419-204241_model-0_epoch8-f0.7963.h5.pkl'],
            ['../../results/pasa-bertptr-20200419-204136-400021/detaillog_bertptr_20200419-204136.txt',
             '../../results/pasa-bertptr-20200419-204241-960472/detaillog_bertptr_20200419-204241.txt',
             '../../results/pasa-bertptr-20200419-204241-360608/detaillog_bertptr_20200419-204241.txt',
             '../../results/pasa-bertptr-20200419-204242-504153/detaillog_bertptr_20200419-204242.txt',
             '../../results/pasa-bertptr-20200419-204241-858968/detaillog_bertptr_20200419-204241.txt'])
    }
    return file[model_name]


def remove_file(_path):
    try:
        os.remove(_path)
    except FileNotFoundError:
        pass


def main(model, arguments):
    files = get_files(model)
    all_scores, dep_scores, zero_scores = {}, {}, {}
    tp, fp, fn = {}, {}, {}
    counts = None
    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores = {}, {}, {}
    bin_size = arguments.bin_size

    with_initial_print = True
    for path_pkl, path_detail in zip(files[0], files[1]):
        all_scores[path_detail], dep_scores[path_detail], zero_scores[path_detail],\
        lengthwise_all_scores[path_detail], lengthwise_dep_scores[path_detail], lengthwise_zero_scores[path_detail],\
        lengthwise_itr,\
        tp[path_detail], fp[path_detail], fn[path_detail],\
        counts = run(path_pkl, path_detail, bin_size, with_initial_print)
        with_initial_print = False

    file_extention = ''
    if bin_size != 1:
        file_extention = '-{}'.format(bin_size)
    if arguments.reset_scores:
        remove_file(Path('../../results/ntc-fscores-avg{}.txt'.format(file_extention)))
    with Path('../../results/ntc-fscores-avg{}.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
        for i in lengthwise_itr:
            sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
            count = 0
            tmp_scores = []
            for path_detail in files[1]:
                tmp_scores.append([lengthwise_all_scores[path_detail][i]] + lengthwise_dep_scores[path_detail][i] + lengthwise_zero_scores[path_detail][i])
                sum_tp += tp[path_detail][i]
                sum_fp += fp[path_detail][i]
                sum_fn += fn[path_detail][i]
            count += counts[i]
            avg_tmp_scores = np.mean(np.array(tmp_scores), axis=0)
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
        remove_file(Path('../../results/ntc-final-results.txt'.format(file_extention)))
    with Path('../../results/ntc-final-results.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
        sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
        count = 0
        tmp_scores = []
        for i in lengthwise_itr:
            for path_detail in files[1]:
                tmp_scores.append([lengthwise_all_scores[path_detail][i]] + lengthwise_dep_scores[path_detail][i] + lengthwise_zero_scores[path_detail][i])
                sum_tp += tp[path_detail][i]
                sum_fp += fp[path_detail][i]
                sum_fn += fn[path_detail][i]
            count += counts[i]
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
    parser.add_argument('--bin_size', default=1, type=int)
    arguments = parser.parse_args()

    model = arguments.model
    if model == 'all':
        for item in ['sl', 'spg', 'spl', 'spn', 'bsl', 'bspg', 'bspl', 'bspn']:
            main(item, arguments)
            arguments.reset_scores = False
    else:
        main(model, arguments)
