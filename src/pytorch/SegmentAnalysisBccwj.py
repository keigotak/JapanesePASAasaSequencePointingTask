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


def main(path_pkl, path_detail, bin_size=1, with_initial_print=True):
    tags = {'PB': '出版', 'PM': '雑誌', 'PN': '新聞', 'LB': '図書館', 'OW': '白書', 'OT': '教科書', 'OP': '広報紙',
                 'OB': 'ベストセラー', 'OC': '知恵袋', 'OY': 'ブログ', 'OV': '韻文', 'OL': '法律', 'OM': '国会議事録'}
    ref_texts = {}
    mode = 'key'
    with Path('../../data/BCCWJ-DepParaPAS/raw_dict.txt').open('r', encoding='utf-8') as f:
        key = ''
        for line in f.readlines():
            line = line.strip('\n')
            if mode == 'key':
                key = line
                mode = 'val'
            elif mode == 'val':
                items = line.split(',')
                category = 'NA'
                for tag in tags.keys():
                    if tag in items[0]:
                        category = tags[tag]
                        break
                if key in ref_texts.keys():
                    print('duplicated sentence: {}'.format(key))
                else:
                    ref_texts[key] = [category, items[1]]
                mode = 'key'
    with_bert = False
    if 'bert' in path_pkl:
        with_bert = True
    test_label, test_args, test_preds, _, _, _, _, _, _, _, _ = get_datasets_in_sentences('test', with_bccwj=True, with_bert=with_bert)
    np.random.seed(71)
    random.seed(71)
    sequence_index = list(range(len(test_args)))
    random.shuffle(sequence_index)

    sentences = []
    labels = []
    categories = []
    for i in sequence_index:
        sentences.extend([''.join(test_args[i])])
        labels.extend([test_label[i].tolist()])
        categories.extend([ref_texts[''.join(test_args[i])][0]])

    if with_initial_print:
        count_categories = {'ブログ': 0, '知恵袋': 0, '出版': 0, '新聞': 0, '雑誌': 0, '白書': 0}
        for key in count_categories.keys():
            count_categories[key] = categories.count(key)
        print(count_categories)
        count_categories = {'ブログ': set(), '知恵袋': set(), '出版': set(), '新聞': set(), '雑誌': set(), '白書': set()}
        sentence_length = {}
        bert = BertWithJumanModel(device='cpu')
        for sentence, category in zip(sentences, categories):
            count_categories[category].add(sentence)
            if sentence not in sentence_length.keys():
                bert_tokens = bert.bert_tokenizer.tokenize(" ".join(bert.juman_tokenizer.tokenize(sentence)))
                sentence_length[sentence] = (len(bert_tokens), bert_tokens)
        print(','.join(['{}: {}'.format(category, len(count_categories[category])) for category in count_categories.keys()]))
        if arguments.reset_sentences:
            with Path('../../data/BCCWJ-DepParaPAS/test_sentences.txt').open('w', encoding='utf-8') as f:
                f.write('\n'.join(['{}, {}, {}, {}, {}, {}, {}, {}'.format(ref_texts[''.join(arg)][0], ''.join(arg), pred[0], ''.join(map(str, label)), len(arg), sentence_length[''.join(arg)][0], '|'.join(sentence_length[''.join(arg)][1]), '|'.join(arg)) for arg, pred, label in zip(test_args, test_preds, test_label)]))

    print(path_detail)
    # perplexity_sentences = []
    # perplexity_categories = []
    # for arg in test_args:
    #     sentence = ''.join(arg)
    #     if sentence not in perplexity_sentences:
    #         perplexity_sentences.append(sentence)
    #         perplexity_categories.append(ref_texts[sentence][0])
    # with Path('../../data/BCCWJ-DepParaPAS/test_sentences_perplexity.txt').open('w', encoding='utf-8') as f:
    #     f.write('\n'.join(['{}, {}'.format(category, sentence) for category, sentence in zip(perplexity_categories, perplexity_sentences)]))


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

    all_scores, dep_scores, zero_scores = get_f1_with_categories(predictions, labels, properties, categories)
    keys = set(categories)
    for key in keys:
        print('{}: {}, {}, {}'.format(key, all_scores[key], ','.join(map(str, dep_scores[key])), ','.join(map(str, zero_scores[key]))))

    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, tp, fp, fn, counts = get_f1_with_sentence_length(predictions, labels, properties, categories, bin_size)

    return all_scores, dep_scores, zero_scores, lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, tp, fp, fn, counts


def get_f1_with_categories(outputs, labels, properties, categories):
    keys = set(categories)
    tp_histories, fp_histories, fn_histories = {key: np.array([0]*6) for key in keys}, {key: np.array([0]*6) for key in keys}, {key: np.array([0]*6) for key in keys}
    for output, label, property, category in zip(outputs, labels, properties, categories):
        tp_history, fp_history, fn_history = get_f1(output, label, property)
        tp_histories[category] += tp_history[0]
        fp_histories[category] += fp_history[0]
        fn_histories[category] += fn_history[0]

    all_scores, dep_scores, zero_scores = {key: 0 for key in keys}, {key: [] for key in keys}, {key: [] for key in keys}
    for key in keys:
        num_tp = tp_histories[key]
        num_fp = fp_histories[key]
        num_fn = fn_histories[key]
        all_scores[key], dep_score, zero_score = get_f_score(num_tp.tolist(), num_fp.tolist(), num_fn.tolist())
        dep_scores[key].append(dep_score)
        zero_scores[key].append(zero_score)
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
        dep_scores[key].extend(f1s[0:3])
        zero_scores[key].extend(f1s[3:6])

    return all_scores, dep_scores, zero_scores


def get_f1_with_sentence_length(outputs, labels, properties, categories, bin_size=1):
    keys = set(categories)
    max_sentence_length = max(list(map(len, labels))) + 1
    itr = range(0, max_sentence_length // bin_size + 1)
    tp_histories, fp_histories, fn_histories = {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}
    counts = {key: {i: 0 for i in itr} for key in keys}
    for output, label, property, category in zip(outputs, labels, properties, categories):
        tp_history, fp_history, fn_history = get_f1(output, label, property)
        tp_histories[category][len(label) // bin_size] += tp_history[0]
        fp_histories[category][len(label) // bin_size] += fp_history[0]
        fn_histories[category][len(label) // bin_size] += fn_history[0]
        counts[category][len(label) // bin_size] += 1

    all_scores, dep_scores, zero_scores = {key: {i: 0 for i in itr}for key in keys}, {key: {i: [] for i in itr} for key in keys}, {key: {i: [] for i in itr} for key in keys}
    for key in keys:
        for i in itr:
            num_tp = tp_histories[key][i].tolist()
            num_fp = fp_histories[key][i].tolist()
            num_fn = fn_histories[key][i].tolist()
            all_scores[key][i], dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)
            dep_scores[key][i].append(dep_score)
            zero_scores[key][i].append(zero_score)
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
            dep_scores[key][i].extend(f1s[0:3])
            zero_scores[key][i].extend(f1s[3:6])

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
            ['../../results/pasa-lstm-20200329-003503-990662/sl_20200329-003503_model-0_epoch12-f0.7641.h5.pkl',
             '../../results/pasa-lstm-20200329-003529-533233/sl_20200329-003529_model-0_epoch14-f0.7686.h5.pkl',
             '../../results/pasa-lstm-20200329-003625-441811/sl_20200329-003625_model-0_epoch12-f0.7646.h5.pkl',
             '../../results/pasa-lstm-20200329-003702-744631/sl_20200329-003702_model-0_epoch14-f0.7649.h5.pkl',
             '../../results/pasa-lstm-20200329-003720-611158/sl_20200329-003720_model-0_epoch5-f0.7630.h5.pkl'],
            ['../../results/pasa-lstm-20200329-003503-990662/detaillog_lstm_20200329-003503.txt',
             '../../results/pasa-lstm-20200329-003529-533233/detaillog_lstm_20200329-003529.txt',
             '../../results/pasa-lstm-20200329-003625-441811/detaillog_lstm_20200329-003625.txt',
             '../../results/pasa-lstm-20200329-003702-744631/detaillog_lstm_20200329-003702.txt',
             '../../results/pasa-lstm-20200329-003720-611158/detaillog_lstm_20200329-003720.txt']),
        'spg': (
            ['../../results/pasa-pointer-20200329-022106-069150/sp_20200329-022106_model-0_epoch6-f0.7589.h5.pkl',
             '../../results/pasa-pointer-20200329-022109-056568/sp_20200329-022109_model-0_epoch5-f0.7621.h5.pkl',
             '../../results/pasa-pointer-20200329-022128-955906/sp_20200329-022128_model-0_epoch6-f0.7634.h5.pkl',
             '../../results/pasa-pointer-20200329-022222-724719/sp_20200329-022222_model-0_epoch5-f0.7576.h5.pkl',
             '../../results/pasa-pointer-20200329-022313-903459/sp_20200329-022313_model-0_epoch9-f0.7616.h5.pkl'],
            ['../../results/pasa-pointer-20200329-022106-069150/detaillog_pointer_20200329-022106.txt',
             '../../results/pasa-pointer-20200329-022109-056568/detaillog_pointer_20200329-022109.txt',
             '../../results/pasa-pointer-20200329-022128-955906/detaillog_pointer_20200329-022128.txt',
             '../../results/pasa-pointer-20200329-022222-724719/detaillog_pointer_20200329-022222.txt',
             '../../results/pasa-pointer-20200329-022313-903459/detaillog_pointer_20200329-022313.txt']),
        'spn': (
            ['../../results/pasa-pointer-20200329-035820-082435/sp_20200329-035820_model-0_epoch13-f0.7628.h5.pkl',
             '../../results/pasa-pointer-20200329-035828-731188/sp_20200329-035828_model-0_epoch5-f0.7584.h5.pkl',
             '../../results/pasa-pointer-20200329-035921-430627/sp_20200329-035921_model-0_epoch5-f0.7614.h5.pkl',
             '../../results/pasa-pointer-20200329-040037-823312/sp_20200329-040037_model-0_epoch7-f0.7615.h5.pkl',
             '../../results/pasa-pointer-20200329-040155-312838/sp_20200329-040155_model-0_epoch14-f0.7619.h5.pkl'],
            ['../../results/pasa-pointer-20200329-035820-082435/detaillog_pointer_20200329-035820.txt',
             '../../results/pasa-pointer-20200329-035828-731188/detaillog_pointer_20200329-035828.txt',
             '../../results/pasa-pointer-20200329-035921-430627/detaillog_pointer_20200329-035921.txt',
             '../../results/pasa-pointer-20200329-040037-823312/detaillog_pointer_20200329-040037.txt',
             '../../results/pasa-pointer-20200329-040155-312838/detaillog_pointer_20200329-040155.txt']),
        'spl': (
            ['../../results/pasa-pointer-20200329-053317-066569/sp_20200329-053317_model-0_epoch9-f0.7608.h5.pkl',
             '../../results/pasa-pointer-20200329-053420-334813/sp_20200329-053420_model-0_epoch10-f0.7654.h5.pkl',
             '../../results/pasa-pointer-20200329-053658-852976/sp_20200329-053658_model-0_epoch5-f0.7619.h5.pkl',
             '../../results/pasa-pointer-20200329-053744-584854/sp_20200329-053744_model-0_epoch14-f0.7623.h5.pkl',
             '../../results/pasa-pointer-20200329-053954-847594/sp_20200329-053954_model-0_epoch10-f0.7618.h5.pkl'],
            ['../../results/pasa-pointer-20200329-053317-066569/detaillog_pointer_20200329-053317.txt',
             '../../results/pasa-pointer-20200329-053420-334813/detaillog_pointer_20200329-053420.txt',
             '../../results/pasa-pointer-20200329-053658-852976/detaillog_pointer_20200329-053658.txt',
             '../../results/pasa-pointer-20200329-053744-584854/detaillog_pointer_20200329-053744.txt',
             '../../results/pasa-pointer-20200329-053954-847594/detaillog_pointer_20200329-053954.txt']),
        'bsl': (
            ['../../results/pasa-bertsl-20200403-112105-536009/bsl_20200403-112105_model-0_epoch6-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200402-225320-031641/bsl_20200402-225320_model-0_epoch9-f0.7894.h5.pkl',
             '../../results/pasa-bertsl-20200402-225138-903629/bsl_20200402-225138_model-0_epoch6-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200402-230314-149516/bsl_20200402-230314_model-0_epoch8-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200402-230524-638769/bsl_20200402-230524_model-0_epoch8-f0.7877.h5.pkl'],
            ['../../results/pasa-bertsl-20200403-112105-536009/detaillog_bertsl_20200403-112105.txt',
             '../../results/pasa-bertsl-20200402-225320-031641/detaillog_bertsl_20200402-225320.txt',
             '../../results/pasa-bertsl-20200402-225138-903629/detaillog_bertsl_20200402-225138.txt',
             '../../results/pasa-bertsl-20200402-230314-149516/detaillog_bertsl_20200402-230314.txt',
             '../../results/pasa-bertsl-20200402-230524-638769/detaillog_bertsl_20200402-230524.txt']),
        'bspg': (
            ['../../results/pasa-bertptr-20200403-025538-564688/bsp_20200403-025538_model-0_epoch11-f0.7939.h5.pkl',
             '../../results/pasa-bertptr-20200403-025740-192547/bsp_20200403-025740_model-0_epoch9-f0.7943.h5.pkl',
             '../../results/pasa-bertptr-20200403-025725-718275/bsp_20200403-025725_model-0_epoch11-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200403-030515-753190/bsp_20200403-030515_model-0_epoch9-f0.7950.h5.pkl',
             '../../results/pasa-bertptr-20200403-030648-760108/bsp_20200403-030648_model-0_epoch8-f0.7943.h5.pkl'],
            ['../../results/pasa-bertptr-20200403-025538-564688/detaillog_bertptr_20200403-025538.txt',
             '../../results/pasa-bertptr-20200403-025740-192547/detaillog_bertptr_20200403-025740.txt',
             '../../results/pasa-bertptr-20200403-025725-718275/detaillog_bertptr_20200403-025725.txt',
             '../../results/pasa-bertptr-20200403-030515-753190/detaillog_bertptr_20200403-030515.txt',
             '../../results/pasa-bertptr-20200403-030648-760108/detaillog_bertptr_20200403-030648.txt']),
        'bspn': (
            ['../../results/pasa-bertptr-20200403-010141-686124/bsp_20200403-010141_model-0_epoch11-f0.7924.h5.pkl',
             '../../results/pasa-bertptr-20200403-010141-667945/bsp_20200403-010141_model-0_epoch7-f0.7936.h5.pkl',
             '../../results/pasa-bertptr-20200403-010141-341382/bsp_20200403-010141_model-0_epoch12-f0.7922.h5.pkl',
             '../../results/pasa-bertptr-20200403-011035-863656/bsp_20200403-011035_model-0_epoch11-f0.7945.h5.pkl',
             '../../results/pasa-bertptr-20200403-011130-170880/bsp_20200403-011130_model-0_epoch9-f0.7937.h5.pkl'],
            ['../../results/pasa-bertptr-20200403-010141-686124/detaillog_bertptr_20200403-010141.txt',
             '../../results/pasa-bertptr-20200403-010141-667945/detaillog_bertptr_20200403-010141.txt',
             '../../results/pasa-bertptr-20200403-010141-341382/detaillog_bertptr_20200403-010141.txt',
             '../../results/pasa-bertptr-20200403-011035-863656/detaillog_bertptr_20200403-011035.txt',
             '../../results/pasa-bertptr-20200403-011130-170880/detaillog_bertptr_20200403-011130.txt']),
        'bspl': (
            ['../../results/pasa-bertptr-20200403-045040-398629/bsp_20200403-045040_model-0_epoch9-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200403-045320-503212/bsp_20200403-045320_model-0_epoch12-f0.7940.h5.pkl',
             '../../results/pasa-bertptr-20200403-045346-565331/bsp_20200403-045346_model-0_epoch9-f0.7944.h5.pkl',
             '../../results/pasa-bertptr-20200403-050141-426441/bsp_20200403-050141_model-0_epoch8-f0.7963.h5.pkl',
             '../../results/pasa-bertptr-20200403-050204-373548/bsp_20200403-050204_model-0_epoch8-f0.7924.h5.pkl'],
            ['../../results/pasa-bertptr-20200403-045040-398629/detaillog_bertptr_20200403-045040.txt',
             '../../results/pasa-bertptr-20200403-045320-503212/detaillog_bertptr_20200403-045320.txt',
             '../../results/pasa-bertptr-20200403-045346-565331/detaillog_bertptr_20200403-045346.txt',
             '../../results/pasa-bertptr-20200403-050141-426441/detaillog_bertptr_20200403-050141.txt',
             '../../results/pasa-bertptr-20200403-050204-373548/detaillog_bertptr_20200403-050204.txt'])
    }
    return file[model_name]


def remove_file(_path):
    try:
        os.remove(_path)
    except FileNotFoundError:
        pass


def run(model, arguments):
    files = get_files(model)
    all_scores, dep_scores, zero_scores = {}, {}, {}
    tp, fp, fn = {}, {}, {}
    counts = None
    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores = {}, {}, {}
    categories = ['ブログ', '知恵袋', '出版', '新聞', '雑誌', '白書']
    tags = {'出版': 'PB', '雑誌': 'PM', '新聞': 'PN', '白書': 'OW', '知恵袋': 'OC', 'ブログ': 'OY'}
    bin_size = arguments.bin_size

    with_initial_print = True
    for path_pkl, path_detail in zip(files[0], files[1]):
        all_scores[path_detail], dep_scores[path_detail], zero_scores[path_detail], lengthwise_all_scores[
            path_detail], lengthwise_dep_scores[path_detail], lengthwise_zero_scores[
            path_detail], lengthwise_itr, tp[path_detail], fp[path_detail], fn[path_detail], counts = main(path_pkl, path_detail, bin_size, with_initial_print)
        with_initial_print = False

    if arguments.reset_scores:
        remove_file(Path('../../results/bccwj-categories.txt'))

    with Path('../../results/bccwj-categories.txt').open('a', encoding='utf-8') as f:
        for category in categories:
            for path_detail in files[1]:
                line = '{}, {}, {}, {}, {}, {}'.format(model,
                                                       path_detail,
                                                       category,
                                                       all_scores[path_detail][category],
                                                       ','.join(map(str, dep_scores[path_detail][category])),
                                                       ','.join(map(str, zero_scores[path_detail][category])))
                print(line)
                f.write(line + '\n')

        for category in categories:
            file_extention = tags[category]
            if bin_size != 1:
                file_extention = file_extention + '-{}'.format(bin_size)
            if arguments.reset_scores:
                remove_file(Path('../../results/labelwise-fscores-{}.txt'.format(file_extention)))
            with Path('../../results/labelwise-fscores-{}.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
                for i in lengthwise_itr:
                    for path_detail in files[1]:
                        line = '{}, {}, {}, {}, {}, {}, {}'.format(model,
                                                                   category,
                                                                   i,
                                                                   lengthwise_all_scores[path_detail][category][i],
                                                                   ','.join(map(str, lengthwise_dep_scores[path_detail][
                                                                       category][i])),
                                                                   ','.join(map(str,
                                                                                lengthwise_zero_scores[path_detail][
                                                                                    category][i])),
                                                                   path_detail)

                        print(line)
                        f.write(line + '\n')

        for category in categories:
            file_extention = tags[category]
            if bin_size != 1:
                file_extention = file_extention + '-{}'.format(bin_size)
            if arguments.reset_scores:
                remove_file(Path('../../results/labelwise-fscores-{}-avg.txt'.format(file_extention)))
            with Path('../../results/labelwise-fscores-{}-avg.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
                for i in lengthwise_itr:
                    sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
                    count = 0
                    tmp_scores = []
                    for path_detail in files[1]:
                        tmp_scores.append([lengthwise_all_scores[path_detail][category][i]] + lengthwise_dep_scores[path_detail][category][i] + lengthwise_zero_scores[path_detail][category][i])
                        sum_tp += tp[path_detail][category][i]
                        sum_fp += fp[path_detail][category][i]
                        sum_fn += fn[path_detail][category][i]
                    count += counts[category][i]
                    avg_tmp_scores = np.mean(np.array(tmp_scores), axis=0)
                    all_score, dep_score, zero_score = get_f_score(sum_tp, sum_fp, sum_fn)
                    line = '{}, {}, {}, {}, {}, {}, {}'.format(model,
                                                               category,
                                                               i,
                                                               all_score,
                                                               dep_score,
                                                               zero_score,
                                                               count)
                    print(line)
                    f.write(line + '\n')

    file_extention = ''
    if bin_size != 1:
        file_extention = '-{}'.format(bin_size)
    if arguments.reset_scores:
        remove_file(Path('../../results/fscores-avg{}.txt').format(file_extention))
    with Path('../../results/fscores-avg{}.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
        for i in lengthwise_itr:
            sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
            count = 0
            tmp_scores = []
            for category in categories:
                for path_detail in files[1]:
                    tmp_scores.append([lengthwise_all_scores[path_detail][category][i]] + lengthwise_dep_scores[path_detail][category][i] + lengthwise_zero_scores[path_detail][category][i])
                    sum_tp += tp[path_detail][category][i]
                    sum_fp += fp[path_detail][category][i]
                    sum_fn += fn[path_detail][category][i]
                count += counts[category][i]
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
        remove_file(Path('../../results/final-results.txt').format(file_extention))
    with Path('../../results/final-results.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
        sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
        count = 0
        tmp_scores = []
        for i in lengthwise_itr:
            for category in categories:
                for path_detail in files[1]:
                    tmp_scores.append([lengthwise_all_scores[path_detail][category][i]] + lengthwise_dep_scores[path_detail][category][i] + lengthwise_zero_scores[path_detail][category][i])
                    sum_tp += tp[path_detail][category][i]
                    sum_fp += fp[path_detail][category][i]
                    sum_fn += fn[path_detail][category][i]
                count += counts[category][i]
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PASA bccwj analysis')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--reset_sentences', action='store_true')
    parser.add_argument('--reset_scores', action='store_true')
    parser.add_argument('--bin_size', default=1, type=int)
    arguments = parser.parse_args()

    model = arguments.model
    if model == 'all':
        for item in ['sl', 'spg', 'spl', 'spn', 'bsl', 'bspg', 'bspl', 'bspn']:
            run(item, arguments)
    else:
        run(model, arguments)
