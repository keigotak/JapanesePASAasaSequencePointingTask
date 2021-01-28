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

    test_label, test_args, test_preds, _, _, test_word_pos, _, _, _, _, _ = get_datasets_in_sentences('test', with_bccwj=True, with_bert=False)
    np.random.seed(71)
    random.seed(71)
    sequence_index = list(range(len(test_args)))
    random.shuffle(sequence_index)

    sentences = []
    labels = []
    categories = []
    word_pos = []
    for i in sequence_index:
        sentences.extend([''.join(test_args[i])])
        labels.extend([test_label[i].tolist()])
        categories.extend([ref_texts[''.join(test_args[i])][0]])
        index = test_word_pos[i].tolist().index(0)
        word_pos.extend([[i for i in range(index, -len(test_word_pos[i]) + index, -1)]])

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

    positionwise_all_scores, positionwise_dep_scores, positionwise_zero_scores, positionwise_itr, p_tp, p_fp, p_fn, p_counts = get_f1_with_position_wise(predictions, labels, properties, categories, word_pos, positionwise_bin_size)
    lengthwise_all_scores, lengthwise_dep_scores, lengthwise_zero_scores, lengthwise_itr, l_tp, l_fp, l_fn, l_counts = get_f1_with_sentence_length(predictions, labels, properties, categories, lengthwise_bin_size)

    ret = {
        'category': {'all': all_scores, 'dep': dep_scores, 'zero': zero_scores},
        'position': {'all': positionwise_all_scores, 'dep': positionwise_dep_scores, 'zero': positionwise_zero_scores,
                     'itr': positionwise_itr, 'tp': p_tp, 'fp': p_fp, 'fn': p_fn, 'counts': p_counts},
        'length': {'all': lengthwise_all_scores, 'dep': lengthwise_dep_scores, 'zero': lengthwise_zero_scores,
                     'itr': lengthwise_itr, 'tp': l_tp, 'fp': l_fp, 'fn': l_fn, 'counts': l_counts}
           }

    return ret


def get_step_binid_for_position_wise_analysis(num, bin_size=1):
    if num // bin_size <= -128:
        return -2
    elif -128 < num // bin_size <= -8:
        return -1
    elif -8 < num // bin_size <= 7:
        return 0
    elif 7 < num // bin_size <= 127:
        return 1
    elif 127 < num // bin_size:
        return 2
    else:
        return None


def get_itr_for_position_wise_analysis():
    return range(-2, 3, 1)
    # max_word_pos = max(list(map(max, word_pos)))
    # min_word_pos = min(list(map(min, word_pos)))
    # return range(min_word_pos // bin_size, max_word_pos // bin_size + 1)


def get_f1_with_position_wise(outputs, labels, properties, categories, word_pos, bin_size=1):
    keys = set(categories)

    itr = get_itr_for_position_wise_analysis()
    tp_histories, fp_histories, fn_histories = {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}
    counts = {key: {i: 0 for i in itr} for key in keys}
    for output, label, property, category, pos in zip(outputs, labels, properties, categories, word_pos):
        for io, il, ip, iw in zip(output, label, property, pos):
            tp_history, fp_history, fn_history = get_pr(io, il, ip)
            bin_num = get_step_binid_for_position_wise_analysis(iw)
            tp_histories[category][bin_num] += tp_history
            fp_histories[category][bin_num] += fp_history
            fn_histories[category][bin_num] += fn_history
            counts[category][bin_num] += 1

    all_scores, dep_scores, zero_scores = {key: {i: 0 for i in itr} for key in keys}, {key: {i: [] for i in itr} for key in keys}, {key: {i: [] for i in itr} for key in keys}
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


def get_step_binid_for_sentence_length_analysis(num, bin_size=1):
    if 0 <= num // bin_size < 15:
        return 0
    elif 15 <= num // bin_size < 25:
        return 1
    elif 25 <= num // bin_size < 35:
        return 2
    elif 35 <= num // bin_size < 50:
        return 3
    elif 50 <= num // bin_size < 70:
        return 4
    elif 70 <= num // bin_size:
        return 5
    else:
        return None


def get_itr_for_sentence_length_analysis():
    return range(0, 8, 1)
    # max_sentence_length = max(list(map(len, labels))) + 1
    # itr = range(0, max_sentence_length // bin_size + 1)


def get_f1_with_sentence_length(outputs, labels, properties, categories, bin_size=1):
    keys = set(categories)
    itr = get_itr_for_sentence_length_analysis()
    tp_histories, fp_histories, fn_histories = {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}, {key: {i: np.array([0]*6) for i in itr} for key in keys}
    counts = {key: {i: 0 for i in itr} for key in keys}
    for output, label, property, category in zip(outputs, labels, properties, categories):
        tp_history, fp_history, fn_history = get_f1(output, label, property)
        bin_num = get_step_binid_for_sentence_length_analysis(len(label))
        tp_histories[category][bin_num] += tp_history[0]
        fp_histories[category][bin_num] += fp_history[0]
        fn_histories[category][bin_num] += fn_history[0]
        counts[category][bin_num] += 1

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
            ['../../results/pasa-bertptr-20200419-182745-898846/detaillog_bertptr_20200419-182745.txt',
             '../../results/pasa-bertptr-20200419-182757-754887/detaillog_bertptr_20200419-182757.txt',
             '../../results/pasa-bertptr-20200419-182747-610892/detaillog_bertptr_20200419-182747.txt',
             '../../results/pasa-bertptr-20200419-182806-105741/detaillog_bertptr_20200419-182806.txt',
             '../../results/pasa-bertptr-20200419-182806-354122/detaillog_bertptr_20200419-182806.txt']),
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
             '../../results/pasa-bertptr-20200419-204241-858968/detaillog_bertptr_20200419-204241.txt']),
        'bsl_sep': (
            ['../../results/pasa-bertsl-20210125-221408-111561/bsl_20210125-221408_model-0_epoch8-f0.7906.h5.pkl'],
            ['../../results/pasa-bertsl-20210125-221408-111561/detaillog_bertsl_20210125-221408.txt'])
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
    categories = ['ブログ', '知恵袋', '出版', '新聞', '雑誌', '白書']
    tags = {'出版': 'PB', '雑誌': 'PM', '新聞': 'PN', '白書': 'OW', '知恵袋': 'OC', 'ブログ': 'OY'}
    modes = ['length', 'position']
    bin_size = {'length': arguments.length_bin_size, 'position': arguments.position_bin_size}
    with_initial_print = arguments.with_initial_print

    results = {}
    for path_pkl, path_detail in zip(files[0], files[1]):
        results[path_detail] = run(path_pkl, path_detail, lengthwise_bin_size=bin_size['length'], positionwise_bin_size=bin_size['position'], with_initial_print=with_initial_print)
        # all_scores[path_detail], dep_scores[path_detail], zero_scores[path_detail], lengthwise_all_scores[
        #     path_detail], lengthwise_dep_scores[path_detail], lengthwise_zero_scores[
        #     path_detail], lengthwise_itr, tp[path_detail], fp[path_detail], fn[path_detail], counts = run(path_pkl, path_detail, lengthwise_bin_size=bin_size, with_initial_print=with_initial_print)
        with_initial_print = False

    if arguments.reset_scores:
        remove_file(Path('../../results/bccwj-categories.txt'))

    with Path('../../results/bccwj-categories.txt').open('a', encoding='utf-8') as f:
        for category in categories:
            for path_detail in files[1]:
                line = '{}, {}, {}, {}, {}, {}'.format(model,
                                                       path_detail,
                                                       category,
                                                       results[path_detail]['category']['all'][category],
                                                       ','.join(map(str, results[path_detail]['category']['dep'][category])),
                                                       ','.join(map(str, results[path_detail]['category']['zero'][category])))
                print(line)
                f.write(line + '\n')

    for mode in modes:
        iterator = results[files[1][0]][mode]['itr']
        for category in categories:
            file_extention = tags[category]
            if bin_size != 1:
                file_extention = file_extention + '-{}'.format(bin_size[mode])
            if arguments.reset_scores:
                remove_file(Path('../../results/bccwj-{}wise-fscores-{}.txt'.format(mode, file_extention)))
            with Path('../../results/bccwj-{}wise-fscores-{}.txt'.format(mode, file_extention)).open('a', encoding='utf-8') as f:
                for i in iterator:
                    for path_detail in files[1]:
                        line = '{}, {}, {}, {}, {}, {}, {}'.format(model,
                                                                   category,
                                                                   i,
                                                                   results[path_detail][mode]['all'][category][i],
                                                                   ','.join(map(str, results[path_detail][mode]['dep'][category][i])),
                                                                   ','.join(map(str, results[path_detail][mode]['zero'][category][i])),
                                                                   path_detail)

                        print(line)
                        f.write(line + '\n')

        for category in categories:
            file_extention = tags[category]
            if bin_size != 1:
                file_extention = file_extention + '-{}'.format(bin_size[mode])
            if arguments.reset_scores:
                remove_file(Path('../../results/bccwj-category-{}-wise-fscores-{}-avg.txt'.format(mode, file_extention)))
            with Path('../../results/bccwj-category-{}-wise-fscores-{}-avg.txt'.format(mode, file_extention)).open('a', encoding='utf-8') as f:
                for i in iterator:
                    sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
                    count = 0
                    tmp_scores = []
                    for path_detail in files[1]:
                        tmp_scores.append([results[path_detail][mode]['all'][category][i]] + results[path_detail][mode]['dep'][category][i] + results[path_detail][mode]['zero'][category][i])
                        sum_tp += results[path_detail][mode]['tp'][category][i]
                        sum_fp += results[path_detail][mode]['fp'][category][i]
                        sum_fn += results[path_detail][mode]['fn'][category][i]
                    count += results[path_detail][mode]['counts'][category][i]
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
            file_extention = '-{}'.format(bin_size[mode])
        if arguments.reset_scores:
            remove_file(Path('../../results/bccwj-{}-fscores-avg{}.txt'.format(mode, file_extention)))
        with Path('../../results/bccwj-fscores-{}-avg{}.txt'.format(mode, file_extention)).open('a', encoding='utf-8') as f:
            for i in iterator:
                sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
                count = 0
                tmp_scores = []
                for category in categories:
                    for path_detail in files[1]:
                        tmp_scores.append([results[path_detail][mode]['all'][category][i]] + results[path_detail][mode]['dep'][category][i] + results[path_detail][mode]['zero'][category][i])
                        sum_tp += results[path_detail][mode]['tp'][category][i]
                        sum_fp += results[path_detail][mode]['fp'][category][i]
                        sum_fn += results[path_detail][mode]['fn'][category][i]
                    count += results[path_detail][mode]['counts'][category][i]
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
            remove_file(Path('../../results/bccwj-{}wise-final-results.txt'.format(mode, file_extention)))
        with Path('../../results/bccwj-final-results.txt'.format(file_extention)).open('a', encoding='utf-8') as f:
            sum_tp, sum_fp, sum_fn = np.array([0] * 6), np.array([0] * 6), np.array([0] * 6)
            count = 0
            tmp_scores = []
            for i in iterator:
                for category in categories:
                    for path_detail in files[1]:
                        tmp_scores.append([results[path_detail][mode]['all'][category][i]] + results[path_detail][mode]['dep'][category][i] + results[path_detail][mode]['zero'][category][i])
                        sum_tp += results[path_detail][mode]['tp'][category][i]
                        sum_fp += results[path_detail][mode]['fp'][category][i]
                        sum_fn += results[path_detail][mode]['fn'][category][i]
                    count += results[path_detail][mode]['counts'][category][i]

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
    parser.add_argument('--with_initial_print', action='store_true')
    parser.add_argument('--length_bin_size', default=10, type=int)
    parser.add_argument('--position_bin_size', default=1, type=int)
    arguments = parser.parse_args()

    model = arguments.model
    if model == 'all':
        for item in ['sl', 'spg', 'spl', 'spn', 'bsl', 'bspg', 'bspl', 'bspn', 'bsl_sep']:
            main(item, arguments)
            arguments.reset_scores = False
            arguments.with_initial_print = False
    else:
        main(model, arguments)


'''
bsl_sep
ブログ: 0.7718856731979149, 0.8143224699828473,0.8317683881064164,0.8132022471910112,0.7514619883040936, 0.5517241379310345,0.584045584045584,0.47945205479452047,0.3137254901960785
*白書: 0.7846700908731726, 0.8292682926829269,0.822065981611682,0.8558507631430187,0.7695035460992908, 0.5727272727272729,0.5958702064896756,0.5882352941176471,0.0
*新聞: 0.7636697247706422, 0.8046634403871535,0.8263419034731618,0.8093023255813953,0.7126436781609194, 0.5575221238938053,0.5916666666666667,0.430379746835443,0.3846153846153846
雑誌: 0.7706666666666666, 0.8185110663983903,0.8296769046669327,0.8343057176196033,0.7449933244325768, 0.5398058252427184,0.5844980940279543,0.41414141414141414,0.3111111111111111
知恵袋: 0.8209781605659797, 0.8764983654195423,0.8893320039880359,0.8714544357272179,0.8404761904761905, 0.5140562248995983,0.5437500000000001,0.5454545454545454,0.2173913043478261
出版: 0.8302070645554203, 0.8913891389138914,0.8960410139561379,0.9119850187265918,0.832188420019627, 0.5660621761658031,0.6134453781512604,0.4838709677419355,0.22641509433962267
'''
