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


def main(path_pkl, path_detail):
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
    count_categories = {'ブログ': 0, '知恵袋': 0, '出版': 0, '新聞': 0, '雑誌': 0, '白書': 0}
    for key in count_categories.keys():
        count_categories[key] = categories.count(key)
    print(count_categories)
    count_categories = {'ブログ': set(), '知恵袋': set(), '出版': set(), '新聞': set(), '雑誌': set(), '白書': set()}
    for sentence, category in zip(sentences, categories):
        count_categories[category].add(sentence)
    print(','.join(['{}: {}'.format(category, len(count_categories[category])) for category in count_categories.keys()]))
    if arguments.reset:
        with Path('../../data/BCCWJ-DepParaPAS/test_sentences.txt').open('w', encoding='utf-8') as f:
            f.write('\n'.join(['{}, {}, {}, {}'.format(ref_texts[''.join(arg)][0], ''.join(arg), pred[0], ''.join(map(str, label))) for arg, pred, label in zip(test_args, test_preds, test_label)]))

    with Path(path_pkl).open('rb') as f:
        outputs = pickle.load(f)
    if 'pointer' in path_pkl or 'sp' in path_pkl or 'ptr' in path_pkl:
        properties = [output[3] for output in outputs]
        index = 4
    else:
        properties = [output[1] for output in outputs]
        index = 2

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
    return all_scores, dep_scores, zero_scores


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
            ['../../results/pasa-lstm-20200105-035952/seq_20200105-035952_model-0_epoch19-f0.7649.h5.pkl',
             '../../results/pasa-lstm-20200105-040035/seq_20200105-040035_model-0_epoch17-f0.7646.h5.pkl',
             '../../results/pasa-lstm-20200105-040101/seq_20200105-040101_model-0_epoch10-f0.7630.h5.pkl',
             '../../results/pasa-lstm-20200105-040001/seq_20200105-040001_model-0_epoch19-f0.7686.h5.pkl',
             '../../results/pasa-lstm-20200105-040118/seq_20200105-040118_model-0_epoch17-f0.7641.h5.pkl'],
            ['../../results/pasa-lstm-20200105-035952/detaillog_lstm_20200105-035952.txt',
             '../../results/pasa-lstm-20200105-040035/detaillog_lstm_20200105-040035.txt',
             '../../results/pasa-lstm-20200105-040101/detaillog_lstm_20200105-040101.txt',
             '../../results/pasa-lstm-20200105-040001/detaillog_lstm_20200105-040001.txt',
             '../../results/pasa-lstm-20200105-040118/detaillog_lstm_20200105-040118.txt']),
        'spg': (
            ['../../results/pasa-pointer-20200104-130555/ptr_20200104-130555_model-0_epoch11-f0.7589.h5.pkl',
             '../../results/pasa-pointer-20200104-130649/ptr_20200104-130649_model-0_epoch11-f0.7634.h5.pkl',
             '../../results/pasa-pointer-20200104-130624/ptr_20200104-130624_model-0_epoch10-f0.7621.h5.pkl',
             '../../results/pasa-pointer-20200104-150504/ptr_20200104-150504_model-0_epoch10-f0.7576.h5.pkl',
             '../../results/pasa-pointer-20200104-150511/ptr_20200104-150511_model-0_epoch14-f0.7616.h5.pkl'],
            ['../../results/pasa-pointer-20200104-130555/detaillog_pointer_20200104-130555.txt',
             '../../results/pasa-pointer-20200104-130649/detaillog_pointer_20200104-130649.txt',
             '../../results/pasa-pointer-20200104-130624/detaillog_pointer_20200104-130624.txt',
             '../../results/pasa-pointer-20200104-150504/detaillog_pointer_20200104-150504.txt',
             '../../results/pasa-pointer-20200104-150511/detaillog_pointer_20200104-150511.txt']),
        'spn': (
            ['../../results/pasa-pointer-20200104-150500/ptr_20200104-150500_model-0_epoch10-f0.7614.h5.pkl',
             '../../results/pasa-pointer-20200104-150547/ptr_20200104-150547_model-0_epoch10-f0.7584.h5.pkl',
             '../../results/pasa-pointer-20200104-150552/ptr_20200104-150552_model-0_epoch18-f0.7628.h5.pkl',
             '../../results/pasa-pointer-20200104-152527/ptr_20200104-152527_model-0_epoch19-f0.7619.h5.pkl',
             '../../results/pasa-pointer-20200104-152703/ptr_20200104-152703_model-0_epoch12-f0.7615.h5.pkl'],
            ['../../results/pasa-pointer-20200104-150500/detaillog_pointer_20200104-150500.txt',
             '../../results/pasa-pointer-20200104-150547/detaillog_pointer_20200104-150547.txt',
             '../../results/pasa-pointer-20200104-150552/detaillog_pointer_20200104-150552.txt',
             '../../results/pasa-pointer-20200104-152527/detaillog_pointer_20200104-152527.txt',
             '../../results/pasa-pointer-20200104-152703/detaillog_pointer_20200104-152703.txt']),
        'spl': (
            ['../../results/pasa-pointer-20200104-152844/ptr_20200104-152844_model-0_epoch10-f0.7619.h5.pkl',
             '../../results/pasa-pointer-20200104-165359/ptr_20200104-165359_model-0_epoch9-f0.7608.h5.pkl',
             '../../results/pasa-pointer-20200104-165505/ptr_20200104-165505_model-0_epoch15-f0.7654.h5.pkl',
             '../../results/pasa-pointer-20200104-165511/ptr_20200104-165511_model-0_epoch19-f0.7623.h5.pkl',
             '../../results/pasa-pointer-20200104-165543/ptr_20200104-165543_model-0_epoch15-f0.7618.h5.pkl'],
            ['../../results/pasa-pointer-20200104-152844/detaillog_pointer_20200104-152844.txt',
             '../../results/pasa-pointer-20200104-165359/detaillog_pointer_20200104-165359.txt',
             '../../results/pasa-pointer-20200104-165505/detaillog_pointer_20200104-165505.txt',
             '../../results/pasa-pointer-20200104-165511/detaillog_pointer_20200104-165511.txt',
             '../../results/pasa-pointer-20200104-165543/detaillog_pointer_20200104-165543.txt']),
        'bsl': (
            ['../../results/pasa-bertsl-20200104-152437/ptr_20200104-152437_model-0_epoch11-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200104-153949/ptr_20200104-153949_model-0_epoch13-f0.7918.h5.pkl',
             '../../results/pasa-bertsl-20200104-154415/ptr_20200104-154415_model-0_epoch14-f0.7894.h5.pkl',
             '../../results/pasa-bertsl-20200104-154538/ptr_20200104-154538_model-0_epoch13-f0.7916.h5.pkl',
             '../../results/pasa-bertsl-20200104-163623/ptr_20200104-163623_model-0_epoch13-f0.7877.h5.pkl'],
            ['../../results/pasa-bertsl-20200104-152437/detaillog_bertsl_20200104-152437.txt',
             '../../results/pasa-bertsl-20200104-153949/detaillog_bertsl_20200104-153949.txt',
             '../../results/pasa-bertsl-20200104-154415/detaillog_bertsl_20200104-154415.txt',
             '../../results/pasa-bertsl-20200104-154538/detaillog_bertsl_20200104-154538.txt',
             '../../results/pasa-bertsl-20200104-163623/detaillog_bertsl_20200104-163623.txt']),
        'bspg': (
            ['../../results/pasa-bertptr-20200204-225742/ptr_20200204-225742_model-0_epoch16-f0.7939.h5.pkl',
             '../../results/pasa-bertptr-20200204-233732/ptr_20200204-233732_model-0_epoch16-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200204-000531/ptr_20200204-000531_model-0_epoch14-f0.7943.h5.pkl',
             '../../results/pasa-bertptr-20200204-000527/ptr_20200204-000527_model-0_epoch14-f0.7950.h5.pkl',
             '../../results/pasa-bertptr-20200204-000523/ptr_20200204-000523_model-0_epoch13-f0.7943.h5.pkl'],
            ['../../results/pasa-bertptr-20200204-225742/detaillog_bertptr_20200204-225742.txt',
             '../../results/pasa-bertptr-20200204-233732/detaillog_bertptr_20200204-233732.txt',
             '../../results/pasa-bertptr-20200204-000531/detaillog_bertptr_20200204-000531.txt',
             '../../results/pasa-bertptr-20200204-000527/detaillog_bertptr_20200204-000527.txt',
             '../../results/pasa-bertptr-20200204-000523/detaillog_bertptr_20200204-000523.txt']),
        'bspn': (
            ['../../results/pasa-bertptr-20200104-165547/ptr_20200104-165547_model-0_epoch16-f0.7924.h5.pkl',
             '../../results/pasa-bertptr-20200104-170518/ptr_20200104-170518_model-0_epoch17-f0.7922.h5.pkl',
             '../../results/pasa-bertptr-20200104-171529/ptr_20200104-171529_model-0_epoch12-f0.7936.h5.pkl',
             '../../results/pasa-bertptr-20200104-172439/ptr_20200104-172439_model-0_epoch16-f0.7945.h5.pkl',
             '../../results/pasa-bertptr-20200104-172634/ptr_20200104-172634_model-0_epoch14-f0.7937.h5.pkl'],
            ['../../results/pasa-bertptr-20200104-165547/detaillog_bertptr_20200104-165547.txt',
             '../../results/pasa-bertptr-20200104-170518/detaillog_bertptr_20200104-170518.txt',
             '../../results/pasa-bertptr-20200104-171529/detaillog_bertptr_20200104-171529.txt',
             '../../results/pasa-bertptr-20200104-172439/detaillog_bertptr_20200104-172439.txt',
             '../../results/pasa-bertptr-20200104-172634/detaillog_bertptr_20200104-172634.txt']),
        'bspl': (
            ['../../results/pasa-bertptr-20200205-001801/ptr_20200205-001801_model-0_epoch14-f0.7955.h5.pkl',
             '../../results/pasa-bertptr-20200204-004930/ptr_20200204-004930_model-0_epoch17-f0.7940.h5.pkl',
             '../../results/pasa-bertptr-20200205-005949/ptr_20200205-005949_model-0_epoch14-f0.7944.h5.pkl',
             '../../results/pasa-bertptr-20200204-004932/ptr_20200204-004932_model-0_epoch13-f0.7963.h5.pkl',
             '../../results/pasa-bertptr-20200205-014243/ptr_20200205-014243_model-0_epoch13-f0.7924.h5.pkl'],
            ['../../results/pasa-bertptr-20200205-001801/detaillog_bertptr_20200205-001801.txt',
             '../../results/pasa-bertptr-20200204-004930/detaillog_bertptr_20200204-004930.txt',
             '../../results/pasa-bertptr-20200205-005949/detaillog_bertptr_20200205-005949.txt',
             '../../results/pasa-bertptr-20200204-004932/detaillog_bertptr_20200204-004932.txt',
             '../../results/pasa-bertptr-20200205-014243/detaillog_bertptr_20200205-014243.txt'])
    }
    return file[model_name]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PASA bccwj analysis')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--reset', action='store_true')
    arguments = parser.parse_args()

    model = arguments.model
    files = get_files(model)
    all_scores, dep_scores, zero_scores = {}, {}, {}
    categories = ['ブログ', '知恵袋', '出版', '新聞', '雑誌', '白書']
    for path_pkl, path_detail in zip(files[0], files[1]):
        print(path_detail)
        all_scores[path_detail], dep_scores[path_detail], zero_scores[path_detail] = main(path_pkl, path_detail)

    if arguments.reset:
        os.remove(Path('../../results/bccwj-categories.txt'))

    with Path('../../results/bccwj-categories.txt').open('a', encoding='utf-8') as f:
        for category in categories:
            for path_detail in files[1]:
                line = '{}, {}, {}, {}, {}'.format(path_detail,
                                                  category,
                                                  all_scores[path_detail][category],
                                                  ','.join(map(str, dep_scores[path_detail][category])),
                                                  ','.join(map(str, zero_scores[path_detail][category])))
                print(line)
                f.write(line + '\n')

