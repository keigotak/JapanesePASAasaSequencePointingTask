from pathlib import Path
import pickle
import numpy as np
import torch

import sys
sys.path.append('../')
import os
sys.path.append(os.pardir)
from Decoders import get_restricted_prediction
from utils.HelperFunctions import concat_labels
from Validation import get_pr_numbers, get_f_score

seq = False
global_argmax = False
local_argmax = False
no_restricted = True

if seq:
    item0 = []
    item1 = []
    item2 = []
    item3 = []
    item4 = []
    with Path("../../results/testresult_acl2019/seq/seq_20190427-110202_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/testresult_acl2019/seq/seq_20190427-110222_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/testresult_acl2019/seq/seq_20190427-110231_model-0_epoch19-f0.8465.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/testresult_acl2019/seq/seq_20190427-110302_model-0_epoch17-f0.8473.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/testresult_acl2019/seq/seq_20190427-110328_model-0_epoch18-f0.8455.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)

    num_tp = np.array([0] * 6)
    num_fp = np.array([0] * 6)
    num_fn = np.array([0] * 6)

    for i0, i1, i2, i3, i4 in zip(items0, items1, items2, items3, items4):
        sum_predication = np.array(i0[0]) + np.array(i1[0]) + np.array(i2[0]) + np.array(i3[0]) + np.array(i4[0])
        t_props = [i0[1]]
        t_labels = [i0[2]]
        sum_predication = torch.Tensor([sum_predication.tolist()])
        _, prediction = torch.max(sum_predication, 2)
        prediction = prediction.tolist()

        one_tp, one_fp, one_fn = get_pr_numbers(prediction, t_labels, t_props)
        num_tp = num_tp + one_tp
        num_fp = num_fp + one_fp
        num_fn = num_fn + one_fn
    all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)

    precisions = []
    recalls = []
    f1s = []
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
    print('[Seq]')
    print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp, num_fn))
    print(', '.join(map(str, f1s)))

if global_argmax or local_argmax or no_restricted:
    item0 = []
    item1 = []
    item2 = []
    item3 = []
    item4 = []
    if global_argmax:
        with Path("../../results/testresult_acl2019/global_argmax/ptr_20190427-114739_model-0_epoch14-f0.8461.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/testresult_acl2019/global_argmax/ptr_20190427-114743_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/testresult_acl2019/global_argmax/ptr_20190427-114749_model-0_epoch16-f0.8455.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/testresult_acl2019/global_argmax/ptr_20190427-114838_model-0_epoch17-f0.8456.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/testresult_acl2019/global_argmax/ptr_20190427-120937_model-0_epoch14-f0.8474.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    if local_argmax:
        with Path("../../results/pasa-pointer-20190611-215418/ptr_20190611-215418_model-0_epoch17-f0.8440.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-222931/ptr_20190611-222931_model-0_epoch15-f0.8455.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-215710/ptr_20190611-215710_model-0_epoch14-f0.8476.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-215849/ptr_20190611-215849_model-0_epoch16-f0.8455.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-220002/ptr_20190611-220002_model-0_epoch17-f0.8456.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    if no_restricted:
        with Path("../../results/pasa-pointer-20190611-224115/ptr_20190611-224115_model-0_epoch19-f0.8436.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-224102/ptr_20190611-224102_model-0_epoch14-f0.8454.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-224113/ptr_20190611-224113_model-0_epoch14-f0.8468.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-224120/ptr_20190611-224120_model-0_epoch16-f0.8454.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20190611-231229/ptr_20190611-231229_model-0_epoch17-f0.8455.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)


    num_tp = np.array([0] * 6)
    num_fp = np.array([0] * 6)
    num_fn = np.array([0] * 6)

    for i0, i1, i2, i3, i4 in zip(items0, items1, items2, items3, items4):
        sum_ga = np.array(i0[0]) + np.array(i1[0]) + np.array(i2[0]) + np.array(i3[0]) + np.array(i4[0])
        sum_ni = np.array(i0[1]) + np.array(i1[1]) + np.array(i2[1]) + np.array(i3[1]) + np.array(i4[1])
        sum_wo = np.array(i0[2]) + np.array(i1[2]) + np.array(i2[2]) + np.array(i3[2]) + np.array(i4[2])
        t_props = [i0[3]]
        t_labels = [i0[4]]

        sum_ga = torch.Tensor([sum_ga.tolist()])
        sum_ni = torch.Tensor([sum_ni.tolist()])
        sum_wo = torch.Tensor([sum_wo.tolist()])

        ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(sum_ga, sum_ni, sum_wo)
        s_size = sum_ga.shape[1]
        ga_prediction = np.identity(s_size)[ga_prediction].astype(np.int64)
        ni_prediction = np.identity(s_size)[ni_prediction].astype(np.int64)
        wo_prediction = np.identity(s_size)[wo_prediction].astype(np.int64)
        t_prediction = concat_labels(ga_prediction, ni_prediction, wo_prediction)

        one_tp, one_fp, one_fn = get_pr_numbers(t_prediction, t_labels, t_props)
        num_tp = num_tp + one_tp
        num_fp = num_fp + one_fp
        num_fn = num_fn + one_fn
    all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)

    precisions = []
    recalls = []
    f1s = []
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
    if global_argmax:
        print('[global_argmax]')
    if local_argmax:
        print('[local_argmax]')
    if no_restricted:
        print('[no_restricted]')
    print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp,
                                                                       num_fn))
    print(', '.join(map(str, f1s)))


'''
[Seq]
All: 0.856, Dep: 0.9160205976377802, Zero: 0.5646315398197193 / tp: [13050  9039  1837  2703   325    10], fp: [1010  319  416 1686  247   32], fn: [1250  550  842 2061  458  201]
0.9203102961918195, 0.9541352192959308, 0.7449310624493106, 0.5906260242543427, 0.4797047970479705, 0.07905138339920949

[global_argmax]
All: 0.8582670943885211, Dep: 0.9177175343876987, Zero: 0.5714285714285715 / tp: [13108  9058  1886  2728   358    18], fp: [1108  292  397 1690  282   30], fn: [1192  531  793 2036  425  193]
0.9193435264412961, 0.9565446961296794, 0.7601773478436115, 0.5942060553256371, 0.5031623330990864, 0.13899613899613902

[local_argmax]
All: 0.8579037473844398, Dep: 0.9177940839257052, Zero: 0.5664062499999999 / tp: [13084  9069  1862  2677   352    16], fp: [1063  305  381 1625  291   33], fn: [1216  520  817 2087  431  195]
0.9198861039828453, 0.9564942255972156, 0.756603006907761, 0.5905581292742113, 0.49368863955119213, 0.12307692307692308

[no_restricted]
All: 0.8582670943885211, Dep: 0.9177175343876987, Zero: 0.5714285714285715 / tp: [13108  9058  1886  2728   358    18], fp: [1108  292  397 1690  282   30], fn: [1192  531  793 2036  425  193]
0.9193435264412961, 0.9565446961296794, 0.7601773478436115, 0.5942060553256371, 0.5031623330990864, 0.13899613899613902

'''