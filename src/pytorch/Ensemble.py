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
import argparse


tag_sl = ['sl_ntc', 'sl_bccwj', 'bertsl_ntc', 'bertsl_bccwj']
tag_sp = ['sp_global_ntc', 'sp_local_ntc', 'sp_none_ntc',
          'sp_global_bccwj', 'sp_local_bccwj', 'sp_none_bccwj',
          'bertsp_global_ntc', 'bertsp_local_ntc', 'bertsp_none_ntc',
          # 'bertsp_global_bccwj', 'bertsp_local_bccwj',
          'bertsp_none_bccwj']

parser = argparse.ArgumentParser(description='PASA Ensamble')
parser.add_argument('--mode', default=None, type=str, choices=tag_sl + tag_sp)
arguments = parser.parse_args()

mode = arguments.mode


def get_sl_ntc():
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/seq/seq_20190427-110202_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/seq/seq_20190427-110222_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/seq/seq_20190427-110231_model-0_epoch19-f0.8465.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/seq/seq_20190427-110302_model-0_epoch17-f0.8473.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/seq/seq_20190427-110328_model-0_epoch18-f0.8455.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sl_bccwj():
    with Path("../../results/pasa-lstm-20200105-035952/seq_20200105-035952_model-0_epoch19-f0.7649.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200105-040001/seq_20200105-040001_model-0_epoch19-f0.7686.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200105-040035/seq_20200105-040035_model-0_epoch17-f0.7646.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200105-040101/seq_20200105-040101_model-0_epoch10-f0.7630.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200105-040118/seq_20200105-040118_model-0_epoch17-f0.7641.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsl_ntc():
    with Path("../../results/pasa-bertsl-20200104-045510/ptr_20200104-045510_model-0_epoch16-f0.8647.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-045510/ptr_20200104-045510_model-0_epoch10-f0.8611.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-045510/ptr_20200104-045510_model-0_epoch14-f0.8620.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-045510/ptr_20200104-045510_model-0_epoch12-f0.8650.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-065438/ptr_20200104-065438_model-0_epoch12-f0.8631.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsl_bccwj():
    with Path("../../results/pasa-bertsl-20200104-152437/ptr_20200104-152437_model-0_epoch11-f0.7916.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-153949/ptr_20200104-153949_model-0_epoch13-f0.7918.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-154415/ptr_20200104-154415_model-0_epoch14-f0.7894.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-154538/ptr_20200104-154538_model-0_epoch13-f0.7916.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200104-163623/ptr_20200104-163623_model-0_epoch13-f0.7877.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sp_ntc(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "global":
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/global_argmax/ptr_20190427-114739_model-0_epoch14-f0.8461.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/global_argmax/ptr_20190427-114743_model-0_epoch19-f0.8438.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/global_argmax/ptr_20190427-114749_model-0_epoch16-f0.8455.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/global_argmax/ptr_20190427-114838_model-0_epoch17-f0.8456.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/testresult_acl2019/global_argmax/ptr_20190427-120937_model-0_epoch14-f0.8474.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../../PhD/projects/180630_oomorisan_PASAresults/pasa-pointer-20190611-215418/ptr_20190611-215418_model-0_epoch17-f0.8440.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-222931/ptr_20190611-222931_model-0_epoch15-f0.8455.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215710/ptr_20190611-215710_model-0_epoch14-f0.8476.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215849/ptr_20190611-215849_model-0_epoch16-f0.8455.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-220002/ptr_20190611-220002_model-0_epoch17-f0.8456.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "none":
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224115/ptr_20190611-224115_model-0_epoch19-f0.8436.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224102/ptr_20190611-224102_model-0_epoch14-f0.8454.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224113/ptr_20190611-224113_model-0_epoch14-f0.8468.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224120/ptr_20190611-224120_model-0_epoch16-f0.8454.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-231229/ptr_20190611-231229_model-0_epoch17-f0.8455.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sp_bccwj(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "global":
        with Path("../../results/pasa-pointer-20200104-130555/ptr_20200104-130555_model-0_epoch11-f0.7589.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-130624/ptr_20200104-130624_model-0_epoch10-f0.7621.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-130649/ptr_20200104-130649_model-0_epoch11-f0.7634.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-150504/ptr_20200104-150504_model-0_epoch10-f0.7576.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-150511/ptr_20200104-150511_model-0_epoch14-f0.7616.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-pointer-20200104-152844/ptr_20200104-152844_model-0_epoch10-f0.7619.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-165359/ptr_20200104-165359_model-0_epoch9-f0.7608.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-165505/ptr_20200104-165505_model-0_epoch15-f0.7654.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-165511/ptr_20200104-165511_model-0_epoch19-f0.7623.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-165543/ptr_20200104-165543_model-0_epoch15-f0.7618.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "none":
        with Path("../../results/pasa-pointer-20200104-150500/ptr_20200104-150500_model-0_epoch10-f0.7614.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-150547/ptr_20200104-150547_model-0_epoch10-f0.7584.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-150552/ptr_20200104-150552_model-0_epoch18-f0.7628.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-152527/ptr_20200104-152527_model-0_epoch19-f0.7619.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-152703/ptr_20200104-152703_model-0_epoch12-f0.7615.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsp_ntc(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "global":
        with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch16-f0.8703.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch15-f0.8707.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-075002/ptr_20200104-075002_model-0_epoch15-f0.8719.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch11-f0.8709.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-093240/ptr_20200104-093240_model-0_epoch15-f0.8709.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-pointer-20200104-111614/ptr_20200104-111614_model-0_epoch11-f0.8710.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-111812/ptr_20200104-111812_model-0_epoch15-f0.8705.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-130215/ptr_20200104-130215_model-0_epoch15-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-130233/ptr_20200104-130233_model-0_epoch13-f0.8701.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-130251/ptr_20200104-130251_model-0_epoch15-f0.8707.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "none":
        with Path("../../results/pasa-pointer-20200104-200900/ptr_20200104-200900_model-0_epoch11-f0.8702.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-093250/ptr_20200104-093250_model-0_epoch15-f0.8703.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-093509/ptr_20200104-093509_model-0_epoch19-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-111557/ptr_20200104-111557_model-0_epoch13-f0.8698.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-111556/ptr_20200104-111556_model-0_epoch12-f0.8706.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsp_bccwj(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    # if mode == "global":
    #     with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch16-f0.8703.h5.pkl").open(
    #         'rb') as f:
    #         items0 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch15-f0.8707.h5.pkl").open(
    #         'rb') as f:
    #         items1 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-075002/ptr_20200104-075002_model-0_epoch15-f0.8719.h5.pkl").open(
    #         'rb') as f:
    #         items2 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-075003/ptr_20200104-075003_model-0_epoch11-f0.8709.h5.pkl").open(
    #         'rb') as f:
    #         items3 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-093240/ptr_20200104-093240_model-0_epoch15-f0.8709.h5.pkl").open(
    #         'rb') as f:
    #         items4 = pickle.load(f)
    # elif mode == "local":
    #     with Path("../../results/pasa-pointer-20200104-111614/ptr_20200104-111614_model-0_epoch11-f0.8710.h5.pkl").open('rb') as f:
    #         items0 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-111812/ptr_20200104-111812_model-0_epoch15-f0.8705.h5.pkl").open('rb') as f:
    #         items1 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-130215/ptr_20200104-130215_model-0_epoch15-f0.8718.h5.pkl").open('rb') as f:
    #         items2 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-130233/ptr_20200104-130233_model-0_epoch13-f0.8701.h5.pkl").open('rb') as f:
    #         items3 = pickle.load(f)
    #     with Path("../../results/pasa-pointer-20200104-130251/ptr_20200104-130251_model-0_epoch15-f0.8707.h5.pkl").open('rb') as f:
    #         items4 = pickle.load(f)
    if mode == "none":
        with Path("../../results/pasa-pointer-20200104-165547/ptr_20200104-165547_model-0_epoch16-f0.7924.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-170518/ptr_20200104-170518_model-0_epoch17-f0.7922.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-171529/ptr_20200104-171529_model-0_epoch12-f0.7936.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-172439/ptr_20200104-172439_model-0_epoch16-f0.7945.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200104-172634/ptr_20200104-172634_model-0_epoch14-f0.7937.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4

if mode in tag_sl:
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == 'sl_ntc':
        items0, items1, items2, items3, items4 = get_sl_ntc()
    elif mode == 'sl_bccwj':
        items0, items1, items2, items3, items4 = get_sl_bccwj()
    elif mode == 'bertsl_ntc':
        items0, items1, items2, items3, items4 = get_bertsl_ntc()
    elif mode == 'bertsl_bccwj':
        items0, items1, items2, items3, items4 = get_bertsl_bccwj()

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

elif mode in tag_sp:
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == 'sp_global_ntc':
        item0, item1, item2, item3, item4 = get_sp_ntc(mode='global')
    elif mode == 'sp_local_ntc':
        item0, item1, item2, item3, item4 = get_sp_ntc(mode='local')
    elif mode == 'sp_none_ntc':
        item0, item1, item2, item3, item4 = get_sp_ntc(mode='none')
    elif mode == 'sp_global_bccwj':
        item0, item1, item2, item3, item4 = get_sp_bccwj(mode='global')
    elif mode == 'sp_local_bccwj':
        item0, item1, item2, item3, item4 = get_sp_bccwj(mode='local')
    elif mode == 'sp_none_bccwj':
        item0, item1, item2, item3, item4 = get_sp_bccwj(mode='none')
    elif mode == 'bertsp_global_ntc':
        item0, item1, item2, item3, item4 = get_bertsp_ntc(mode='global')
    elif mode == 'bertsp_local_ntc':
        item0, item1, item2, item3, item4 = get_bertsp_ntc(mode='local')
    elif mode == 'bertsp_none_ntc':
        item0, item1, item2, item3, item4 = get_bertsp_ntc(mode='none')
    elif mode == 'bertsp_global_bccwj':
        item0, item1, item2, item3, item4 = get_bertsp_bccwj(mode='global')
    elif mode == 'bertsp_local_bccwj':
        item0, item1, item2, item3, item4 = get_bertsp_bccwj(mode='local')
    elif mode == 'bertsp_none_bccwj':
        item0, item1, item2, item3, item4 = get_bertsp_bccwj(mode='none')

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
    if 'global' in mode:
        print('[global_argmax]')
    elif 'local' in mode:
        print('[local_argmax]')
    elif 'none' in mode:
        print('[no_restricted]')
    print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp,
                                                                       num_fn))
    print(', '.join(map(str, f1s)))


'''
* ntc
[sl]
All: 0.856, Dep: 0.9160205976377802, Zero: 0.5646315398197193 / tp: [13050  9039  1837  2703   325    10], fp: [1010  319  416 1686  247   32], fn: [1250  550  842 2061  458  201]
0.9203102961918195, 0.9541352192959308, 0.7449310624493106, 0.5906260242543427, 0.4797047970479705, 0.07905138339920949

[sp global_argmax]
All: 0.8582670943885211, Dep: 0.9177175343876987, Zero: 0.5714285714285715 / tp: [13108  9058  1886  2728   358    18], fp: [1108  292  397 1690  282   30], fn: [1192  531  793 2036  425  193]
0.9193435264412961, 0.9565446961296794, 0.7601773478436115, 0.5942060553256371, 0.5031623330990864, 0.13899613899613902

[sp local_argmax]
All: 0.8579037473844398, Dep: 0.9177940839257052, Zero: 0.5664062499999999 / tp: [13084  9069  1862  2677   352    16], fp: [1063  305  381 1625  291   33], fn: [1216  520  817 2087  431  195]
0.9198861039828453, 0.9564942255972156, 0.756603006907761, 0.5905581292742113, 0.49368863955119213, 0.12307692307692308

[sp no_restricted]
All: 0.8582670943885211, Dep: 0.9177175343876987, Zero: 0.5714285714285715 / tp: [13108  9058  1886  2728   358    18], fp: [1108  292  397 1690  282   30], fn: [1192  531  793 2036  425  193]
0.9193435264412961, 0.9565446961296794, 0.7601773478436115, 0.5942060553256371, 0.5031623330990864, 0.13899613899613902



'''