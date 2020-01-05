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
        with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch16-f0.8703.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch15-f0.8707.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-075002/ptr_20200104-075002_model-0_epoch15-f0.8719.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch11-f0.8709.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-093240/ptr_20200104-093240_model-0_epoch15-f0.8709.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-bertptr-20200104-111614/ptr_20200104-111614_model-0_epoch11-f0.8710.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-111812/ptr_20200104-111812_model-0_epoch15-f0.8705.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-130215/ptr_20200104-130215_model-0_epoch15-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-130233/ptr_20200104-130233_model-0_epoch13-f0.8701.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-130251/ptr_20200104-130251_model-0_epoch15-f0.8707.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "none":
        with Path("../../results/pasa-bertptr-20200104-200900/ptr_20200104-200900_model-0_epoch11-f0.8702.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-093250/ptr_20200104-093250_model-0_epoch15-f0.8703.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-093509/ptr_20200104-093509_model-0_epoch19-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-111557/ptr_20200104-111557_model-0_epoch13-f0.8698.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-111556/ptr_20200104-111556_model-0_epoch15-f0.8706.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsp_bccwj(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    # if mode == "global":
    #     with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch16-f0.8703.h5.pkl").open(
    #         'rb') as f:
    #         items0 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch15-f0.8707.h5.pkl").open(
    #         'rb') as f:
    #         items1 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-075002/ptr_20200104-075002_model-0_epoch15-f0.8719.h5.pkl").open(
    #         'rb') as f:
    #         items2 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-075003/ptr_20200104-075003_model-0_epoch11-f0.8709.h5.pkl").open(
    #         'rb') as f:
    #         items3 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-093240/ptr_20200104-093240_model-0_epoch15-f0.8709.h5.pkl").open(
    #         'rb') as f:
    #         items4 = pickle.load(f)
    # elif mode == "local":
    #     with Path("../../results/pasa-bertptr-20200104-111614/ptr_20200104-111614_model-0_epoch11-f0.8710.h5.pkl").open('rb') as f:
    #         items0 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-111812/ptr_20200104-111812_model-0_epoch15-f0.8705.h5.pkl").open('rb') as f:
    #         items1 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-130215/ptr_20200104-130215_model-0_epoch15-f0.8718.h5.pkl").open('rb') as f:
    #         items2 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-130233/ptr_20200104-130233_model-0_epoch13-f0.8701.h5.pkl").open('rb') as f:
    #         items3 = pickle.load(f)
    #     with Path("../../results/pasa-bertptr-20200104-130251/ptr_20200104-130251_model-0_epoch15-f0.8707.h5.pkl").open('rb') as f:
    #         items4 = pickle.load(f)
    if mode == "none":
        with Path("../../results/pasa-bertptr-20200104-165547/ptr_20200104-165547_model-0_epoch16-f0.7924.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-170518/ptr_20200104-170518_model-0_epoch17-f0.7922.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-171529/ptr_20200104-171529_model-0_epoch12-f0.7936.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-172439/ptr_20200104-172439_model-0_epoch16-f0.7945.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200104-172634/ptr_20200104-172634_model-0_epoch14-f0.7937.h5.pkl").open('rb') as f:
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
        items0, items1, items2, items3, items4 = get_sp_ntc(mode='global')
    elif mode == 'sp_local_ntc':
        items0, items1, items2, items3, items4 = get_sp_ntc(mode='local')
    elif mode == 'sp_none_ntc':
        items0, items1, items2, items3, items4 = get_sp_ntc(mode='none')
    elif mode == 'sp_global_bccwj':
        items0, items1, items2, items3, items4 = get_sp_bccwj(mode='global')
    elif mode == 'sp_local_bccwj':
        items0, items1, items2, items3, items4 = get_sp_bccwj(mode='local')
    elif mode == 'sp_none_bccwj':
        items0, items1, items2, items3, items4 = get_sp_bccwj(mode='none')
    elif mode == 'bertsp_global_ntc':
        items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='global')
    elif mode == 'bertsp_local_ntc':
        items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='local')
    elif mode == 'bertsp_none_ntc':
        items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='none')
    elif mode == 'bertsp_global_bccwj':
        items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='global')
    elif mode == 'bertsp_local_bccwj':
        items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='local')
    elif mode == 'bertsp_none_bccwj':
        items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='none')

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

[bertsl]
All: 0.8746221662468513, Dep: 0.9261248268402376, Zero: 0.623856601681604 / tp: [13320  9157  1925  2938   409    29], fp: [ 897  244  586 1409  227   53], fn: [ 980  432  754 1826  374  182]
0.934179612161167, 0.9644023170089521, 0.7418111753371869, 0.6449346943255405, 0.5764622973925301, 0.19795221843003416

[bertsp global_argmax]
All: 0.8788624038340269, Dep: 0.9292825462090496, Zero: 0.6358381502890174 / tp: [13320  9154  1935  3022   415    28], fp: [ 869  247  440 1395  245   36], fn: [ 980  435  744 1742  368  183]
0.9350977570290288, 0.9640863612427594, 0.7657301147605857, 0.6583160875721599, 0.5751905751905751, 0.20363636363636364

[bertsp local_argmax]
All: 0.8793070806512685, Dep: 0.9295763921941932, Zero: 0.6361543420204439 / tp: [13324  9143  1946  3012   413    29], fp: [ 870  238  436 1380  233   34], fn: [ 976  446  733 1752  370  182]
0.9352144311083035, 0.9639430680021086, 0.7690179806362378, 0.6579292267365662, 0.5780265920223933, 0.21167883211678834

[bertsp no_restricted]
All: 0.8794243607507243, Dep: 0.9296412598724902, Zero: 0.6388255676119267 / tp: [13357  9144  1923  3057   421    25], fp: [ 909  242  402 1433  241   32], fn: [ 943  445  756 1707  362  186]
0.9351676818595533, 0.963794466403162, 0.7685851318944844, 0.66068727036957, 0.5826989619377162, 0.18656716417910446


* bccwj
[sl]
All: 0.7836949375410914, Dep: 0.8370541611624835, Zero: 0.5241002570694088 / tp: [6505 4388 1780 1359  231   41], fp: [910 752 512 695 222  96], fn: [1432  863  465 1330  401  218]
0.8474465867639397, 0.8445770378211913, 0.7846594666078908, 0.573055028462998, 0.42580645161290326, 0.2070707070707071

[sp global_argmax]
All: 0.7801530459410148, Dep: 0.8359683794466404, Zero: 0.5137556987894984 / tp: [6603 4372 1715 1374  231   29], fp: [1051  758  428  824  262   61], fn: [1334  879  530 1315  401  230]
0.8470271310371368, 0.8423080628070514, 0.7816773017319965, 0.5623081645181094, 0.4106666666666667, 0.166189111747851

[sp local_argmax]
All: 0.7833669761808639, Dep: 0.8386632188799058, Zero: 0.5275590551181102 / tp: [6684 4392 1735 1460  243   39], fp: [1087  751  469  947  270   65], fn: [1253  859  510 1229  389  220]
0.851031321619557, 0.8451029440061574, 0.7799505506855472, 0.5729984301412873, 0.42445414847161567, 0.21487603305785122

[sp no_restricted]
All: 0.781899627155924, Dep: 0.8384985437052066, Zero: 0.5246876859012493 / tp: [6704 4384 1723 1497  229   38], fp: [1134  740  439 1048  254   78], fn: [1233  867  522 1192  403  221]
0.8499524564183836, 0.8451084337349398, 0.7819378261856138, 0.5720290408865112, 0.410762331838565, 0.20266666666666666

[bertsl]
All: 0.8118676510801204, Dep: 0.8576707357307819, Zero: 0.5900958466453674 / tp: [6784 4438 1776 1515  288   44], fp: [898 616 393 630 178  45], fn: [1140  804  463 1156  342  215]
0.8694092015891324, 0.8620823620823622, 0.8058076225045372, 0.6291528239202657, 0.5255474452554746, 0.25287356321839083

[bertsp no_restricted]
All: 0.8098063961762574, Dep: 0.8576651867328283, Zero: 0.5879860795884401 / tp: [6852 4525 1759 1580  324   39], fp: [973 736 382 808 246  52], fn: [1072  717  480 1091  306  220]
0.8701504857451267, 0.8616585737408359, 0.8031963470319634, 0.6246293733939514, 0.54, 0.2228571428571429



'''