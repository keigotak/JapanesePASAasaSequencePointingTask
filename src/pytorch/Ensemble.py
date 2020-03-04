from pathlib import Path
import pickle
import numpy as np
import torch
from decimal import Decimal, ROUND_HALF_UP

import sys
sys.path.append('../')
import os
sys.path.append(os.pardir)
from Decoders import get_restricted_prediction, get_no_decode_prediction, get_ordered_prediction
from utils.HelperFunctions import concat_labels
from Validation import get_pr_numbers, get_f_score
import argparse
import torch.nn.functional as F


def get_sl_ntc():
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110202/seq_20190427-110202_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110222/seq_20190427-110222_model-0_epoch19-f0.8438.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110231/seq_20190427-110231_model-0_epoch19-f0.8465.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110302/seq_20190427-110302_model-0_epoch17-f0.8473.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110328/seq_20190427-110328_model-0_epoch18-f0.8455.h5.pkl").open('rb') as f:
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
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114739/ptr_20190427-114739_model-0_epoch14-f0.8461.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114743/ptr_20190427-114743_model-0_epoch19-f0.8438.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114749/ptr_20190427-114749_model-0_epoch16-f0.8455.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114838/ptr_20190427-114838_model-0_epoch17-f0.8456.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-120937/ptr_20190427-120937_model-0_epoch14-f0.8474.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215418/ptr_20190611-215418_model-0_epoch17-f0.8440.h5.pkl").open('rb') as f:
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
    if mode == "global":
        with Path("../../results/pasa-bertptr-20200204-225742/ptr_20200204-225742_model-0_epoch16-f0.7939.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-233732/ptr_20200204-233732_model-0_epoch16-f0.7955.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-000531/ptr_20200204-000531_model-0_epoch14-f0.7943.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-000527/ptr_20200204-000527_model-0_epoch14-f0.7950.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-000523/ptr_20200204-000523_model-0_epoch13-f0.7943.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-bertptr-20200205-001801/ptr_20200205-001801_model-0_epoch14-f0.7955.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-004930/ptr_20200204-004930_model-0_epoch17-f0.7940.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200205-005949/ptr_20200205-005949_model-0_epoch14-f0.7944.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200204-004932/ptr_20200204-004932_model-0_epoch13-f0.7963.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200205-014243/ptr_20200205-014243_model-0_epoch13-f0.7924.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "none":
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


def main(mode, corpus, with_softmax=False):
    mode = '{}_{}'.format(mode, corpus)
    tag_sl = ['sl_ntc', 'sl_bccwj', 'bertsl_ntc', 'bertsl_bccwj']
    tag_sp = ['spg_ntc', 'spl_ntc', 'spn_ntc',
              'spg_bccwj', 'spl_bccwj', 'spn_bccwj',
              'bertspg_ntc', 'bertspl_ntc', 'bertspn_ntc',
              'bertspg_bccwj', 'bertspl_bccwj',
              'bertspn_bccwj']

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

            sum_predication /= 5

            if with_softmax:
                sum_predication = F.softmax(torch.Tensor([sum_predication.tolist()]), dim=2)
            else:
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
        print('[{}]'.format(mode))
        print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp, num_fn))
        print(', '.join(map(str, f1s)))
        print('{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(Decimal(str(all_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(dep_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[0])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[1])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[2])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(zero_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[3])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[4])).quantize(Decimal('0.0001'),
                                                                                    rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[5])).quantize(Decimal('0.0001'),
                                                                                    rounding=ROUND_HALF_UP)
                                                      ))

    elif mode in tag_sp:
        items0, items1, items2, items3, items4 = [], [], [], [], []
        if mode == 'spg_ntc':
            items0, items1, items2, items3, items4 = get_sp_ntc(mode='global')
        elif mode == 'spl_ntc':
            items0, items1, items2, items3, items4 = get_sp_ntc(mode='local')
        elif mode == 'spn_ntc':
            items0, items1, items2, items3, items4 = get_sp_ntc(mode='none')
        elif mode == 'spg_bccwj':
            items0, items1, items2, items3, items4 = get_sp_bccwj(mode='global')
        elif mode == 'spl_bccwj':
            items0, items1, items2, items3, items4 = get_sp_bccwj(mode='local')
        elif mode == 'spn_bccwj':
            items0, items1, items2, items3, items4 = get_sp_bccwj(mode='none')
        elif mode == 'bertspg_ntc':
            items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='global')
        elif mode == 'bertspl_ntc':
            items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='local')
        elif mode == 'bertspn_ntc':
            items0, items1, items2, items3, items4 = get_bertsp_ntc(mode='none')
        elif mode == 'bertspg_bccwj':
            items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='global')
        elif mode == 'bertspl_bccwj':
            items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='local')
        elif mode == 'bertspn_bccwj':
            items0, items1, items2, items3, items4 = get_bertsp_bccwj(mode='none')

        num_tp = np.array([0] * 6)
        num_fp = np.array([0] * 6)
        num_fn = np.array([0] * 6)

        for i0, i1, i2, i3, i4 in zip(items0, items1, items2, items3, items4):
            sum_ga = np.array(i0[0]) + np.array(i1[0]) + np.array(i2[0]) + np.array(i3[0]) + np.array(i4[0])
            sum_ni = np.array(i0[1]) + np.array(i1[1]) + np.array(i2[1]) + np.array(i3[1]) + np.array(i4[1])
            sum_wo = np.array(i0[2]) + np.array(i1[2]) + np.array(i2[2]) + np.array(i3[2]) + np.array(i4[2])

            sum_ga /= 5
            sum_ni /= 5
            sum_wo /= 5

            t_props = [i0[3]]
            t_labels = [i0[4]]

            if with_softmax:
                sum_ga = F.softmax(torch.Tensor([sum_ga.tolist()]), dim=1)
                sum_ni = F.softmax(torch.Tensor([sum_ni.tolist()]), dim=1)
                sum_wo = F.softmax(torch.Tensor([sum_wo.tolist()]), dim=1)
            else:
                sum_ga = torch.Tensor([sum_ga.tolist()])
                sum_ni = torch.Tensor([sum_ni.tolist()])
                sum_wo = torch.Tensor([sum_wo.tolist()])


            if 'global' in mode:
                ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(sum_ga, sum_ni, sum_wo)
            elif 'local' in mode:
                ga_prediction, ni_prediction, wo_prediction = get_ordered_prediction(sum_ga, sum_ni, sum_wo)
            else:
                ga_prediction, ni_prediction, wo_prediction = get_no_decode_prediction(sum_ga, sum_ni, sum_wo)
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
        print('[{}]'.format(mode))
        print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp,
                                                                           num_fn))
        print(', '.join(map(str, f1s)))
        print('{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(Decimal(str(all_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(dep_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[0])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[1])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[2])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(zero_score)).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[3])).quantize(Decimal('0.0001'),
                                                                                       rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[4])).quantize(Decimal('0.0001'),
                                                                                    rounding=ROUND_HALF_UP),
                                                      Decimal(str(f1s[5])).quantize(Decimal('0.0001'),
                                                                                    rounding=ROUND_HALF_UP)
                                                      ))


if __name__ == "__main__":
    tags = ['sl', 'spg', 'spl', 'spn', 'bertsl', 'bertspg', 'bertspl', 'bertspn']
    parser = argparse.ArgumentParser(description='PASA Ensamble')
    parser.add_argument('--mode', default=None, type=str, choices=tags + ['all'])
    parser.add_argument('--corpus', default=None, type=str, choices=['ntc', 'bccwj', 'all'])
    parser.add_argument('--with_softmax', action='store_true')
    arguments = parser.parse_args()
    if arguments.mode == "all":
        for mode in tags:
            if arguments.corpus == "all":
                for corpus in ['ntc', 'bccwj']:
                    main(mode=mode, corpus=corpus, with_softmax=arguments.with_softmax)
            else:
                main(mode=mode, corpus=arguments.corpus, with_softmax=arguments.with_softmax)
    else:
        main(mode=arguments.mode, corpus=arguments.corpus, with_softmax=arguments.with_softmax)


'''
* ntc
[sl]
All: 0.856, Dep: 0.9160205976377802, Zero: 0.5646315398197193 / tp: [13050  9039  1837  2703   325    10], fp: [1010  319  416 1686  247   32], fn: [1250  550  842 2061  458  201]
0.9203102961918195, 0.9541352192959308, 0.7449310624493106, 0.5906260242543427, 0.4797047970479705, 0.07905138339920949
0.8560, 0.9160, 0.9203, 0.9541, 0.7449, 0.5646, 0.5906, 0.4797, 0.0791

[sp global_argmax]
All: 0.8582670943885211, Dep: 0.9177175343876987, Zero: 0.5714285714285715 / tp: [13108  9058  1886  2728   358    18], fp: [1108  292  397 1690  282   30], fn: [1192  531  793 2036  425  193]
0.9193435264412961, 0.9565446961296794, 0.7601773478436115, 0.5942060553256371, 0.5031623330990864, 0.13899613899613902
0.8583, 0.9177, 0.9193, 0.9565, 0.7602, 0.5714, 0.5942, 0.5032, 0.1390

[sp local_argmax]
All: 0.8581165691822404, Dep: 0.9180058471710011, Zero: 0.5666728659103588 / tp: [13082  9079  1860  2679   352    16], fp: [1055  309  380 1623  293   33], fn: [1218  510  819 2085  431  195]
0.9200689242887787, 0.9568424935448174, 0.7562512705834519, 0.5909993381866314, 0.49299719887955185, 0.12307692307692308
0.8581, 0.9180, 0.9201, 0.9568, 0.7563, 0.5667, 0.5910, 0.4930, 0.1231

[sp no_restricted]
All: 0.8576570887275023, Dep: 0.9169906242853877, Zero: 0.571060382916053 / tp: [13110  9064  1886  2726   358    18], fp: [1121  296  402 1690  284   30], fn: [1201  541  795 2038  425  193]
0.9186462055917595, 0.9558660690746111, 0.7591064600523244, 0.5938997821350763, 0.5024561403508772, 0.13899613899613902
0.8577, 0.9170, 0.9186, 0.9559, 0.7591, 0.5711, 0.5939, 0.5025, 0.1390

[bertsl]
All: 0.8746221662468513, Dep: 0.9261248268402376, Zero: 0.623856601681604 / tp: [13320  9157  1925  2938   409    29], fp: [ 897  244  586 1409  227   53], fn: [ 980  432  754 1826  374  182]
0.934179612161167, 0.9644023170089521, 0.7418111753371869, 0.6449346943255405, 0.5764622973925301, 0.19795221843003416
0.8746, 0.9261, 0.9342, 0.9644, 0.7418, 0.6239, 0.6449, 0.5765, 0.1980

[bertsp global_argmax]
All: 0.8788624038340269, Dep: 0.9292825462090496, Zero: 0.6358381502890174 / tp: [13320  9154  1935  3022   415    28], fp: [ 869  247  440 1395  245   36], fn: [ 980  435  744 1742  368  183]
0.9350977570290288, 0.9640863612427594, 0.7657301147605857, 0.6583160875721599, 0.5751905751905751, 0.20363636363636364
0.8789, 0.9293, 0.9351, 0.9641, 0.7657, 0.6358, 0.6583, 0.5752, 0.2036

[bertsp local_argmax]
All: 0.8791354421393073, Dep: 0.9294063892167689, Zero: 0.6359701629984345 / tp: [13318  9146  1945  3012   413    28], fp: [ 870  245  434 1380  234   34], fn: [ 982  443  734 1752  370  183]
0.9349901713001966, 0.9637513171759748, 0.7690786872281534, 0.6579292267365662, 0.5776223776223777, 0.20512820512820515
0.8791, 0.9294, 0.9350, 0.9638, 0.7691, 0.6360, 0.6579, 0.5776, 0.2051

[bertsp no_restricted]
All: 0.8789618560755013, Dep: 0.9290724863600934, Zero: 0.6387167335034634 / tp: [13362  9148  1926  3057   422    25], fp: [ 918  247  405 1432  241   32], fn: [ 950  454  757 1709  363  187]
0.9346670397313934, 0.9630994367531717, 0.7682489030714001, 0.6606158833063209, 0.5828729281767956, 0.1858736059479554
0.8790, 0.9291, 0.9347, 0.9631, 0.7682, 0.6387, 0.6606, 0.5829, 0.1859


* bccwj
[sl]
All: 0.7836949375410914, Dep: 0.8370541611624835, Zero: 0.5241002570694088 / tp: [6505 4388 1780 1359  231   41], fp: [910 752 512 695 222  96], fn: [1432  863  465 1330  401  218]
0.8474465867639397, 0.8445770378211913, 0.7846594666078908, 0.573055028462998, 0.42580645161290326, 0.2070707070707071
0.7837, 0.8371, 0.8474, 0.8446, 0.7847, 0.5241, 0.5731, 0.4258, 0.2071

[sp global_argmax]
All: 0.7801530459410148, Dep: 0.8359683794466404, Zero: 0.5137556987894984 / tp: [6603 4372 1715 1374  231   29], fp: [1051  758  428  824  262   61], fn: [1334  879  530 1315  401  230]
0.8470271310371368, 0.8423080628070514, 0.7816773017319965, 0.5623081645181094, 0.4106666666666667, 0.166189111747851
0.7802, 0.8360, 0.8470, 0.8423, 0.7817, 0.5138, 0.5623, 0.4107, 0.1662

[sp local_argmax]
All: 0.7832920658843795, Dep: 0.8386632188799058, Zero: 0.5271763815291446 / tp: [6680 4399 1732 1459  243   39], fp: [1083  757  467  948  271   65], fn: [1257  852  513 1230  389  220]
0.8509554140127389, 0.8453925242625155, 0.7794779477947795, 0.5726059654631084, 0.4240837696335079, 0.21487603305785122
0.7833, 0.8387, 0.8510, 0.8454, 0.7795, 0.5272, 0.5726, 0.4241, 0.2149

[sp no_restricted]
All: 0.7810074407151651, Dep: 0.8372776236330994, Zero: 0.5247510034190576 / tp: [6711 4388 1726 1497  230   38], fp: [1148  749  444 1046  255   78], fn: [1244  870  530 1194  403  221]
0.8487416213481724, 0.8442520442520443, 0.7799367374604609, 0.5720290408865113, 0.41144901610017887, 0.20266666666666666
0.7810, 0.8373, 0.8487, 0.8443, 0.7799, 0.5248, 0.5720, 0.4114, 0.2027

[bertsl]
All: 0.8118676510801204, Dep: 0.8576707357307819, Zero: 0.5900958466453674 / tp: [6784 4438 1776 1515  288   44], fp: [898 616 393 630 178  45], fn: [1140  804  463 1156  342  215]
0.8694092015891324, 0.8620823620823622, 0.8058076225045372, 0.6291528239202657, 0.5255474452554746, 0.25287356321839083
0.8119, 0.8577, 0.8694, 0.8621, 0.8058, 0.5901, 0.6292, 0.5255, 0.2529

[bertsp global_argmax]
All: 0.8096386833923019, Dep: 0.8578234622287486, Zero: 0.5880234128770825 / tp: [6859 4515 1770 1608  316   35], fp: [1017  709  370  864  237   43], fn: [1065  727  469 1063  314  224]
0.8682278481012659, 0.8627938085228358, 0.808403745147294, 0.6253159634454599, 0.5342349957734573, 0.20771513353115728
0.8096, 0.8578, 0.8682, 0.8628, 0.8084, 0.5880, 0.6253, 0.5342, 0.2077

[bertsp local_argmax]
All: 0.8084569772745566, Dep: 0.85651776384535, Zero: 0.587031743643749 / tp: [6859 4506 1750 1600  316   35], fp: [1012  722  370  842  257   37], fn: [1065  736  489 1071  314  224]
0.8685026907249129, 0.8607449856733524, 0.8029364533149804, 0.6258556620379425, 0.5253532834580216, 0.21148036253776437
0.8085, 0.8565, 0.8685, 0.8607, 0.8029, 0.5870, 0.6259, 0.5254, 0.2115

[bertsp no_restricted]
All: 0.8092135253264687, Dep: 0.8570776553432875, Zero: 0.5873999093518658 / tp: [6855 4530 1760 1581  324   39], fp: [977 744 382 811 248  52], fn: [1078  722  481 1092  307  221]
0.8696479543292103, 0.8607258217746533, 0.8031028975587499, 0.624284304047384, 0.5386533665835412, 0.2222222222222222
0.8092, 0.8571, 0.8696, 0.8607, 0.8031, 0.5874, 0.6243, 0.5387, 0.2222


with_softmax
[sl_ntc]
All: 0.856, Dep: 0.9160205976377802, Zero: 0.5646315398197193 / tp: [13050  9039  1837  2703   325    10], fp: [1010  319  416 1686  247   32], fn: [1250  550  842 2061  458  201]
0.9203102961918195, 0.9541352192959308, 0.7449310624493106, 0.5906260242543427, 0.4797047970479705, 0.07905138339920949
0.8560, 0.9160, 0.9203, 0.9541, 0.7449, 0.5646, 0.5906, 0.4797, 0.0791
[bertsl_ntc]
All: 0.8746221662468513, Dep: 0.9261248268402376, Zero: 0.623856601681604 / tp: [13320  9157  1925  2938   409    29], fp: [ 897  244  586 1409  227   53], fn: [ 980  432  754 1826  374  182]
0.934179612161167, 0.9644023170089521, 0.7418111753371869, 0.6449346943255405, 0.5764622973925301, 0.19795221843003416
0.8746, 0.9261, 0.9342, 0.9644, 0.7418, 0.6239, 0.6449, 0.5765, 0.1980
[spg_ntc]
All: 0.8576570887275023, Dep: 0.9169906242853877, Zero: 0.571060382916053 / tp: [13110  9064  1886  2726   358    18], fp: [1121  296  402 1690  284   30], fn: [1201  541  795 2038  425  193]
0.9186462055917595, 0.9558660690746111, 0.7591064600523244, 0.5938997821350763, 0.5024561403508772, 0.13899613899613902
0.8577, 0.9170, 0.9186, 0.9559, 0.7591, 0.5711, 0.5939, 0.5025, 0.1390
[spl_ntc]
All: 0.857572686387534, Dep: 0.9173744584200179, Zero: 0.5662482566248256 / tp: [13089  9079  1864  2677   352    16], fp: [1071  309  382 1626  293   33], fn: [1221  527  819 2087  431  195]
0.9194942044257113, 0.9559861008739601, 0.7563400284033274, 0.5904929965810081, 0.49299719887955185, 0.12307692307692308
0.8576, 0.9174, 0.9195, 0.9560, 0.7563, 0.5662, 0.5905, 0.4930, 0.1231
[spn_ntc]
All: 0.8576570887275023, Dep: 0.9169906242853877, Zero: 0.571060382916053 / tp: [13110  9064  1886  2726   358    18], fp: [1121  296  402 1690  284   30], fn: [1201  541  795 2038  425  193]
0.9186462055917595, 0.9558660690746111, 0.7591064600523244, 0.5938997821350763, 0.5024561403508772, 0.13899613899613902
0.8577, 0.9170, 0.9186, 0.9559, 0.7591, 0.5711, 0.5939, 0.5025, 0.1390
[bertspg_ntc]
All: 0.8785067338741435, Dep: 0.9288526273749073, Zero: 0.6357300073367572 / tp: [13325  9157  1938  3023   415    28], fp: [ 873  253  443 1395  246   36], fn: [ 989  439  744 1743  368  184]
0.9346941638608305, 0.9635904451225928, 0.765554019356113, 0.6583188153310104, 0.574792243767313, 0.2028985507246377
0.8785, 0.9289, 0.9347, 0.9636, 0.7656, 0.6357, 0.6583, 0.5748, 0.2029
[bertspl_ntc]
All: 0.8787692501694487, Dep: 0.929017041996348, Zero: 0.6356189599631845 / tp: [13328  9146  1948  3011   413    29], fp: [ 876  245  440 1382  234   35], fn: [ 986  451  734 1754  371  183]
0.9347079037800687, 0.9633452706972826, 0.7684418145956609, 0.6575671544005242, 0.5772187281621245, 0.21014492753623187
0.8788, 0.9290, 0.9347, 0.9633, 0.7684, 0.6356, 0.6576, 0.5772, 0.2101
[bertspn_ntc]
All: 0.8789618560755013, Dep: 0.9290724863600934, Zero: 0.6387167335034634 / tp: [13362  9148  1926  3057   422    25], fp: [ 918  247  405 1432  241   32], fn: [ 950  454  757 1709  363  187]
0.9346670397313934, 0.9630994367531717, 0.7682489030714001, 0.6606158833063209, 0.5828729281767956, 0.1858736059479554
0.8790, 0.9291, 0.9347, 0.9631, 0.7682, 0.6387, 0.6606, 0.5829, 0.1859
[sl_bccwj]
All: 0.7836949375410914, Dep: 0.8370541611624835, Zero: 0.5241002570694088 / tp: [6505 4388 1780 1359  231   41], fp: [910 752 512 695 222  96], fn: [1432  863  465 1330  401  218]
0.8474465867639397, 0.8445770378211913, 0.7846594666078908, 0.573055028462998, 0.42580645161290326, 0.2070707070707071
0.7837, 0.8371, 0.8474, 0.8446, 0.7847, 0.5241, 0.5731, 0.4258, 0.2071
[bertsl_bccwj]
All: 0.8118676510801204, Dep: 0.8576707357307819, Zero: 0.5900958466453674 / tp: [6784 4438 1776 1515  288   44], fp: [898 616 393 630 178  45], fn: [1140  804  463 1156  342  215]
0.8694092015891324, 0.8620823620823622, 0.8058076225045372, 0.6291528239202657, 0.5255474452554746, 0.25287356321839083
0.8119, 0.8577, 0.8694, 0.8621, 0.8058, 0.5901, 0.6292, 0.5255, 0.2529
[spg_bccwj]
All: 0.7799091230647839, Dep: 0.8356146354303764, Zero: 0.5137556987894984 / tp: [6607 4373 1718 1374  231   29], fp: [1054  761  431  824  262   61], fn: [1339  879  532 1315  401  230]
0.8466713654129557, 0.8420951280569998, 0.7810866105933167, 0.5623081645181094, 0.4106666666666667, 0.166189111747851
0.7799, 0.8356, 0.8467, 0.8421, 0.7811, 0.5138, 0.5623, 0.4107, 0.1662
[spl_bccwj]
All: 0.7829334479701229, Dep: 0.8380921267559622, Zero: 0.5274625510667271 / tp: [6691 4399 1737 1461  243   39], fp: [1095  757  469  946  271   67], fn: [1259  863  513 1230  389  220]
0.8504067107269954, 0.8444999040122865, 0.7796229802513465, 0.5731659474303648, 0.4240837696335079, 0.2136986301369863
0.7829, 0.8381, 0.8504, 0.8445, 0.7796, 0.5275, 0.5732, 0.4241, 0.2137
[spn_bccwj]
All: 0.7810074407151651, Dep: 0.8372776236330994, Zero: 0.5247510034190576 / tp: [6711 4388 1726 1497  230   38], fp: [1148  749  444 1046  255   78], fn: [1244  870  530 1194  403  221]
0.8487416213481724, 0.8442520442520443, 0.7799367374604609, 0.5720290408865113, 0.41144901610017887, 0.20266666666666666
0.7810, 0.8373, 0.8487, 0.8443, 0.7799, 0.5248, 0.5720, 0.4114, 0.2027
[bertspg_bccwj]
All: 0.809180749350545, Dep: 0.8573850668405608, Zero: 0.5874943769680611 / tp: [6861 4516 1771 1608  316   35], fp: [1020  715  370  864  239   43], fn: [1068  729  472 1066  314  225]
0.8679316888045541, 0.8621611302023672, 0.8079379562043796, 0.624951418577536, 0.5333333333333332, 0.2071005917159763
0.8092, 0.8574, 0.8679, 0.8622, 0.8079, 0.5875, 0.6250, 0.5333, 0.2071
[bertspl_bccwj]
All: 0.8083976833976835, Dep: 0.8563317779880575, Zero: 0.5874567604151 / tp: [6865 4506 1751 1602  316   35], fp: [1016  722  370  840  257   37], fn: [1066  740  489 1071  314  224]
0.8683278522641033, 0.8604162688562154, 0.8030268287090118, 0.6263929618768329, 0.5253532834580216, 0.21148036253776437
0.8084, 0.8563, 0.8683, 0.8604, 0.8030, 0.5875, 0.6264, 0.5254, 0.2115
[bertspn_bccwj]
All: 0.8092135253264687, Dep: 0.8570776553432875, Zero: 0.5873999093518658 / tp: [6855 4530 1760 1581  324   39], fp: [977 744 382 811 248  52], fn: [1078  722  481 1092  307  221]
0.8696479543292103, 0.8607258217746533, 0.8031028975587499, 0.624284304047384, 0.5386533665835412, 0.2222222222222222
0.8092, 0.8571, 0.8696, 0.8607, 0.8031, 0.5874, 0.6243, 0.5387, 0.2222

'''