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
    with Path("../../results/pasa-lstm-20200329-172255-856357/sl_20200329-172255_model-0_epoch17-f0.8438.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-172256-779451/sl_20200329-172256_model-0_epoch13-f0.8455.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-172259-504777/sl_20200329-172259_model-0_epoch16-f0.8465.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-172333-352525/sl_20200329-172333_model-0_epoch16-f0.8438.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-172320-621931/sl_20200329-172320_model-0_epoch13-f0.8473.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sp_ntc(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "none":
        with Path("../../results/pasa-pointer-20200328-224527-671480/sp_20200328-224527_model-0_epoch10-f0.8453.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-224523-646336/sp_20200328-224523_model-0_epoch10-f0.8450.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-224545-172444/sp_20200328-224545_model-0_epoch14-f0.8459.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-224538-434833/sp_20200328-224538_model-0_epoch11-f0.8450.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-224530-441394/sp_20200328-224530_model-0_epoch10-f0.8446.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "global":
        with Path("../../results/pasa-pointer-20200328-220620-026555/sp_20200328-220620_model-0_epoch13-f0.8462.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-220701-953235/sp_20200328-220701_model-0_epoch10-f0.8466.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-220650-498845/sp_20200328-220650_model-0_epoch10-f0.8469.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-220618-338695/sp_20200328-220618_model-0_epoch9-f0.8466.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200328-220642-006275/sp_20200328-220642_model-0_epoch11-f0.8461.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-pointer-20200329-181219-050793/sp_20200329-181219_model-0_epoch10-f0.8455.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-181242-757471/sp_20200329-181242_model-0_epoch11-f0.8455.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-181255-253679/sp_20200329-181255_model-0_epoch12-f0.8440.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-181329-741718/sp_20200329-181329_model-0_epoch9-f0.8476.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-181405-914906/sp_20200329-181405_model-0_epoch12-f0.8456.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsl_ntc():
    with Path("../../results/pasa-bertsl-20200402-123819-415751/bsl_20200402-123819_model-0_epoch7-f0.8631.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-123818-814117/bsl_20200402-123818_model-0_epoch7-f0.8650.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-123820-333582/bsl_20200402-123820_model-0_epoch9-f0.8620.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-123820-545980/bsl_20200402-123820_model-0_epoch5-f0.8611.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-201956-237530/bsl_20200402-201956_model-0_epoch11-f0.8647.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsp_ntc(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "none":
        with Path("../../results/pasa-bertptr-20200402-165144-157728/bsp_20200402-165144_model-0_epoch6-f0.8702.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-165338-628976/bsp_20200402-165338_model-0_epoch10-f0.8703.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-165557-747882/bsp_20200402-165557_model-0_epoch17-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-170544-734496/bsp_20200402-170544_model-0_epoch8-f0.8698.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-170813-804379/bsp_20200402-170813_model-0_epoch10-f0.8706.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "global":
        with Path("../../results/pasa-bertptr-20200402-134057-799938/bsp_20200402-134057_model-0_epoch11-f0.8703.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-134057-825245/bsp_20200402-134057_model-0_epoch6-f0.8709.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-134057-738238/bsp_20200402-134057_model-0_epoch10-f0.8719.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-134057-896365/bsp_20200402-134057_model-0_epoch10-f0.8709.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-134106-778681/bsp_20200402-134106_model-0_epoch10-f0.8707.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-bertptr-20200402-195131-152329/bsp_20200402-195131_model-0_epoch6-f0.8710.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-195230-748475/bsp_20200402-195230_model-0_epoch10-f0.8705.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-195441-889702/bsp_20200402-195441_model-0_epoch10-f0.8718.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-200529-393340/bsp_20200402-200529_model-0_epoch8-f0.8701.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200402-200821-141107/bsp_20200402-200821_model-0_epoch10-f0.8707.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sl_bccwj():
    with Path("../../results/pasa-lstm-20200329-003503-990662/sl_20200329-003503_model-0_epoch12-f0.7641.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-003529-533233/sl_20200329-003529_model-0_epoch14-f0.7686.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-003625-441811/sl_20200329-003625_model-0_epoch12-f0.7646.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-003702-744631/sl_20200329-003702_model-0_epoch14-f0.7649.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-lstm-20200329-003720-611158/sl_20200329-003720_model-0_epoch5-f0.7630.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_sp_bccwj(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "none":
        with Path(
            "../../results/pasa-pointer-20200329-035820-082435/sp_20200329-035820_model-0_epoch13-f0.7628.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-035828-731188/sp_20200329-035828_model-0_epoch5-f0.7584.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-035921-430627/sp_20200329-035921_model-0_epoch5-f0.7614.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-040037-823312/sp_20200329-040037_model-0_epoch7-f0.7615.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-040155-312838/sp_20200329-040155_model-0_epoch14-f0.7619.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "global":
        with Path("../../results/pasa-pointer-20200329-022106-069150/sp_20200329-022106_model-0_epoch6-f0.7589.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-022109-056568/sp_20200329-022109_model-0_epoch5-f0.7621.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-022128-955906/sp_20200329-022128_model-0_epoch6-f0.7634.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-022222-724719/sp_20200329-022222_model-0_epoch5-f0.7576.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-pointer-20200329-022313-903459/sp_20200329-022313_model-0_epoch9-f0.7616.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path(
            "../../results/pasa-pointer-20200329-053317-066569/sp_20200329-053317_model-0_epoch9-f0.7608.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-053420-334813/sp_20200329-053420_model-0_epoch10-f0.7654.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-053658-852976/sp_20200329-053658_model-0_epoch5-f0.7619.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-053744-584854/sp_20200329-053744_model-0_epoch14-f0.7623.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path(
            "../../results/pasa-pointer-20200329-053954-847594/sp_20200329-053954_model-0_epoch10-f0.7618.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsl_bccwj():
    with Path("../../results/pasa-bertsl-20200403-112105-536009/bsl_20200403-112105_model-0_epoch6-f0.7916.h5.pkl").open('rb') as f:
        items0 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-225320-031641/bsl_20200402-225320_model-0_epoch9-f0.7894.h5.pkl").open('rb') as f:
        items1 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-225138-903629/bsl_20200402-225138_model-0_epoch6-f0.7916.h5.pkl").open('rb') as f:
        items2 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-230314-149516/bsl_20200402-230314_model-0_epoch8-f0.7916.h5.pkl").open('rb') as f:
        items3 = pickle.load(f)
    with Path("../../results/pasa-bertsl-20200402-230524-638769/bsl_20200402-230524_model-0_epoch8-f0.7877.h5.pkl").open('rb') as f:
        items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_bertsp_bccwj(mode="global"):
    items0, items1, items2, items3, items4 = [], [], [], [], []
    if mode == "none":
        with Path("../../results/pasa-bertptr-20200403-010141-686124/bsp_20200403-010141_model-0_epoch11-f0.7924.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-010141-667945/bsp_20200403-010141_model-0_epoch7-f0.7936.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-010141-341382/bsp_20200403-010141_model-0_epoch12-f0.7922.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-011035-863656/bsp_20200403-011035_model-0_epoch11-f0.7945.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-011130-170880/bsp_20200403-011130_model-0_epoch9-f0.7937.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    elif mode == "global":
        with Path("../../results/pasa-bertptr-20200403-025538-564688/bsp_20200403-025538_model-0_epoch11-f0.7939.h5.pkl").open(
            'rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-025740-192547/bsp_20200403-025740_model-0_epoch9-f0.7943.h5.pkl").open(
            'rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-025725-718275/bsp_20200403-025725_model-0_epoch11-f0.7955.h5.pkl").open(
            'rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-030515-753190/bsp_20200403-030515_model-0_epoch9-f0.7950.h5.pkl").open(
            'rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-030648-760108/bsp_20200403-030648_model-0_epoch8-f0.7943.h5.pkl").open(
            'rb') as f:
            items4 = pickle.load(f)
    elif mode == "local":
        with Path("../../results/pasa-bertptr-20200403-045040-398629/bsp_20200403-045040_model-0_epoch9-f0.7955.h5.pkl").open('rb') as f:
            items0 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-045320-503212/bsp_20200403-045320_model-0_epoch12-f0.7940.h5.pkl").open('rb') as f:
            items1 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-045346-565331/bsp_20200403-045346_model-0_epoch9-f0.7944.h5.pkl").open('rb') as f:
            items2 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-050141-426441/bsp_20200403-050141_model-0_epoch8-f0.7963.h5.pkl").open('rb') as f:
            items3 = pickle.load(f)
        with Path("../../results/pasa-bertptr-20200403-050204-373548/bsp_20200403-050204_model-0_epoch8-f0.7924.h5.pkl").open('rb') as f:
            items4 = pickle.load(f)
    return items0, items1, items2, items3, items4


def get_mean_std_of_softmax(item1, item2, item3, item4, item5, dim):
    ## それぞれに対してsoftmax 取ったのち和を取る
    avg_prediction = torch.mean(torch.Tensor([F.softmax(torch.Tensor(np.array(item1)), dim=dim).tolist(),
                                              F.softmax(torch.Tensor(np.array(item2)), dim=dim).tolist(),
                                              F.softmax(torch.Tensor(np.array(item3)), dim=dim).tolist(),
                                              F.softmax(torch.Tensor(np.array(item4)), dim=dim).tolist(),
                                              F.softmax(torch.Tensor(np.array(item5)), dim=dim).tolist()]), dim=0).unsqueeze(0)
    dev_prediction = torch.std(torch.Tensor([F.softmax(torch.Tensor(np.array(item1)), dim=dim).tolist(),
                                             F.softmax(torch.Tensor(np.array(item2)), dim=dim).tolist(),
                                             F.softmax(torch.Tensor(np.array(item3)), dim=dim).tolist(),
                                             F.softmax(torch.Tensor(np.array(item4)), dim=dim).tolist(),
                                             F.softmax(torch.Tensor(np.array(item5)), dim=dim).tolist()]), dim=0).unsqueeze(0)
    return avg_prediction, dev_prediction


def main(mode, corpus):
    mode = '{}_{}'.format(mode, corpus)
    print('[{}]'.format(mode))
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
            t_props = [i0[1]]
            t_labels = [i0[2]]

            avg_prediction, std_prediction = get_mean_std_of_softmax(i0[0], i1[0], i2[0], i3[0], i4[0], dim=1)

            _, prediction = torch.max(avg_prediction, 2)
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
        print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp, num_fn))
        print(', '.join(map(str, f1s)))
        print('{} & - & {} & {} & {}, {} & {} & {} & {} & {}'.format(Decimal(str(all_score * 100)).quantize(Decimal('0.01'),
                                                                                                      rounding=ROUND_HALF_UP),
                                                                     Decimal(str(dep_score * 100)).quantize(Decimal('0.01'),
                                                                                                      rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[0] * 100)).quantize(Decimal('0.01'),
                                                                                                   rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[1] * 100)).quantize(Decimal('0.01'),
                                                                                                   rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[2] * 100)).quantize(Decimal('0.01'),
                                                                                                   rounding=ROUND_HALF_UP),
                                                                     Decimal(str(zero_score * 100)).quantize(Decimal('0.01'),
                                                                                                       rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[3] * 100)).quantize(Decimal('0.01'),
                                                                                                   rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[4] * 100)).quantize(Decimal('0.01'),
                                                                                                   rounding=ROUND_HALF_UP),
                                                                     Decimal(str(f1s[5] * 100)).quantize(Decimal('0.01'),
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
            t_props = [i0[3]]
            t_labels = [i0[4]]
            avg_ga, std_ga = get_mean_std_of_softmax(i0[0], i1[0], i2[0], i3[0], i4[0], dim=0)
            avg_ni, std_ni = get_mean_std_of_softmax(i0[1], i1[1], i2[1], i3[1], i4[1], dim=0)
            avg_wo, std_wo = get_mean_std_of_softmax(i0[2], i1[2], i2[2], i3[2], i4[2], dim=0)

            if 'global' in mode:
                ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(avg_ga, avg_ni, avg_wo)
            elif 'local' in mode:
                ga_prediction, ni_prediction, wo_prediction = get_ordered_prediction(avg_ga, avg_ni, avg_wo)
            else:
                ga_prediction, ni_prediction, wo_prediction = get_no_decode_prediction(avg_ga, avg_ni, avg_wo)
            s_size = avg_ga.shape[1]
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
        print('All: {}, Dep: {}, Zero: {} / tp: {}, fp: {}, fn: {}'.format(all_score, dep_score, zero_score, num_tp, num_fp,
                                                                           num_fn))
        print(', '.join(map(str, f1s)))
        print('{} & - & {} & {} & {} & {} & {} & {} & {} & {}'.format(Decimal(str(all_score * 100)).quantize(Decimal('0.01'),
                                                                                                       rounding=ROUND_HALF_UP),
                                                                      Decimal(str(dep_score * 100)).quantize(Decimal('0.01'),
                                                                                                       rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[0] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[1] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[2] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP),
                                                                      Decimal(str(zero_score * 100)).quantize(Decimal('0.01'),
                                                                                                        rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[3] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[4] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP),
                                                                      Decimal(str(f1s[5] * 100)).quantize(Decimal('0.01'),
                                                                                                    rounding=ROUND_HALF_UP)
                                                                      ))


if __name__ == "__main__":
    tags = ['sl', 'spn', 'spg', 'spl', 'bertsl', 'bertspn', 'bertspg', 'bertspl']
    parser = argparse.ArgumentParser(description='PASA Ensamble')
    parser.add_argument('--mode', default=None, type=str, choices=tags + ['all'])
    parser.add_argument('--corpus', default=None, type=str, choices=['ntc', 'bccwj', 'all'])
    arguments = parser.parse_args()
    if arguments.corpus == "all" and arguments.mode == "all":
        for corpus in ['ntc', 'bccwj']:
            for mode in tags:
                main(mode=mode, corpus=corpus)
    elif arguments.corpus == "all" and arguments.mode != "all":
        for corpus in ['ntc', 'bccwj']:
            main(mode=arguments.mode, corpus=corpus)
    elif arguments.corpus != "all" and arguments.mode == "all":
        for mode in tags:
            main(mode=mode, corpus=arguments.corpus)
    else:
        main(mode=arguments.mode, corpus=arguments.corpus)


'''
[sl_ntc]
All: 0.8560959146282665, Dep: 0.9138582526134075, Zero: 0.5575493209080995 / tp: [12913  9019  1890  2485   313    14], fp: [ 973  322  450 1270  215   32], fn: [1387  570  789 2279  470  197]
0.9162704888951962, 0.9528790279978869, 0.7531380753138075, 0.5834018077239113, 0.47749809305873386, 0.10894941634241245
0.8561 & 0.9139 & 0.9163 & 0.9529 & 0.7531 & 0.5575 & 0.5834 & 0.4775 & 0.1089
[spn_ntc]
All: 0.857993224202894, Dep: 0.9175306944683126, Zero: 0.5691523853635942 / tp: [13033  9055  1938  2735   321    16], fp: [ 987  295  461 1648  274   43], fn: [1282  549  745 2029  462  195]
0.9199223575083818, 0.9554711406563258, 0.7626918536009445, 0.5980102765934185, 0.4658925979680697, 0.11851851851851852
0.8580 & 0.9175 & 0.9199 & 0.9555 & 0.7627 & 0.5692 & 0.5980 & 0.4659 & 0.1185
[spg_ntc]
All: 0.8555865214086648, Dep: 0.9158892797063772, Zero: 0.5645756457564576 / tp: [13023  9045  1888  2727   322    11], fp: [1030  307  420 1691  299   32], fn: [1291  555  797 2037  461  200]
0.9181795748581097, 0.9545166737019839, 0.756258762267174, 0.5939882378566761, 0.45868945868945876, 0.08661417322834646
0.8556 & 0.9159 & 0.9182 & 0.9545 & 0.7563 & 0.5646 & 0.5940 & 0.4587 & 0.0866
[spl_ntc]
All: 0.8570072911888907, Dep: 0.9162772065696354, Zero: 0.5694187338022955 / tp: [13054  9044  1919  2730   331    15], fp: [1077  291  441 1671  264   35], fn: [1258  557  765 2034  452  196]
0.9179059874134233, 0.9552175749894382, 0.760904044409199, 0.5957446808510637, 0.48040638606676345, 0.11494252873563217
0.8570 & 0.9163 & 0.9179 & 0.9552 & 0.7609 & 0.5694 & 0.5957 & 0.4804 & 0.1149
[bertsl_ntc]
All: 0.8751924103917318, Dep: 0.92562358276644, Zero: 0.6268378931695514 / tp: [13362  9164  1966  2962   383    23], fp: [ 940  248  672 1372  197   51], fn: [ 938  425  713 1802  400  188]
0.9343402559261591, 0.9645808115362349, 0.7395147639646417, 0.6511321169487799, 0.561995597945708, 0.1614035087719298
0.8752 & 0.9256 & 0.9343 & 0.9646 & 0.7395 & 0.6268 & 0.6511 & 0.5620 & 0.1614
[bertspn_ntc]
All: 0.8782098427353101, Dep: 0.9274137247872603, Zero: 0.6390977443609023 / tp: [13376  9151  2049  3040   422    23], fp: [ 950  263  612 1383  237   39], fn: [ 930  456  636 1725  363  189]
0.9343392008941046, 0.9621996740444771, 0.7665544332211, 0.6617326948193296, 0.5844875346260386, 0.1678832116788321
0.8782 & 0.9274 & 0.9343 & 0.9622 & 0.7666 & 0.6391 & 0.6617 & 0.5845 & 0.1679
[bertspg_ntc]
All: 0.8777360506591662, Dep: 0.927307852570173, Zero: 0.636355252236466 / tp: [13348  9132  2000  3018   410    22], fp: [ 906  240  580 1365  230   38], fn: [ 958  469  685 1747  374  189]
0.934733893557423, 0.962631107363095, 0.7597340930674265, 0.6598163533012681, 0.5758426966292134, 0.16236162361623616
0.8777 & 0.9273 & 0.9347 & 0.9626 & 0.7597 & 0.6364 & 0.6598 & 0.5758 & 0.1624
[bertspl_ntc]
All: 0.8780778707176599, Dep: 0.9270238590388324, Zero: 0.6389581601551676 / tp: [13356  9149  2012  3017   421    21], fp: [ 906  255  610 1341  228   38], fn: [ 956  458  675 1748  364  190]
0.9348358647721704, 0.9624953974015044, 0.7579581842154831, 0.661405239504549, 0.5871687587168759, 0.15555555555555559
0.8781 & 0.9270 & 0.9348 & 0.9625 & 0.7580 & 0.6390 & 0.6614 & 0.5872 & 0.1556
[sl_bccwj]
All: 0.7872246208347171, Dep: 0.8405507882658153, Zero: 0.5229146060006594 / tp: [6629 4315 1692 1337  219   30], fp: [983 608 406 682 167  51], fn: [1308  936  553 1352  413  229]
0.8526593350054666, 0.8482406133280912, 0.7791848952337095, 0.5679694137638062, 0.4302554027504911, 0.17647058823529413
0.7872 & 0.8406 & 0.8527 & 0.8482 & 0.7792 & 0.5229 & 0.5680 & 0.4303 & 0.1765
[spn_bccwj]
All: 0.7850462280584215, Dep: 0.842580771874693, Zero: 0.5252734259532958 / tp: [6716 4451 1703 1489  260   28], fp: [1067  747  404 1026  314   64], fn: [1233  809  549 1204  372  232]
0.8538011695906433, 0.8512143813348633, 0.7813718742830924, 0.5718125960061444, 0.4311774461028193, 0.15909090909090912
0.7850 & 0.8426 & 0.8538 & 0.8512 & 0.7814 & 0.5253 & 0.5718 & 0.4312 & 0.1591
[spg_bccwj]
All: 0.783164223975949, Dep: 0.8411679979049366, Zero: 0.5189382642409782 / tp: [6729 4407 1712 1476  233   31], fp: [1092  723  419 1100  224   57], fn: [1226  855  537 1214  403  228]
0.8530679513184584, 0.8481524249422632, 0.7817351598173515, 0.5605772882643373, 0.4263494967978042, 0.1786743515850144
0.7832 & 0.8412 & 0.8531 & 0.8482 & 0.7817 & 0.5189 & 0.5606 & 0.4263 & 0.1787
[spl_bccwj]
All: 0.7824644930055902, Dep: 0.8390947038872946, Zero: 0.524170757102484 / tp: [6737 4422 1706 1487  244   31], fp: [1133  755  431 1041  287   51], fn: [1218  845  552 1204  388  228]
0.851437598736177, 0.8468019915741094, 0.7763367463026166, 0.5698409657022417, 0.4196044711951849, 0.18181818181818182
0.7825 & 0.8391 & 0.8514 & 0.8468 & 0.7763 & 0.5242 & 0.5698 & 0.4196 & 0.1818
[bertsl_bccwj]
All: 0.8109907500886792, Dep: 0.8575557526193057, Zero: 0.582392776523702 / tp: [6789 4531 1735 1469  302   35], fp: [896 720 371 588 212  36], fn: [1135  711  504 1202  328  224]
0.8698827599461848, 0.863623367959592, 0.7986191024165706, 0.621404399323181, 0.527972027972028, 0.21212121212121215
0.8110 & 0.8576 & 0.8699 & 0.8636 & 0.7986 & 0.5824 & 0.6214 & 0.5280 & 0.2121
[bertspn_bccwj]
All: 0.8129822478821562, Dep: 0.8595235761199882, Zero: 0.5926040538449636 / tp: [6890 4494 1768 1578  300   37], fp: [983 674 375 780 165  43], fn: [1038  754  475 1093  330  222]
0.872096702740333, 0.8629032258064516, 0.8062015503875969, 0.627560151123484, 0.547945205479452, 0.21828908554572268
0.8130 & 0.8595 & 0.8721 & 0.8629 & 0.8062 & 0.5926 & 0.6276 & 0.5479 & 0.2183
[bertspg_bccwj]
All: 0.8106239279588335, Dep: 0.858984375, Zero: 0.5852548543689321 / tp: [6871 4539 1784 1591  303   35], fp: [1032  688  375  842  212   44], fn: [1060  715  462 1081  330  225]
0.8678792471895921, 0.866138727220685, 0.8099886492622019, 0.6233104799216455, 0.5278745644599304, 0.20648967551622416
0.8106 & 0.8590 & 0.8679 & 0.8661 & 0.8100 & 0.5853 & 0.6233 & 0.5279 & 0.2065
[bertspl_bccwj]
All: 0.8113116855372784, Dep: 0.8578663723134862, Zero: 0.5955461931989168 / tp: [6852 4544 1816 1613  325   41], fp: [998 730 406 807 236  55], fn: [1095  719  430 1063  308  219]
0.8675064885737799, 0.8624845781531745, 0.8128916741271263, 0.6330455259026687, 0.5443886097152428, 0.23033707865168537
0.8113 & 0.8579 & 0.8675 & 0.8625 & 0.8129 & 0.5955 & 0.6330 & 0.5444 & 0.2303

---------------

[sl_ntc]
All: 0.8560046977415925, Dep: 0.9159347073119389, Zero: 0.5647321428571428 / tp: [13039  9043  1850  2693   330    13], fp: [1012  329  416 1671  254   33], fn: [1261  546  829 2071  453  198]
0.919826461147755, 0.9538526449026951, 0.7482305358948432, 0.5900525854513584, 0.48280907095830283, 0.10116731517509725
0.8560, 0.9159, 0.9198, 0.9539, 0.7482, 0.5647, 0.5901, 0.4828, 0.1012
[spn_ntc]
All: 0.8572965277228506, Dep: 0.9167844185602444, Zero: 0.5700322729368372 / tp: [13081  9072  1853  2722   353    16], fp: [1114  292  364 1690  279   26], fn: [1231  529  828 2043  430  195]
0.9177395025783142, 0.9567097284471394, 0.7566353613719886, 0.5932221858995315, 0.4989399293286219, 0.12648221343873517
0.8573, 0.9168, 0.9177, 0.9567, 0.7566, 0.5700, 0.5932, 0.4989, 0.1265
[spg_ntc]
All: 0.8572965277228506, Dep: 0.9167844185602444, Zero: 0.5700322729368372 / tp: [13081  9072  1853  2722   353    16], fp: [1114  292  364 1690  279   26], fn: [1231  529  828 2043  430  195]
0.9177395025783142, 0.9567097284471394, 0.7566353613719886, 0.5932221858995315, 0.4989399293286219, 0.12648221343873517
0.8573, 0.9168, 0.9177, 0.9567, 0.7566, 0.5700, 0.5932, 0.4989, 0.1265
[spl_ntc]
All: 0.8570069903466642, Dep: 0.9172780070304142, Zero: 0.5633435725588756 / tp: [13072  9093  1842  2660   353    13], fp: [1069  314  357 1641  289   29], fn: [1240  510  840 2104  430  198]
0.9188486275612413, 0.9566543924250395, 0.7547633681622619, 0.5868725868725869, 0.495438596491228, 0.10276679841897232
0.8570, 0.9173, 0.9188, 0.9567, 0.7548, 0.5633, 0.5869, 0.4954, 0.1028
[bertsl_ntc]
All: 0.8740409307894694, Dep: 0.925968822982133, Zero: 0.6209513233388859 / tp: [13331  9145  1908  2919   407    29], fp: [ 890  248  577 1412  231   50], fn: [ 969  444  771 1845  376  182]
0.9348199572245013, 0.9635444104941523, 0.7389620449264135, 0.6418911489829577, 0.5728360309641098, 0.2
0.8740, 0.9260, 0.9348, 0.9635, 0.7390, 0.6210, 0.6419, 0.5728, 0.2000
[bertspn_ntc]
All: 0.8783153783153784, Dep: 0.9291191749281679, Zero: 0.6342444464759118 / tp: [13360  9146  1908  3033   415    21], fp: [ 899  247  402 1426  249   34], fn: [ 950  452  775 1733  368  191]
0.9352794987573942, 0.9631930914643778, 0.7642699779691569, 0.657560975609756, 0.5736005528680028, 0.15730337078651685
0.8783, 0.9291, 0.9353, 0.9632, 0.7643, 0.6342, 0.6576, 0.5736, 0.1573
[bertspg_ntc]
All: 0.8775371804374911, Dep: 0.9284204512996286, Zero: 0.6322071244950422 / tp: [13304  9154  1920  3010   409    24], fp: [ 865  253  425 1394  257   36], fn: [1008  445  763 1757  374  188]
0.9342368596608266, 0.9632747553404188, 0.7637231503579952, 0.6564169665249155, 0.5645272601794341, 0.17647058823529413
0.8775, 0.9284, 0.9342, 0.9633, 0.7637, 0.6322, 0.6564, 0.5645, 0.1765
[bertspl_ntc]
All: 0.878018009493621, Dep: 0.9286040493225757, Zero: 0.6332074776682937 / tp: [13320  9144  1936  3009   405    24], fp: [ 876  253  428 1381  244   36], fn: [ 995  453  747 1756  378  188]
0.9343762056750026, 0.9628303674844688, 0.7671884287695659, 0.6573457127252867, 0.5656424581005587, 0.17647058823529413
0.8780, 0.9286, 0.9344, 0.9628, 0.7672, 0.6332, 0.6573, 0.5656, 0.1765

[sl_bccwj]
All: 0.7842600700525395, Dep: 0.8376006334961066, Zero: 0.5250160359204618 / tp: [6524 4394 1775 1368  230   39], fp: [919 749 514 698 227  94], fn: [1413  857  470 1321  402  220]
0.8483745123537061, 0.8454877814123533, 0.7829730921923247, 0.5753943217665615, 0.4224058769513315, 0.1989795918367347
0.7843, 0.8376, 0.8484, 0.8455, 0.7830, 0.5250, 0.5754, 0.4224, 0.1990
[spn_bccwj]
All: 0.7797716150081565, Dep: 0.8363517702724752, Zero: 0.5228215767634855 / tp: [6725 4383 1707 1491  233   40], fp: [1158  761  435 1065  266   70], fn: [1230  877  554 1199  400  220]
0.8492233867912615, 0.8425605536332179, 0.7753804224392459, 0.5684330918795273, 0.411660777385159, 0.21621621621621623
0.7798, 0.8364, 0.8492, 0.8426, 0.7754, 0.5228, 0.5684, 0.4117, 0.2162
[spg_bccwj]
All: 0.7793294571537249, Dep: 0.835075951864273, Zero: 0.5140801001251565 / tp: [6607 4377 1715 1379  234   30], fp: [1066  767  434  846  261   61], fn: [1340  875  534 1311  398  229]
0.8459667093469911, 0.8420546363986149, 0.779899954524784, 0.5611393692777212, 0.4152617568766637, 0.17142857142857143
0.7793, 0.8351, 0.8460, 0.8421, 0.7799, 0.5141, 0.5611, 0.4153, 0.1714
[spl_bccwj]
All: 0.782298320004294, Dep: 0.8378060724779629, Zero: 0.525934861278649 / tp: [6700 4398 1733 1463  241   40], fp: [1114  768  458  965  277   64], fn: [1249  861  518 1228  391  219]
0.8500919875658187, 0.8437410071942447, 0.780279153534444, 0.5715960148466498, 0.4191304347826087, 0.22038567493112948
0.7823, 0.8378, 0.8501, 0.8437, 0.7803, 0.5259, 0.5716, 0.4191, 0.2204
[bertsl_bccwj]
All: 0.8121251771889653, Dep: 0.8576696407993941, Zero: 0.5929329741720805 / tp: [6803 4444 1778 1531  295   45], fp: [915 629 399 647 188  45], fn: [1121  798  461 1140  335  214]
0.8698376166730598, 0.8616577799321378, 0.8052536231884059, 0.6314704062693338, 0.5300988319856244, 0.2578796561604585
0.8121, 0.8577, 0.8698, 0.8617, 0.8053, 0.5929, 0.6315, 0.5301, 0.2579
[bertspn_bccwj]
All: 0.8086534074133719, Dep: 0.8569097732093327, Zero: 0.5849969751966123 / tp: [6855 4517 1758 1576  321   37], fp: [974 736 382 824 243  48], fn: [1077  734  482 1097  309  223]
0.8698686631558912, 0.86005331302361, 0.8027397260273973, 0.6213286024048885, 0.5376884422110553, 0.21449275362318843
0.8087, 0.8569, 0.8699, 0.8601, 0.8027, 0.5850, 0.6213, 0.5377, 0.2145
[bertspg_bccwj]
All: 0.8095289136609679, Dep: 0.8577574967405476, Zero: 0.5866224766495933 / tp: [6871 4519 1768 1600  311   36], fp: [1027  700  372  849  238   42], fn: [1062  727  476 1072  319  224]
0.8680437117048828, 0.8636407071189679, 0.8065693430656935, 0.6248779535247021, 0.527565733672604, 0.21301775147928995
0.8095, 0.8578, 0.8680, 0.8636, 0.8066, 0.5866, 0.6249, 0.5276, 0.2130
[bertspl_bccwj]
All: 0.8080493694660584, Dep: 0.8556872734397963, Zero: 0.5886599488644909 / tp: [6860 4493 1748 1604  319   34], fp: [1024  714  360  832  256   41], fn: [1071  758  492 1070  311  225]
0.8675308251659817, 0.8592465098489194, 0.8040478380864765, 0.6277886497064579, 0.5294605809128631, 0.20359281437125748
0.8080, 0.8557, 0.8675, 0.8592, 0.8040, 0.5887, 0.6278, 0.5295, 0.2036

'''