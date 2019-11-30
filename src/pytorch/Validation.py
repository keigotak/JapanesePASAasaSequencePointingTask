# -*- coding: utf-8 -*-
import sys
import os

import numpy as np

sys.path.append(os.pardir)
from utils.Scores import get_prs, calculate_f


def get_pr_numbers(y_pred, y_label, y_property, type='sentence'):
    if type == 'pair':
        y_label = y_label.unsqueeze(1).data.cpu().numpy()
        y_pred = y_pred.unsqueeze(1).data.cpu().numpy()
        y_property = np.expand_dims(y_property, axis=1)
    return get_prs(y_pred, y_label, y_property)


def get_f_score(tp, fp, fn):
    all_score = calculate_f(tp, fp, fn)
    dep_score = calculate_f(tp[0:3], fp[0:3], fn[0:3])
    zero_score = calculate_f(tp[3:6], fp[3:6], fn[3:6])
    return all_score, dep_score, zero_score


if __name__ == "__main__":
    import torch

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1.0
    assert score_dep == 1.0
    assert score_zero == 1.0
    assert list(tp) == [1, 1, 1, 1, 1, 1]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    tp, fp, fn = get_pr_numbers(torch.tensor([[1,2,0,1,2,0,1,2,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[3,3,3,3,3,3,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    tp, fp, fn = get_pr_numbers(torch.tensor([[3,3,3,3,3,3,3,3,3]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,0,0,0,0,0,0,0,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [1, 0, 0, 1, 0, 0]
    assert list(fp) == [2, 0, 0, 2, 0, 0]
    assert list(fn) == [0, 1, 1, 0, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[1,1,1,1,1,1,1,1,1]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 1, 0, 0, 1, 0]
    assert list(fp) == [0, 2, 0, 0, 2, 0]
    assert list(fn) == [1, 0, 1, 1, 0, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[2,2,2,2,2,2,2,2,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 0, 1, 0, 0, 1]
    assert list(fp) == [0, 0, 2, 0, 0, 2]
    assert list(fn) == [1, 1, 0, 1, 1, 0]

    tp = [3,0,0,1,0,0]
    fp = [32,59,96,43,102,129]
    fn = [21,13,5,5,2,0]
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all - 0.01553 < 1e-4
    assert score_dep - 0.02586 < 1e-4
    assert score_zero - 0.00707 < 1e-4

    tp = [3, 0, 0, 2, 0, 0]
    fp = [2, 4, 3, 1, 2, 3]
    fn = [2, 1, 2, 1, 0, 0]
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all - 0.322581 < 1e-4
    assert score_dep - 0.30000 < 1e-4
    assert score_zero - 0.363636 < 1e-4
