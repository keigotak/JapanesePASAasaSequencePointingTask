# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossEntropyLossWithOneHot(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLossWithOneHot, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ret = 0.0
        input = input.float()
        target = target.float()
        for input_item, target_item in zip(input, target):
            if target_item != self.ignore_index and target_item != 0:
                ret -= torch.log(input_item) * target_item
        return ret


class MSELossWithIgnore(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean', ignore=-1):
        super(MSELossWithIgnore, self).__init__(size_average, reduce, reduction)
        self.ignore = ignore

    def forward(self, input, target):
        ret = 0.0
        for input_item, target_item in zip(input, target):
            idx = self.get_ignore_idx(target_item)
            if idx != -1:
                input_item = input_item.narrow(0, 0, idx)
                input_item = input_item.narrow(1, 0, idx)
                target_item = target_item.narrow(0, 0, idx)
                target_item = target_item.narrow(1, 0, idx)
            ret += F.mse_loss(input_item.float(), target_item.float(), reduction=self.reduction)
        return ret

    def get_ignore_idx(self, matrix):
        for idx, item in enumerate(matrix[0]):
            if self.ignore == item:
                return idx
        return -1


class BCELossWithLogitsAndIgnore(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, reduction='elementwise_mean', ignore=-1):
        super(BCELossWithLogitsAndIgnore, self).__init__(weight, reduction)
        self.ignore = ignore
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ret = 0.0
        for input_item, target_item in zip(input, target):
            idx = self.get_ignore_idx(target_item)
            if idx != -1:
                input_item = input_item.narrow(0, 0, idx)
                input_item = input_item.narrow(1, 0, idx)
                target_item = target_item.narrow(0, 0, idx)
                target_item = target_item.narrow(1, 0, idx)
            ret += F.binary_cross_entropy_with_logits(input_item.float(), target_item.float(), weight=self.weight, reduction=self.reduction)
        return ret

    def get_ignore_idx(self, matrix):
        for idx, item in enumerate(matrix[0]):
            if self.ignore == item:
                return idx
        return -1