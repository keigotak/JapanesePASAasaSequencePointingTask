import torch
from scipy.stats import rankdata
import numpy as np
# from munkres import Munkres


def argmax(data):
    max_data = max(data)
    max_index = np.where(data == max_data)
    return max_index[0][0]


def get_restricted_prediction(batch_ga_pred, batch_ni_pred, batch_wo_pred):
    ga_pred = []
    ni_pred = []
    wo_pred = []
    batch_ga_pred[batch_ga_pred != batch_ga_pred] = 0
    batch_ni_pred[batch_ni_pred != batch_ni_pred] = 0
    batch_wo_pred[batch_wo_pred != batch_wo_pred] = 0

    for sentence_ga, sentence_ni, sentence_wo in zip(batch_ga_pred, batch_ni_pred, batch_wo_pred):
        sentence_length = sentence_ga.shape[0]
        ret = [-1] * 3
        flatten_sentence = torch.cat((sentence_wo, sentence_ga, sentence_ni))
        rank = rankdata(flatten_sentence.cpu())

        def is_on_restricted(indexes, target):
            if target == 0:
                return False
            for index in indexes:
                if target == index:
                    return True
            return False

        def get_property(item, sentence_length):
            kaku = int(item / sentence_length)
            kou = item % sentence_length
            return kaku, kou

        item = argmax(rank)
        target_kaku1, target_kou1 = get_property(item, sentence_length)
        rank[item] = -2
        ret[target_kaku1] = target_kou1

        item = argmax(rank)
        target_kaku2, target_kou2 = get_property(item, sentence_length)
        rank[item] = -2
        while is_on_restricted([target_kou1], target_kou2) or target_kaku2 in [target_kaku1]:
            item = argmax(rank)
            target_kaku2, target_kou2 = get_property(item, sentence_length)
            rank[item] = -2
        ret[target_kaku2] = target_kou2

        item = argmax(rank)
        target_kaku3, target_kou3 = get_property(item, sentence_length)
        rank[item] = -2
        while is_on_restricted([target_kou1, target_kou2], target_kou3) or target_kaku3 in [target_kaku1, target_kaku2]:
            item = argmax(rank)
            target_kaku3, target_kou3 = get_property(item, sentence_length)
            rank[item] = -2
        ret[target_kaku3] = target_kou3

        ga_pred.append(ret[1] % sentence_length)
        ni_pred.append(ret[2] % sentence_length)
        wo_pred.append(ret[0] % sentence_length)
    return ga_pred, ni_pred, wo_pred


def get_ordered_prediction(batch_ga_pred, batch_ni_pred, batch_wo_pred):
    ga_pred = []
    ni_pred = []
    wo_pred = []

    batch_ga_pred[batch_ga_pred != batch_ga_pred] = 0
    batch_ni_pred[batch_ni_pred != batch_ni_pred] = 0
    batch_wo_pred[batch_wo_pred != batch_wo_pred] = 0

    for sentence_ga, sentence_ni, sentence_wo in zip(batch_ga_pred, batch_ni_pred, batch_wo_pred):
        def is_on_restricted(indexes, target):
            if target == 0:
                return False
            for index in indexes:
                if target == index:
                    return True
            return False
        wo_index = argmax(sentence_wo.cpu())

        ga_index = argmax(sentence_ga.cpu())
        while is_on_restricted([wo_index], ga_index):
            sentence_ga[ga_index] = -2
            ga_index = argmax(sentence_ga.cpu())

        ni_index = argmax(sentence_ni.cpu())
        while is_on_restricted([wo_index, ga_index], ni_index):
            sentence_ni[ni_index] = -2
            ni_index = argmax(sentence_ni.cpu())

        ga_pred.append(int(ga_index))
        ni_pred.append(int(ni_index))
        wo_pred.append(int(wo_index))
    return ga_pred, ni_pred, wo_pred


def get_no_decode_prediction(batch_ga_pred, batch_ni_pred, batch_wo_pred):
    ga_pred = []
    ni_pred = []
    wo_pred = []
    batch_ga_pred[batch_ga_pred != batch_ga_pred] = 0
    batch_ni_pred[batch_ni_pred != batch_ni_pred] = 0
    batch_wo_pred[batch_wo_pred != batch_wo_pred] = 0
    for sentence_ga, sentence_ni, sentence_wo in zip(batch_ga_pred, batch_ni_pred, batch_wo_pred):
        ga_index = argmax(sentence_ga.cpu())
        ni_index = argmax(sentence_ni.cpu())
        wo_index = argmax(sentence_wo.cpu())
        ga_pred.append(int(ga_index))
        ni_pred.append(int(ni_index))
        wo_pred.append(int(wo_index))
    return ga_pred, ni_pred, wo_pred


def get_hungarian_prediction(batch_ga_pred, batch_ni_pred, batch_wo_pred):
    ga_pred = []
    ni_pred = []
    wo_pred = []
    batch_ga_pred[batch_ga_pred != batch_ga_pred] = 0
    batch_ni_pred[batch_ni_pred != batch_ni_pred] = 0
    batch_wo_pred[batch_wo_pred != batch_wo_pred] = 0
    m = Munkres()
    for sentence_ga, sentence_ni, sentence_wo in zip(batch_ga_pred, batch_ni_pred, batch_wo_pred):
        sentence_ga *= -1
        sentence_ni *= -1
        sentence_wo *= -1
        sentence_ga = sentence_ga.tolist()
        sentence_ni = sentence_ni.tolist()
        sentence_wo = sentence_wo.tolist()
        sentence = []
        sentence.extend(sentence_ga)
        sentence.extend(sentence_ni)
        sentence.extend(sentence_wo)
        min_val = min(sentence)

        concat_sentence = [sentence_ga, sentence_ni, sentence_wo]
        while len(concat_sentence) < len(concat_sentence[0]):
            concat_sentence.append([min_val] * len(concat_sentence[0]))
        rets = m.compute(concat_sentence)
        for item in rets:
            if item[0] == 0:
                ga_pred.append(int(item[1]))
            elif item[0] == 1:
                ni_pred.append(int(item[1]))
            elif item[0] == 2:
                wo_pred.append(int(item[1]))
    return ga_pred, ni_pred, wo_pred


if __name__ == "__main__":
    gold_ga = torch.Tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
    gold_ni = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    gold_wo = torch.Tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    pred_ga = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, 1], [0.0, 4, 2, 1]])
    pred_ni = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, -1], [0.0, -4, -2, -1]])
    pred_wo = torch.Tensor([[0.0, -2, -3, -1], [0.0, -4, 2, 1], [0.0, -4, 2, 1]])
    ret_ga, ret_ni, ret_wo = get_ordered_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 3, 1]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 2, 2]

    pred_ga = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, 1], [0.0, 4, 2, 1]])
    pred_ni = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, -1], [0.0, -4, -2, -1]])
    pred_wo = torch.Tensor([[0.0, -2, -3, -1], [0.0, -4, 2, 1], [0.0, -4, 2, 1]])
    ret_ga, ret_ni, ret_wo = get_restricted_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 3, 1]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 2, 2]

    pred_ga = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, 1], [0.0, 4, 2, 1]])
    pred_ni = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, -1], [0.0, -4, -2, -1]])
    pred_wo = torch.Tensor([[0.0, -2, -3, -1], [0.0, -4, 2, 1], [0.0, -4, 2, 1]])
    ret_ga, ret_ni, ret_wo = get_no_decode_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 2, 1]
    assert list(ret_ni) == [0, 2, 0]
    assert list(ret_wo) == [0, 2, 2]

    pred_ga = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, 1], [0.0, 4, 2, 1]])
    pred_ni = torch.Tensor([[0.0, -3, -2, -1], [0.0, -4, 2, -1], [0.0, -4, -2, -1]])
    pred_wo = torch.Tensor([[0.0, -2, -3, -1], [0.0, -4, 2, 1], [0.0, -4, 2, 1]])
    ret_ga, ret_ni, ret_wo = get_hungarian_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 2, 1]
    assert list(ret_ni) == [0, 2, 0]
    assert list(ret_wo) == [0, 2, 2]

    pred_ga = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_ni = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_wo = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    assert torch.isnan(pred_ga).any()
    ret_ga, ret_ni, ret_wo = get_ordered_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 0, 0]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 0, 0]

    pred_ga = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_ni = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_wo = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    ret_ga, ret_ni, ret_wo = get_restricted_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 0, 0]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 0, 0]

    pred_ga = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_ni = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_wo = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    ret_ga, ret_ni, ret_wo = get_no_decode_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 0, 0]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 0, 0]

    pred_ga = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_ni = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    pred_wo = torch.Tensor([[0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')], [0.0, float('NaN'), float('NaN'), float('NaN')]])
    ret_ga, ret_ni, ret_wo = get_hungarian_prediction(pred_ga, pred_ni, pred_wo)
    assert list(ret_ga) == [0, 0, 0]
    assert list(ret_ni) == [0, 0, 0]
    assert list(ret_wo) == [0, 0, 0]