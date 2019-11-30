import os
import sys
import collections
from collections import defaultdict

sys.path.append('../')
sys.path.append(os.pardir)
from utils.Datasets import get_datasets, get_datasets_in_sentences, get_datasets_in_sentences_test

TRAIN = "train"
DEV = "dev"
TEST = "test"

# train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test(TRAIN)
# dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test(DEV)
# test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test(TEST)
train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN)
dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV)
test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST)

def get_unique_list(seq):
    seen = []
    for sentence in seq:
        sentence = sentence.tolist()
        if not sentence in seen:
            seen.append(sentence)
    return seen

train_sentence_unique = get_unique_list(train_args)
dev_sentence_unique = get_unique_list(dev_args)
test_sentence_unique = get_unique_list(test_args)


def get_number(batch_labels, batch_props):
    ret_ga = 0
    ret_ni = 0
    ret_wo = 0
    ret_el = 0
    for labels, props in zip(batch_labels, batch_props):
        for label, prop in zip(labels, props):
            if prop == 'dep' or prop == 'rentai' or prop == 'zero':
                if label == 0:
                    ret_ga += 1
                elif label == 1:
                    ret_wo += 1
                elif label == 2:
                    ret_ni += 1
                elif label == 3:
                    ret_el += 1
    return ret_ga, ret_wo, ret_ni, ret_el


train_ga, train_wo, train_ni, train_el = get_number(train_label, train_prop)
dev_ga, dev_wo, dev_ni, dev_el = get_number(dev_label, dev_prop)
test_ga, test_wo, test_ni, test_el = get_number(test_label, test_prop)
print('[Train] number of sentence: {} / number of predicates:  {} / number of ga: {} / number of wo: {} / number of ni: {} / number of else: {}'.format(len(train_sentence_unique), len(train_args), train_ga, train_wo, train_ni, train_el))
print('[Dev] number of sentence: {} / number of predicates:  {} / number of ga: {} / number of wo: {} / number of ni: {} / number of else: {}'.format(len(dev_sentence_unique), len(dev_args), dev_ga, dev_wo, dev_ni, dev_el))
print('[Test] number of sentence: {} / number of predicates:  {} / number of ga: {} / number of wo: {} / number of ni: {} / number of else: {}'.format(len(test_sentence_unique), len(test_args), test_ga, test_wo, test_ni, test_el))


def get_zerodep(batch_labels, batch_props):
    ret_ga = defaultdict(int)
    ret_wo = defaultdict(int)
    ret_ni = defaultdict(int)
    for labels, props in zip(batch_labels, batch_props):
        for label, prop in zip(labels, props):
            if label != 3:
                if label == 0:
                    ret_ga[prop] += 1
                elif label == 1:
                    ret_wo[prop] += 1
                elif label == 2:
                    ret_ni[prop] += 1
    return ret_ga, ret_wo, ret_ni


train_ga, train_wo, train_ni = get_zerodep(train_label, train_prop)
dev_ga, dev_wo, dev_ni = get_zerodep(dev_label, dev_prop)
test_ga, test_wo, test_ni = get_zerodep(test_label, test_prop)
print('[Train]')
print('total: {}| total ga_dep: {} | ga_dep: {} / ga_rentai:  {} | ga_zero: {}'.format(train_ga['dep'] + train_ga['rentai'] + train_ga['zero'], train_ga['dep'] + train_ga['rentai'], train_ga['dep'], train_ga['rentai'], train_ga['zero']))
print('total: {}| total wo_dep: {} | wo_dep: {} / wo_rentai:  {} | wo_zero: {}'.format(train_wo['dep'] + train_wo['rentai'] + train_wo['zero'], train_wo['dep'] + train_wo['rentai'], train_wo['dep'], train_wo['rentai'], train_wo['zero']))
print('total: {}| total ni_dep: {} | ni_dep: {} / ni_rentai:  {} | ni_zero: {}'.format(train_ni['dep'] + train_ni['rentai'] + train_ni['zero'], train_ni['dep'] + train_ni['rentai'], train_ni['dep'], train_ni['rentai'], train_ni['zero']))
print('[Dev]')
print('total: {}| total ga_dep: {} | ga_dep: {} / ga_rentai:  {} | ga_zero: {}'.format(dev_ga['dep'] + dev_ga['rentai'] + dev_ga['zero'], dev_ga['dep'] + dev_ga['rentai'], dev_ga['dep'], dev_ga['rentai'], dev_ga['zero']))
print('total: {}| total wo_dep: {} | wo_dep: {} / wo_rentai:  {} | wo_zero: {}'.format(dev_wo['dep'] + dev_wo['rentai'] + dev_wo['zero'], dev_wo['dep'] + dev_wo['rentai'], dev_wo['dep'], dev_wo['rentai'], dev_wo['zero']))
print('total: {}| total ni_dep: {} | ni_dep: {} / ni_rentai:  {} | ni_zero: {}'.format(dev_ni['dep'] + dev_ni['rentai'] + dev_ni['zero'], dev_ni['dep'] + dev_ni['rentai'], dev_ni['dep'], dev_ni['rentai'], dev_ni['zero']))
print('[Test]')
print('total: {}| total ga_dep: {} | ga_dep: {} / ga_rentai:  {} | ga_zero: {}'.format(test_ga['dep'] + test_ga['rentai'] + test_ga['zero'], test_ga['dep'] + test_ga['rentai'], test_ga['dep'], test_ga['rentai'], test_ga['zero']))
print('total: {}| total wo_dep: {} | wo_dep: {} / wo_rentai:  {} | wo_zero: {}'.format(test_wo['dep'] + test_wo['rentai'] + test_wo['zero'], test_wo['dep'] + test_wo['rentai'], test_wo['dep'], test_wo['rentai'], test_wo['zero']))
print('total: {}| total ni_dep: {} | ni_dep: {} / ni_rentai:  {} | ni_zero: {}'.format(test_ni['dep'] + test_ni['rentai'] + test_ni['zero'], test_ni['dep'] + test_ni['rentai'], test_ni['dep'], test_ni['rentai'], test_ni['zero']))


'''
[Train] number of sentence: 23218 / number of predicates:  62489 / number of ga: 49161 / number of wo: 26799 / number of ni: 6214 / number of else: 1903594
[Dev] number of sentence: 4628 / number of predicates:  12724 / number of ga: 10072 / number of wo: 5499 / number of ni: 1749 / number of else: 397177
[Test] number of sentence: 8816 / number of predicates:  23981 / number of ga: 18994 / number of wo: 10315 / number of ni: 2758 / number of else: 742462
[Train]
total: 49161| total ga_dep: 37615 | ga_dep: 28498 / ga_rentai:  9117 | ga_zero: 11546
total: 26799| total wo_dep: 24997 | wo_dep: 23183 / wo_rentai:  1814 | wo_zero: 1802
total: 6214| total ni_dep: 5855 | ni_dep: 5645 / ni_rentai:  210 | ni_zero: 359
[Dev]
total: 10072| total ga_dep: 7520 | ga_dep: 5793 / ga_rentai:  1727 | ga_zero: 2552
total: 5499| total wo_dep: 5105 | wo_dep: 4704 / wo_rentai:  401 | wo_zero: 394
total: 1749| total ni_dep: 1637 | ni_dep: 1565 / ni_rentai:  72 | ni_zero: 112
[Test]
total: 18994| total ga_dep: 14230 | ga_dep: 10793 / ga_rentai:  3437 | ga_zero: 4764
total: 10315| total wo_dep: 9532 | wo_dep: 8861 / wo_rentai:  671 | wo_zero: 783
total: 2758| total ni_dep: 2547 | ni_dep: 2408 / ni_rentai:  139 | ni_zero: 211
'''