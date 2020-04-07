import os
import sys
import collections
from collections import defaultdict

sys.path.append('../')
sys.path.append(os.pardir)
from utils.Datasets import get_datasets, get_datasets_in_sentences, get_datasets_in_sentences_test

TRAIN = "train2"
DEV = "dev"
TEST = "test"

with_bccwj = True
with_bert = True

# train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test(TRAIN)
# dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test(DEV)
# test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test(TEST)
train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN, with_bccwj=with_bccwj, with_bert=with_bert)
dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV, with_bccwj=with_bccwj, with_bert=with_bert)
test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST, with_bccwj=with_bccwj, with_bert=with_bert)

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
            # if prop == 'dep' or prop == 'rentai' or prop == 'zero':
            if prop != 0:
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
* ntc
[Train] number of sentence: 23218 / number of predicates:  62489 / number of ga: 50890 / number of wo: 27216 / number of ni: 6701 / number of else: 2102725
[Dev] number of sentence: 4628 / number of predicates:  12724 / number of ga: 10121 / number of wo: 5532 / number of ni: 1850 / number of else: 437920
[Test] number of sentence: 8816 / number of predicates:  23981 / number of ga: 19064 / number of wo: 10372 / number of ni: 2890 / number of else: 819131
[Train]
total: 50722| total ga_dep: 37660 | ga_dep: 28537 / ga_rentai:  9123 | ga_zero: 13062
total: 27092| total wo_dep: 25019 | wo_dep: 23199 / wo_rentai:  1820 | wo_zero: 2073
total: 6370| total ni_dep: 5861 | ni_dep: 5650 / ni_rentai:  211 | ni_zero: 509
[Dev]
total: 10072| total ga_dep: 7520 | ga_dep: 5793 / ga_rentai:  1727 | ga_zero: 2552
total: 5499| total wo_dep: 5105 | wo_dep: 4704 / wo_rentai:  401 | wo_zero: 394
total: 1749| total ni_dep: 1637 | ni_dep: 1565 / ni_rentai:  72 | ni_zero: 112
[Test]
total: 18994| total ga_dep: 14230 | ga_dep: 10793 / ga_rentai:  3437 | ga_zero: 4764
total: 10315| total wo_dep: 9532 | wo_dep: 8861 / wo_rentai:  671 | wo_zero: 783
total: 2758| total ni_dep: 2547 | ni_dep: 2408 / ni_rentai:  139 | ni_zero: 211

* ntc bert
[Train] number of sentence: 23218 / number of predicates:  62489 / number of ga: 50890 / number of wo: 27216 / number of ni: 6701 / number of else: 2102725
[Dev] number of sentence: 4628 / number of predicates:  12724 / number of ga: 10121 / number of wo: 5532 / number of ni: 1850 / number of else: 437920
[Test] number of sentence: 8816 / number of predicates:  23981 / number of ga: 19064 / number of wo: 10372 / number of ni: 2890 / number of else: 819131
[Train]
total: 50722| total ga_dep: 37660 | ga_dep: 28537 / ga_rentai:  9123 | ga_zero: 13062
total: 27092| total wo_dep: 25019 | wo_dep: 23199 / wo_rentai:  1820 | wo_zero: 2073
total: 6370| total ni_dep: 5861 | ni_dep: 5650 / ni_rentai:  211 | ni_zero: 509
[Dev]
total: 10072| total ga_dep: 7520 | ga_dep: 5793 / ga_rentai:  1727 | ga_zero: 2552
total: 5499| total wo_dep: 5105 | wo_dep: 4704 / wo_rentai:  401 | wo_zero: 394
total: 1749| total ni_dep: 1637 | ni_dep: 1565 / ni_rentai:  72 | ni_zero: 112
[Test]
total: 18994| total ga_dep: 14230 | ga_dep: 10793 / ga_rentai:  3437 | ga_zero: 4764
total: 10315| total wo_dep: 9532 | wo_dep: 8861 / wo_rentai:  671 | wo_zero: 783
total: 2758| total ni_dep: 2547 | ni_dep: 2408 / ni_rentai:  139 | ni_zero: 211

* bccwj
[Train] number of sentence: 30056 / number of predicates:  78268 / number of ga: 56391 / number of wo: 31580 / number of ni: 13628 / number of else: 2597212
[Dev] number of sentence: 6373 / number of predicates:  15430 / number of ga: 10779 / number of wo: 6054 / number of ni: 2839 / number of else: 472432
[Test] number of sentence: 5809 / number of predicates:  14582 / number of ga: 10312 / number of wo: 5775 / number of ni: 2430 / number of else: 491575
[Train]
total: 56391| total ga_dep: 41937 | ga_dep: 30735 / ga_rentai:  11202 | ga_zero: 14454
total: 31580| total wo_dep: 28388 | wo_dep: 26040 / wo_rentai:  2348 | wo_zero: 3192
total: 13628| total ni_dep: 12501 | ni_dep: 11812 / ni_rentai:  689 | ni_zero: 1127
[Dev]
total: 10779| total ga_dep: 8049 | ga_dep: 6107 / ga_rentai:  1942 | ga_zero: 2730
total: 6054| total wo_dep: 5351 | wo_dep: 4926 / wo_rentai:  425 | wo_zero: 703
total: 2839| total ni_dep: 2588 | ni_dep: 2446 / ni_rentai:  142 | ni_zero: 251
[Test]
total: 10312| total ga_dep: 7623 | ga_dep: 5648 / ga_rentai:  1975 | ga_zero: 2689
total: 5775| total wo_dep: 5143 | wo_dep: 4702 / wo_rentai:  441 | wo_zero: 632
total: 2430| total ni_dep: 2171 | ni_dep: 2054 / ni_rentai:  117 | ni_zero: 259

* bccwj bert
[Train] number of sentence: 30054 / number of predicates:  78223 / number of ga: 56355 / number of wo: 31558 / number of ni: 13619 / number of else: 2586739
[Dev] number of sentence: 6372 / number of predicates:  15426 / number of ga: 10777 / number of wo: 6052 / number of ni: 2839 / number of else: 471513
[Test] number of sentence: 5808 / number of predicates:  14550 / number of ga: 10282 / number of wo: 5764 / number of ni: 2424 / number of else: 483295
[Train]
total: 56355| total ga_dep: 41919 | ga_dep: 30718 / ga_rentai:  11201 | ga_zero: 14436
total: 31558| total wo_dep: 28372 | wo_dep: 26026 / wo_rentai:  2346 | wo_zero: 3186
total: 13619| total ni_dep: 12494 | ni_dep: 11806 / ni_rentai:  688 | ni_zero: 1125
[Dev]
total: 10777| total ga_dep: 8048 | ga_dep: 6107 / ga_rentai:  1941 | ga_zero: 2729
total: 6052| total wo_dep: 5349 | wo_dep: 4924 / wo_rentai:  425 | wo_zero: 703
total: 2839| total ni_dep: 2588 | ni_dep: 2446 / ni_rentai:  142 | ni_zero: 251
[Test]
total: 10282| total ga_dep: 7611 | ga_dep: 5640 / ga_rentai:  1971 | ga_zero: 2671
total: 5764| total wo_dep: 5134 | wo_dep: 4694 / wo_rentai:  440 | wo_zero: 630
total: 2424| total ni_dep: 2165 | ni_dep: 2048 / ni_rentai:  117 | ni_zero: 259

'''