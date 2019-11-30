import pandas as pd
from pathlib import Path

# lstm_log_path = Path("../../results/packed/lstm/detaillog_lstm.txt")
# argmax_log_path = Path("../../results/packed/global_argmax/detaillog_global_argmax.txt")
lstm_log_path = Path("../../results/testresult_acl2019/detaillog_lstm.txt")
argmax_log_path = Path("../../results/testresult_acl2019/detaillog_global_argmax.txt")
ordered_log_path = Path("../../results/packed/ordered/detaillog_ordered.txt")
no_decode_log_path = Path("../../results/packed/no_decode/detaillog_no_decode.txt")

with lstm_log_path.open("r", encoding="utf-8") as f:
    lstm_log = pd.read_csv(f)
with argmax_log_path.open("r", encoding="utf-8") as f:
    argmax_log = pd.read_csv(f)
with ordered_log_path.open("r", encoding="utf-8") as f:
    ordered_log = pd.read_csv(f)
with no_decode_log_path.open("r", encoding="utf-8") as f:
    no_decode_log = pd.read_csv(f)

sentence_data = lstm_log['sentence']
log = lstm_log.drop(["lstm1", "lstm2", "lstm3", "lstm4", "lstm5", "sentence"], axis=1).rename(columns={'combined': 'lstm'})
log = log.join([argmax_log.rename(columns={'combined': 'global_argmax'})["global_argmax"]])
log = log.join([ordered_log.rename(columns={'combined': 'ordered'})["ordered"]])
log = log.join([no_decode_log.rename(columns={'combined': 'no_decode'})["no_decode"]])
log = log.join([no_decode_log.rename(columns={'conflict': 'no_decode_conflicted'})["no_decode_conflicted"]])
log = log.join(sentence_data)

print("conflicted: {}".format(log.query('no_decode_conflicted == True').shape[0]))
print("----------")
print("all correct: {}".format(log.query('label == lstm and label == global_argmax and label == ordered and label == no_decode').shape[0]))
print("all incorrect: {}".format(log.query('label != lstm and label != global_argmax and label != ordered and label != no_decode').shape[0]))
print("----------")
print("num of 4: {} / {} / {} / {}".format(log.query('lstm == 4').shape[0], log.query('global_argmax == 4').shape[0], log.query('ordered == 4').shape[0], log.query('no_decode == 4').shape[0]))
print("all are 4: {}".format(log.query('lstm == 4 and global_argmax == 4 and ordered == 4 and no_decode == 4').shape[0]))
print("only one is 4: {} / {} / {} / {}".format(log.query('lstm == 4 and global_argmax != 4 and ordered != 4 and no_decode != 4').shape[0],
                                                log.query('lstm != 4 and global_argmax == 4 and ordered != 4 and no_decode != 4').shape[0],
                                                log.query('lstm != 4 and global_argmax != 4 and ordered == 4 and no_decode != 4').shape[0],
                                                log.query('lstm != 4 and global_argmax != 4 and ordered != 4 and no_decode == 4').shape[0]))
print('two models are 4: {} / {} / {} / {}'.format(log.query('lstm == 4 and global_argmax == 4 and ordered == 4 and no_decode != 4').shape[0],
                                                   log.query('lstm == 4 and global_argmax == 4 and ordered != 4 and no_decode == 4').shape[0],
                                                   log.query('lstm == 4 and global_argmax != 4 and ordered == 4 and no_decode == 4').shape[0],
                                                   log.query('lstm != 4 and global_argmax == 4 and ordered == 4 and no_decode == 4').shape[0]))
print("----------")
print("only correct in lstm w/ all: {}".format(log.query('label == lstm and label != global_argmax and label != ordered and label != no_decode').shape[0]))
print("only correct in global_argmax w/ all: {}".format(log.query('label != lstm and label == global_argmax and label != ordered and label != no_decode').shape[0]))
print("only correct in ordered w/ all: {}".format(log.query('label != lstm and label != global_argmax and label == ordered and label != no_decode').shape[0]))
print("only correct in no_decode w/ all: {}".format(log.query('label != lstm and label != global_argmax and label != ordered and label == no_decode').shape[0]))
print("----------")
print("only incorrect in lstm: {}".format(log.query('label != lstm and label == global_argmax and label == ordered and label == no_decode').shape[0]))
print("only incorrect in global_argmax: {}".format(log.query('label == lstm and label != global_argmax and label == ordered and label == no_decode').shape[0]))
print("only incorrect in ordered: {}".format(log.query('label == lstm and label == global_argmax and label != ordered and label == no_decode').shape[0]))
print("only incorrect in no_decode: {}".format(log.query('label == lstm and label == global_argmax and label == ordered and label != no_decode').shape[0]))
print("----------")
print("only correct in global_argmax w/ lstm only: {}".format(log.query('label != lstm and label == global_argmax').shape[0]))
print("only correct in ordered w/ lstm only: {}".format(log.query('label != lstm and label == ordered').shape[0]))
print("only correct in no_decode w/ no_decode only: {}".format(log.query('label != lstm and label == no_decode').shape[0]))
print("only incorrect in global_argmax w/ lstm only: {}".format(log.query('label == lstm and label != global_argmax').shape[0]))
print("only incorrect in ordered w/ lstm only: {}".format(log.query('label == lstm and label != ordered').shape[0]))
print("only incorrect in no_decode w/ no_decode only: {}".format(log.query('label == lstm and label != no_decode').shape[0]))
print("----------")
print("only correct in lstm w/ global_argmax only: {}".format(log.query('label != global_argmax and label == lstm').shape[0]))
print("only correct in ordered w/ global_argmax only: {}".format(log.query('label != global_argmax and label == ordered').shape[0]))
print("only correct in no_decode w/ global_argmax only: {}".format(log.query('label != global_argmax and label == no_decode').shape[0]))
print("only incorrect in lstm w/ global_argmax only: {}".format(log.query('label == global_argmax and label != lstm').shape[0]))
print("only incorrect in ordered w/ global_argmax only: {}".format(log.query('label == global_argmax and label != ordered').shape[0]))
print("only incorrect in no_decode w/ global_argmax only: {}".format(log.query('label == global_argmax and label != no_decode').shape[0]))
# print("----------")
# print("only correct in lstm w/ ordered only: {}".format(log.query('label != ordered and label == lstm').shape[0]))
# print("only correct in global_argmax w/ ordered only: {}".format(log.query('label != ordered and label == global_argmax').shape[0]))
# print("only correct in no_decode w/ ordered only: {}".format(log.query('label != ordered and label == no_decode').shape[0]))
# print("only incorrect in lstm w/ ordered only: {}".format(log.query('label == ordered and label != lstm').shape[0]))
# print("only incorrect in global_argmax w/ ordered only: {}".format(log.query('label == ordered and label != global_argmax').shape[0]))
# print("only incorrect in no_decode w/ ordered only: {}".format(log.query('label == ordered and label != no_decode').shape[0]))
# print("----------")
# print("only correct in lstm w/ no_decode only: {}".format(log.query('label != no_decode and label == lstm').shape[0]))
# print("only correct in global_argmax w/ no_decode only: {}".format(log.query('label != no_decode and label == global_argmax').shape[0]))
# print("only correct in ordered w/ no_decode only: {}".format(log.query('label != no_decode and label == ordered').shape[0]))
# print("only incorrect in lstm w/ no_decode only: {}".format(log.query('label == no_decode and label != lstm').shape[0]))
# print("only incorrect in global_argmax w/ no_decode only: {}".format(log.query('label == no_decode and label != global_argmax').shape[0]))
# print("only incorrect in ordered w/ no_decode only: {}".format(log.query('label == no_decode and label != ordered').shape[0]))

np_index = log.index.values
np_label = log['label'].values
np_lstm = log['lstm'].values
np_argmax = log['global_argmax'].values
np_ordered = log['ordered'].values
np_no_decode = log['no_decode'].values
np_sentence = log['sentence'].values
np_pred = log['pred'].values


def get_counts(index, labels, lstms, pointers):
    ret_labels = [0, 0, 0, 0, 0]
    ret_lstms = [0, 0, 0, 0, 0]
    ret_pointers = [0, 0, 0, 0, 0]

    lstm_correct = True
    pointer_correct = True
    both_correct = True

    for label, lstm, pointer in zip(labels, lstms, pointers):
        ret_labels[label] += 1
        ret_lstms[lstm] += 1
        ret_pointers[pointer] += 1
        if lstm != label:
            lstm_correct = False
        if pointer != label:
            pointer_correct = False
        if lstm != label or pointer != label:
            both_correct = False

    lstm_wrong_word = False
    lstm_wrong_other_kaku = False
    lstm_wrong_same_kaku = False
    lstm_missed = False
    lstm_ignored = False
    if ret_lstms[-1] > 0:
        lstm_ignored = True
    else:
        for ret_label, ret_lstm in zip(ret_labels[0:3], ret_lstms[0:3]):
            if ret_label > 0 and ret_lstm == 0:
                lstm_missed = True
            if ret_label < ret_lstm and ret_label > 0:
                lstm_wrong_same_kaku = True
            if ret_label == 0 and ret_lstm > 0:
                lstm_wrong_other_kaku = True
        if not lstm_correct and not lstm_wrong_same_kaku and not lstm_wrong_other_kaku:
            lstm_wrong_word = True

    pointer_wrong_word = False
    pointer_wrong_other_kaku = False
    pointer_wrong_same_kaku = False
    pointer_missed = False
    pointer_ignored = False
    if ret_pointers[-1] > 0:
        pointer_ignored = True
    else:
        for ret_label, ret_pointer in zip(ret_labels[0:3], ret_pointers[0:3]):
            if ret_label > 0 and ret_pointer == 0:
                pointer_missed = True
            if ret_label < ret_pointer and ret_label > 0:
                pointer_wrong_same_kaku = True
            if ret_label == 0 and ret_pointer > 0:
                pointer_wrong_other_kaku = True
        if not pointer_correct and not pointer_wrong_same_kaku and not pointer_wrong_other_kaku:
            pointer_wrong_word = True

    ret_labels = [str(item) for item in ret_labels]
    ret_lstms = [str(item) for item in ret_lstms]
    ret_pointers = [str(item) for item in ret_pointers]

    return ', '.join([str(both_correct), str(lstm_correct), str(pointer_correct),
                      str(lstm_wrong_word), str(lstm_wrong_same_kaku), str(lstm_wrong_other_kaku), str(lstm_missed), str(lstm_ignored),
                      str(pointer_wrong_word), str(pointer_wrong_same_kaku), str(pointer_wrong_other_kaku), str(pointer_missed), str(pointer_ignored),
                      str(index)] + ret_labels + ret_lstms + ret_pointers)

start_i = 0
bef_pred = np_pred[0]
bef_sentence = np_sentence[0]
rets = []
for i, (pred, sentence) in enumerate(zip(np_pred, np_sentence)):
    if bef_pred != pred or bef_sentence != sentence:
        ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:i], np_lstm[start_i:i], np_argmax[start_i:i])
        rets.append(ret)
        bef_pred = pred
        bef_sentence = sentence
        start_i = i
else:
    ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:], np_lstm[start_i:], np_argmax[start_i:])
    rets.append(ret)

path = Path('./analized_result_argmax.csv')
with path.open('w', encoding='utf-8') as f:
    f.writelines('\n'.join(rets))

# start_i = 0
# bef_pred = np_pred[0]
# bef_sentence = np_sentence[0]
# rets = []
# for i, (pred, sentence) in enumerate(zip(np_pred, np_sentence)):
#     if bef_pred != pred or bef_sentence != sentence:
#         ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:i], np_lstm[start_i:i], np_ordered[start_i:i])
#         rets.append(ret)
#         bef_pred = pred
#         bef_sentence = sentence
#         start_i = i
# else:
#     ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:], np_lstm[start_i:], np_ordered[start_i:])
#     rets.append(ret)
#
# path = Path('./analized_result_ordered.csv')
# with path.open('w', encoding='utf-8') as f:
#     f.writelines('\n'.join(rets))

# start_i = 0
# bef_pred = np_pred[0]
# bef_sentence = np_sentence[0]
# rets = []
# for i, (pred, sentence) in enumerate(zip(np_pred, np_sentence)):
#     if bef_pred != pred or bef_sentence != sentence:
#         ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:i], np_lstm[start_i:i], np_ordered[start_i:i])
#         rets.append(ret)
#         bef_pred = pred
#         bef_sentence = sentence
#         start_i = i
# else:
#     ret = bef_sentence +', ' + bef_pred +', ' + get_counts(start_i, np_label[start_i:], np_lstm[start_i:], np_ordered[start_i:])
#     rets.append(ret)
#
# path = Path('./analized_result_no_decode.csv')
# with path.open('w', encoding='utf-8') as f:
#     f.writelines('\n'.join(rets))


'''
https://docs.google.com/spreadsheets/d/1N6fS5SnKMw34vwp3t90XJ-em_GjrCh_UR4OiBeBqLtI/edit

conflicted: 0
----------
all correct: 836266
all incorrect: 7919
----------
num of 4: 4789 / 4564 / 4468 / 4458
all are 4: 706
only one is 4: 2867 / 1529 / 5 / 6
two models are 4: 6 / 2 / 561 / 1671
----------
only correct in lstm w/ all: 1949
only correct in global_argmax w/ all: 554
only correct in ordered w/ all: 2
only correct in no_decode w/ all: 4
----------
only incorrect in lstm: 2591
only incorrect in global_argmax: 680
only incorrect in ordered: 3
only incorrect in no_decode: 4
----------
only correct in global_argmax w/ lstm only: 3150
only correct in ordered w/ lstm only: 3190
only correct in no_decode w/ no_decode only: 3195
only incorrect in global_argmax w/ lstm only: 2632
only incorrect in ordered w/ lstm only: 2836
only incorrect in no_decode w/ no_decode only: 2834
----------
only correct in lstm w/ global_argmax only: 2632
only correct in ordered w/ global_argmax only: 1278
only correct in no_decode w/ global_argmax only: 1283
only incorrect in lstm w/ global_argmax only: 3150
only incorrect in ordered w/ global_argmax only: 1442
only incorrect in no_decode w/ global_argmax only: 1440
----------
only correct in lstm w/ ordered only: 2836
only correct in global_argmax w/ ordered only: 1442
only correct in no_decode w/ ordered only: 14
only incorrect in lstm w/ ordered only: 3190
only incorrect in global_argmax w/ ordered only: 1278
only incorrect in no_decode w/ ordered only: 7
----------
only correct in lstm w/ no_decode only: 2834
only correct in global_argmax w/ no_decode only: 1440
only correct in ordered w/ no_decode only: 7
only incorrect in lstm w/ no_decode only: 3195
only incorrect in global_argmax w/ no_decode only: 1283
only incorrect in ordered w/ no_decode only: 14
'''