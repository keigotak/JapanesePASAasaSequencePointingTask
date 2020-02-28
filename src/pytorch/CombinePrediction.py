import pandas as pd
from pathlib import Path
import collections

ref = Path("../../results/packed/lstm/detaillog_lstm.txt")
with ref.open("r", encoding="utf-8") as f:
    ref_data = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "pointer", "sentence"))
sentence_data = ref_data['sentence']
ref_data = ref_data.drop('pointer', axis=1)
ref_data = ref_data.drop('sentence', axis=1)

log = None
log1 = None
log2 = None
log3 = None
log4 = None
log5 = None
item1 = None
item2 = None
item3 = None
item4 = None
item5 = None
summary_conflict = []

events = ['jsai', 'pacling', 'acm']
tags = ['sl', 'spg', 'spl', 'spn']
corpus = ['ntc', 'bccwj']
model = ['glove', 'bert']

tag = events[2] + tags[0] + corpus[0] + model[0]

if tag in {'jsaislntcglove', 'paclingslntcglove', 'acmslntcglove', 'acmslntcbert', 'acmslbccwjglove', 'acmslbccwjbert'}:
    # For JSAI 2019
    if tag == 'jsaislntcglove':
        log1 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155232.txt")
        log2 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155442.txt")
        log3 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155529.txt")
        log4 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155633.txt")
        log5 = Path("../../results/packed/lstm/detaillog_lstm_20190212-170438.txt")

    # For PACLING 2019
    elif tag == 'paclingslntcglove':
        log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")

    elif tag == 'acmslntcglove':
        log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")
    elif tag == 'acmslbccwjbert':
        log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")

    elif tag == 'acmslbccwjglove':
        log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")
    elif tag == 'acmslbccwjbert':
        log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")

    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1"))
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl2"))
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl3"))
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl4"))
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl5"))

    log = data1.join([data2["sl2"], data3["sl3"], data4["sl4"], data5["sl5"]])

    item1 = log.lstm1.values
    item2 = log.lstm2.values
    item3 = log.lstm3.values
    item4 = log.lstm4.values
    item5 = log.lstm5.values
elif tag in {'jsaispgntcglove', 'jsaisplntcglove',
             'paclingspgntcglove', 'paclingsplntcglove',
             'acmspgntcglove', 'acmsplntcglove',
             'acmspgntcbert', 'acmsplntcbert',
             'acmspgbccwjglove', 'acmsplbccwjglove',
             'acmspgbccwjbert', 'acmsplbccwjbert'}:

    # For JSAI 2019
    if tag == 'jsaispgntcglove':
        log1 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165509.txt")
        log2 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165519.txt")
        log3 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165654.txt")
        log4 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165702.txt")
        log5 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165835.txt")
    # For PACLING 2019
    elif tag == 'paclingspgntcglove':
        log1 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001425.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001518.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001538.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001642.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001652.txt")
    elif tag == 'jsaisplntcglove':
        log1 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220541.txt")
        log2 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220636.txt")
        log3 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220640.txt")
        log4 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220732.txt")
        log5 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220903.txt")
    # For PACLING 2019 local argmax
    elif tag == 'paclingsplntcglove':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmspgntcglove':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmsplntcglove':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")

    elif tag == 'acmspgntcbert':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmsplntcbert':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmspgbccwjglove':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmsplbccwjglove':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmspgbccwjbert':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
    elif tag == 'acmsplbccwjbert':
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")



    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sentence", "conflict"))
        data1 = data1.drop("sentence", axis=1)
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp2", "sentence", "conflict"))
        data2 = data2.drop("sentence", axis=1)
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp3", "sentence", "conflict"))
        data3 = data3.drop("sentence", axis=1)
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp4", "sentence", "conflict"))
        data4 = data4.drop("sentence", axis=1)
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp5", "sentence", "conflict"))
        data5 = data5.drop("sentence", axis=1)

    log = data1.join([data2["sp2"], data3["sp3"], data4["sp4"], data5["sp5"]])

    item1 = log.ptr1.values
    item2 = log.ptr2.values
    item3 = log.ptr3.values
    item4 = log.ptr4.values
    item5 = log.ptr5.values

elif tag in {'jsaispnntcglove', 'paclingspnntcglove', 'acmspnntcglove', 'acmspnntcbert', 'acmspnbccwjglove', 'acmspnbccwjbert'}:
    if tag == 'jsaispnntcglove':
        log1 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-203841.txt")
        log2 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204003.txt")
        log3 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204013.txt")
        log4 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204205.txt")
        log5 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204212.txt")

    # For PACLING 2019 no restriction
    elif tag == 'paclingspnntcglove':
        log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
        log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
        log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
        log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
        log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")
    elif tag == 'acmspnntcglove':
        log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
        log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
        log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
        log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
        log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")
    elif tag == 'acmspnntcbert':
        log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
        log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
        log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
        log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
        log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")
    elif tag == 'acmspnbccwjglove':
        log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
        log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
        log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
        log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
        log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")
    elif tag == 'acmspnbccwjbert':
        log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
        log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
        log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
        log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
        log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")




    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sentence", "conflict"))
        data1 = data1.drop("sentence", axis=1)
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp2", "sentence", "conflict"))
        data2 = data2.drop("sentence", axis=1)
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp3", "sentence", "conflict"))
        data3 = data3.drop("sentence", axis=1)
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp4", "sentence", "conflict"))
        data4 = data4.drop("sentence", axis=1)
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp5", "sentence", "conflict"))
        data5 = data5.drop("sentence", axis=1)

    item1 = data1.conflict
    item2 = data2.conflict
    item3 = data3.conflict
    item4 = data4.conflict
    item5 = data5.conflict

    for idx in range(data1.shape[0]):
        items = [item1[idx], item2[idx], item3[idx], item4[idx], item5[idx]]
        c = collections.Counter(items)

        ans = 'False'
        if c['True'] >= 1:
            ans = 'True'
            break

        for key, val in c.most_common():
            if val >= 5:
                ans = key
                break
        summary_conflict.append(ans)

    log = data1.join([data2["sp2"], data3["sp3"], data4["sp4"], data5["sp5"]])

    item1 = log.ptr1.values
    item2 = log.ptr2.values
    item3 = log.ptr3.values
    item4 = log.ptr4.values
    item5 = log.ptr5.values

summary = []
for idx in range(log.shape[0]):
    items = [item1[idx], item2[idx], item3[idx], item4[idx], item5[idx]]
    c = collections.Counter(items)

    ans = 4
    for key, val in c.most_common():
        if val >= 4:
            ans = key
            break
    summary.append(ans)
log['combined'] = summary
log['sentence'] = sentence_data.reset_index(drop=True)
if tag in {'jsaispnntcglove', 'paclingspnntcglove', 'acmspnntcglove', 'acmspnntcbert', 'acmspnbccwjglove', 'acmspnbccwjbert'}:
    log.assign(conflict=summary_conflict)
    print(log.query('conflict == True').shape[0])

ret_path = str(Path('../../results/packed/detaillog_{}.txt'.format(tag)).resolve())
log.to_csv(ret_path, header=True, index=False, mode='w')
