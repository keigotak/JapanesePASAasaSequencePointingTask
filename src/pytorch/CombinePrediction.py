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

tags = ['lstm', 'global_argmax', 'ordered', 'no_decode']
tag = tags[3]

if tag == 'lstm':
    # For JSAI 2019
    # log1 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155232.txt")
    # log2 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155442.txt")
    # log3 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155529.txt")
    # log4 = Path("../../results/packed/lstm/detaillog_lstm_20190212-155633.txt")
    # log5 = Path("../../results/packed/lstm/detaillog_lstm_20190212-170438.txt")

    # For ACL SRW 2019
    log1 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014305.txt")
    log2 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014359.txt")
    log3 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014450.txt")
    log4 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014525.txt")
    log5 = Path("../../results/testresult_acl2019/detaillog_lstm_20190416-014719.txt")

    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "lstm1"))
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "lstm2"))
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "lstm3"))
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "lstm4"))
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "lstm5"))

    log = data1.join([data2["lstm2"], data3["lstm3"], data4["lstm4"], data5["lstm5"]])

    item1 = log.lstm1.values
    item2 = log.lstm2.values
    item3 = log.lstm3.values
    item4 = log.lstm4.values
    item5 = log.lstm5.values
elif tag == 'global_argmax' or tag == 'ordered':
    if tag == 'global_argmax':
        # For JSAI 2019
        # log1 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165509.txt")
        # log2 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165519.txt")
        # log3 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165654.txt")
        # log4 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165702.txt")
        # log5 = Path("../../results/packed/global_argmax/detaillog_pointer_20190214-165835.txt")
        # For ACL SRW 2019
        log1 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001425.txt")
        log2 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001518.txt")
        log3 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001538.txt")
        log4 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001642.txt")
        log5 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001652.txt")
    elif tag == 'ordered':
        # log1 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220541.txt")
        # log2 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220636.txt")
        # log3 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220640.txt")
        # log4 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220732.txt")
        # log5 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220903.txt")

        # For PACLING 2019 local argmax
        log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
        log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
        log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
        log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
        log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")

    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr1", "sentence", "conflict"))
        data1 = data1.drop("sentence", axis=1)
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr2", "sentence", "conflict"))
        data2 = data2.drop("sentence", axis=1)
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr3", "sentence", "conflict"))
        data3 = data3.drop("sentence", axis=1)
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr4", "sentence", "conflict"))
        data4 = data4.drop("sentence", axis=1)
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr5", "sentence", "conflict"))
        data5 = data5.drop("sentence", axis=1)

    log = data1.join([data2["ptr2"], data3["ptr3"], data4["ptr4"], data5["ptr5"]])

    item1 = log.ptr1.values
    item2 = log.ptr2.values
    item3 = log.ptr3.values
    item4 = log.ptr4.values
    item5 = log.ptr5.values

elif tag == 'no_decode':
    # log1 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-203841.txt")
    # log2 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204003.txt")
    # log3 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204013.txt")
    # log4 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204205.txt")
    # log5 = Path("../../results/packed/no_decode/detaillog_pointer_20190213-204212.txt")

    # For PACLING 2019 no restriction
    log1 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224115.txt")
    log2 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224102.txt")
    log3 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224113.txt")
    log4 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-224120.txt")
    log5 = Path("../../results/testresult_pacling2019/no_restriction/detaillog_pointer_20190611-231229.txt")

    with log1.open("r", encoding="utf-8") as f:
        data1 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr1", "sentence", "conflict"))
        data1 = data1.drop("sentence", axis=1)
    with log2.open("r", encoding="utf-8") as f:
        data2 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr2", "sentence", "conflict"))
        data2 = data2.drop("sentence", axis=1)
    with log3.open("r", encoding="utf-8") as f:
        data3 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr3", "sentence", "conflict"))
        data3 = data3.drop("sentence", axis=1)
    with log4.open("r", encoding="utf-8") as f:
        data4 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr4", "sentence", "conflict"))
        data4 = data4.drop("sentence", axis=1)
    with log5.open("r", encoding="utf-8") as f:
        data5 = pd.read_csv(f, names=("arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "ptr5", "sentence", "conflict"))
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

    log = data1.join([data2["ptr2"], data3["ptr3"], data4["ptr4"], data5["ptr5"]])

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
if tag == 'no_decode':
    log.assign(conflict=summary_conflict)
    print(log.query('conflict == True').shape[0])

ret_path = None
if tag == 'lstm':
    # ret_path = str(Path('../../results/packed/lstm/detaillog_lstm.txt').resolve())
    ret_path = str(Path('../../results/testresult_acl2019/detaillog_lstm.txt').resolve())
elif tag == 'global_argmax':
    # ret_path = str(Path('../../results/packed/global_argmax/detaillog_global_argmax.txt').resolve())
    ret_path = str(Path('../../results/testresult_acl2019/detaillog_global_argmax.txt').resolve())
elif tag == 'ordered':
    # ret_path = str(Path('../../results/packed/ordered/detaillog_ordered.txt').resolve())
    ret_path = str(Path('../../results/testresult_pacling2019/detaillog_ordered.txt').resolve())
elif tag == 'no_decode':
    # ret_path = str(Path('../../results/packed/no_decode/detaillog_no_decode.txt').resolve())
    ret_path = str(Path('../../results/testresult_pacling2019/detaillog_no_decode.txt').resolve())
log.to_csv(ret_path, header=True, index=False, mode='w')
