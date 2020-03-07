import pandas as pd
from pathlib import Path
import collections
import argparse



def main(tag):
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

    if tag in {'jsaislntcglove', 'paclingslntcglove', 'acmslntcglove', 'acmslbccwjglove'}:
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
        # For acm 2020
        elif tag == 'acmslntcglove':
            log1 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110202/detaillog_lstm_20190427-110202.txt")
            log2 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110222/detaillog_lstm_20190427-110222.txt")
            log3 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110231/detaillog_lstm_20190427-110231.txt")
            log4 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110302/detaillog_lstm_20190427-110302.txt")
            log5 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190427-110328/detaillog_lstm_20190427-110328.txt")
        elif tag == 'acmslbccwjglove':
            log1 = Path("../../results/pasa-lstm-20200105-035952/detaillog_lstm_20200105-035952.txt")
            log2 = Path("../../results/pasa-lstm-20200105-040001/detaillog_lstm_20200105-040001.txt")
            log3 = Path("../../results/pasa-lstm-20200105-040035/detaillog_lstm_20200105-040035.txt")
            log4 = Path("../../results/pasa-lstm-20200105-040101/detaillog_lstm_20200105-040101.txt")
            log5 = Path("../../results/pasa-lstm-20200105-040118/detaillog_lstm_20200105-040118.txt")

        with log1.open("r", encoding="utf-8") as f:
            data1 = [line.strip().split(',') for line in f.readlines()]
        with log2.open("r", encoding="utf-8") as f:
            data2 = [line.strip().split(',') for line in f.readlines()]
        with log3.open("r", encoding="utf-8") as f:
            data3 = [line.strip().split(',') for line in f.readlines()]
        with log4.open("r", encoding="utf-8") as f:
            data4 = [line.strip().split(',') for line in f.readlines()]
        with log5.open("r", encoding="utf-8") as f:
            data5 = [line.strip().split(',') for line in f.readlines()]

        if tag in ['acmslntcglove', 'acmslbccwjglove']:
            if tag in ['acmslntcglove']:
                ref = Path("../../results/packed/sentences_ntc.txt")
            else:
                ref = Path("../../results/packed/sentence_bccwj.txt")
            with ref.open("r", encoding="utf-8") as f:
                sentences = f.readlines()
            log = [i1[:7] + [i2[7]] + [i3[7]] + [i4[7]] + [i5[7]] + i1[7:] + [s[-2]] for i1, i2, i3, i4, i5, s in
                   zip(data1, data2, data3, data4, data5, sentences)]
            columns = ["arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1", "sl2", "sl3", "sl4", "sl5", "sentence"]
        else:
            log = [i1[:7] + [i2[7]] + [i3[7]] + [i4[7]] + [i5[7]] + i1[7:] for i1, i2, i3, i4, i5 in
                   zip(data1, data2, data3, data4, data5)]
            columns = ["arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1", "sl2", "sl3", "sl4", "sl5", "sentence", "conflict"]
        log = pd.DataFrame(log, columns=columns)

        item1 = log.sl1.values
        item2 = log.sl2.values
        item3 = log.sl3.values
        item4 = log.sl4.values
        item5 = log.sl5.values

    elif tag in {'acmslntcbert', 'acmslbccwjbert'}:
        # For acm 2020
        if tag == 'acmslntcbert':
            log1 = Path("../../results/pasa-bertsl-20200306-003922/detaillog_bertsl_20200306-003922.txt")
            log2 = Path("../../results/pasa-bertsl-20200306-015335/detaillog_bertsl_20200306-015335.txt")
            log3 = Path("../../results/pasa-bertsl-20200306-030810/detaillog_bertsl_20200306-030810.txt")
            log4 = Path("../../results/pasa-bertsl-20200306-042253/detaillog_bertsl_20200306-042253.txt")
            log5 = Path("../../results/pasa-bertsl-20200306-053732/detaillog_bertsl_20200306-053732.txt")
        elif tag == 'acmslbccwjbert':
            log1 = Path("../../results/pasa-bertsl-20200104-152437/detaillog_bertsl_20200104-152437.txt")
            log2 = Path("../../results/pasa-bertsl-20200104-153949/detaillog_bertsl_20200104-153949.txt")
            log3 = Path("../../results/pasa-bertsl-20200104-154415/detaillog_bertsl_20200104-154415.txt")
            log4 = Path("../../results/pasa-bertsl-20200104-154538/detaillog_bertsl_20200104-154538.txt")
            log5 = Path("../../results/pasa-bertsl-20200104-163623/detaillog_bertsl_20200104-163623.txt")

        with log1.open("r", encoding="utf-8") as f:
            data1 = [line.strip().split(',') for line in f.readlines()]
        with log2.open("r", encoding="utf-8") as f:
            data2 = [line.strip().split(',') for line in f.readlines()]
        with log3.open("r", encoding="utf-8") as f:
            data3 = [line.strip().split(',') for line in f.readlines()]
        with log4.open("r", encoding="utf-8") as f:
            data4 = [line.strip().split(',') for line in f.readlines()]
        with log5.open("r", encoding="utf-8") as f:
            data5 = [line.strip().split(',') for line in f.readlines()]

        log = [i1[:7] + [i2[7]] + [i3[7]] + [i4[7]] + [i5[7]] + i1[7:] for i1, i2, i3, i4, i5 in zip(data1, data2, data3, data4, data5)]
        log = pd.DataFrame(log, columns=["arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1", "sl2", "sl3", "sl4", "sl5", "sentence", "conflict"])

        item1 = log.sl1.values
        item2 = log.sl2.values
        item3 = log.sl3.values
        item4 = log.sl4.values
        item5 = log.sl5.values

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
        elif tag == 'jsaisplntcglove':
            log1 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220541.txt")
            log2 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220636.txt")
            log3 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220640.txt")
            log4 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220732.txt")
            log5 = Path("../../results/packed/ordered/detaillog_pointer_20190210-220903.txt")
        # For PACLING 2019
        elif tag == 'paclingspgntcglove':
            log1 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001425.txt")
            log2 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001518.txt")
            log3 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001538.txt")
            log4 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001642.txt")
            log5 = Path("../../results/testresult_acl2019/detaillog_pointer_20190415-001652.txt")
        elif tag == 'paclingsplntcglove':
            log1 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215418.txt")
            log2 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-222931.txt")
            log3 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215710.txt")
            log4 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-215849.txt")
            log5 = Path("../../results/testresult_pacling2019/local_argmax/detaillog_pointer_20190611-220002.txt")
        # For ACM TALLIP
        elif tag == 'acmspgntcglove':
            log1 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114743/detaillog_pointer_20190427-114743.txt")
            log2 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114739/detaillog_pointer_20190427-114739.txt")
            log3 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-120937/detaillog_pointer_20190427-120937.txt")
            log4 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114749/detaillog_pointer_20190427-114749.txt")
            log5 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190427-114838/detaillog_pointer_20190427-114838.txt")
        elif tag == 'acmsplntcglove':
            log1 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215418/detaillog_pointer_20190611-215418.txt")
            log2 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-222931/detaillog_pointer_20190611-222931.txt")
            log3 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215710/detaillog_pointer_20190611-215710.txt")
            log4 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-215849/detaillog_pointer_20190611-215849.txt")
            log5 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-220002/detaillog_pointer_20190611-220002.txt")
        elif tag == 'acmspgntcbert':
            log1 = Path("../../results/pasa-bertptr-20200306-003313/detaillog_bertptr_20200306-003313.txt")
            log2 = Path("../../results/pasa-bertptr-20200306-013908/detaillog_bertptr_20200306-013908.txt")
            log3 = Path("../../results/pasa-bertptr-20200306-024455/detaillog_bertptr_20200306-024455.txt")
            log4 = Path("../../results/pasa-bertptr-20200306-035112/detaillog_bertptr_20200306-035112.txt")
            log5 = Path("../../results/pasa-bertptr-20200306-045715/detaillog_bertptr_20200306-045715.txt")
        elif tag == 'acmsplntcbert':
            log1 = Path("../../results/pasa-bertptr-20200104-111614/detaillog_bertptr_20200104-111614.txt")
            log2 = Path("../../results/pasa-bertptr-20200104-111812/detaillog_bertptr_20200104-111812.txt")
            log3 = Path("../../results/pasa-bertptr-20200104-130215/detaillog_bertptr_20200104-130215.txt")
            log4 = Path("../../results/pasa-bertptr-20200104-130233/detaillog_bertptr_20200104-130233.txt")
            log5 = Path("../../results/pasa-bertptr-20200104-130251/detaillog_bertptr_20200104-130251.txt")
        elif tag == 'acmspgbccwjglove':
            log1 = Path("../../results/pasa-pointer-20200104-130555/detaillog_pointer_20200104-130555.txt")
            log2 = Path("../../results/pasa-pointer-20200104-130624/detaillog_pointer_20200104-130624.txt")
            log3 = Path("../../results/pasa-pointer-20200104-130649/detaillog_pointer_20200104-130649.txt")
            log4 = Path("../../results/pasa-pointer-20200104-150504/detaillog_pointer_20200104-150504.txt")
            log5 = Path("../../results/pasa-pointer-20200104-150511/detaillog_pointer_20200104-150511.txt")
        elif tag == 'acmsplbccwjglove':
            log1 = Path("../../results/pasa-pointer-20200104-152844/detaillog_pointer_20200104-152844.txt")
            log2 = Path("../../results/pasa-pointer-20200104-165359/detaillog_pointer_20200104-165359.txt")
            log3 = Path("../../results/pasa-pointer-20200104-165505/detaillog_pointer_20200104-165505.txt")
            log4 = Path("../../results/pasa-pointer-20200104-165511/detaillog_pointer_20200104-165511.txt")
            log5 = Path("../../results/pasa-pointer-20200104-165543/detaillog_pointer_20200104-165543.txt")
        elif tag == 'acmspgbccwjbert':
            log1 = Path("../../results/pasa-bertptr-20200204-225742/detaillog_bertptr_20200204-225742.txt")
            log2 = Path("../../results/pasa-bertptr-20200204-233732/detaillog_bertptr_20200204-233732.txt")
            log3 = Path("../../results/pasa-bertptr-20200204-000531/detaillog_bertptr_20200204-000531.txt")
            log4 = Path("../../results/pasa-bertptr-20200204-000527/detaillog_bertptr_20200204-000527.txt")
            log5 = Path("../../results/pasa-bertptr-20200204-000523/detaillog_bertptr_20200204-000523.txt")
        elif tag == 'acmsplbccwjbert':
            log1 = Path("../../results/pasa-bertptr-20200205-001801/detaillog_bertptr_20200205-001801.txt")
            log2 = Path("../../results/pasa-bertptr-20200204-004930/detaillog_bertptr_20200204-004930.txt")
            log3 = Path("../../results/pasa-bertptr-20200205-005949/detaillog_bertptr_20200205-005949.txt")
            log4 = Path("../../results/pasa-bertptr-20200204-004932/detaillog_bertptr_20200204-004932.txt")
            log5 = Path("../../results/pasa-bertptr-20200205-014243/detaillog_bertptr_20200205-014243.txt")

        with log1.open("r", encoding="utf-8") as f:
            data1 = [line.strip().split(',') for line in f.readlines()]
        with log2.open("r", encoding="utf-8") as f:
            data2 = [line.strip().split(',') for line in f.readlines()]
        with log3.open("r", encoding="utf-8") as f:
            data3 = [line.strip().split(',') for line in f.readlines()]
        with log4.open("r", encoding="utf-8") as f:
            data4 = [line.strip().split(',') for line in f.readlines()]
        with log5.open("r", encoding="utf-8") as f:
            data5 = [line.strip().split(',') for line in f.readlines()]

        log = [i1[:7] + [i2[7]] + [i3[7]] + [i4[7]] + [i5[7]] + i1[7:] for i1, i2, i3, i4, i5 in zip(data1, data2, data3, data4, data5)]
        log = pd.DataFrame(log, columns=["arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sp2", "sp3", "sp4", "sp5", "sentence", "conflict"])

        item1 = log.sp1.values
        item2 = log.sp2.values
        item3 = log.sp3.values
        item4 = log.sp4.values
        item5 = log.sp5.values

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
        # For ACM TALLIP
        elif tag == 'acmspnntcglove':
            log1 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224115/detaillog_pointer_20190611-224115.txt")
            log2 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224102/detaillog_pointer_20190611-224102.txt")
            log3 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224113/detaillog_pointer_20190611-224113.txt")
            log4 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-224120/detaillog_pointer_20190611-224120.txt")
            log5 = Path("../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190611-231229/detaillog_pointer_20190611-231229.txt")
        elif tag == 'acmspnntcbert':
            log1 = Path("../../results/pasa-bertptr-20200104-200900/detaillog_bertptr_20200104-200900.txt")
            log2 = Path("../../results/pasa-bertptr-20200104-093250/detaillog_bertptr_20200104-093250.txt")
            log3 = Path("../../results/pasa-bertptr-20200104-093509/detaillog_bertptr_20200104-093509.txt")
            log4 = Path("../../results/pasa-bertptr-20200104-111557/detaillog_bertptr_20200104-111557.txt")
            log5 = Path("../../results/pasa-bertptr-20200104-111556/detaillog_bertptr_20200104-111556.txt")
        elif tag == 'acmspnbccwjglove':
            log1 = Path("../../results/pasa-pointer-20200104-150500/detaillog_pointer_20200104-150500.txt")
            log2 = Path("../../results/pasa-pointer-20200104-150547/detaillog_pointer_20200104-150547.txt")
            log3 = Path("../../results/pasa-pointer-20200104-150552/detaillog_pointer_20200104-150552.txt")
            log4 = Path("../../results/pasa-pointer-20200104-152527/detaillog_pointer_20200104-152527.txt")
            log5 = Path("../../results/pasa-pointer-20200104-152703/detaillog_pointer_20200104-152703.txt")
        elif tag == 'acmspnbccwjbert':
            log1 = Path("../../results/pasa-bertptr-20200104-165547/detaillog_bertptr_20200104-165547.txt")
            log2 = Path("../../results/pasa-bertptr-20200104-170518/detaillog_bertptr_20200104-170518.txt")
            log3 = Path("../../results/pasa-bertptr-20200104-171529/detaillog_bertptr_20200104-171529.txt")
            log4 = Path("../../results/pasa-bertptr-20200104-172439/detaillog_bertptr_20200104-172439.txt")
            log5 = Path("../../results/pasa-bertptr-20200104-172634/detaillog_bertptr_20200104-172634.txt")

        with log1.open("r", encoding="utf-8") as f:
            data1 = [line.strip().split(',') for line in f.readlines()]
        with log2.open("r", encoding="utf-8") as f:
            data2 = [line.strip().split(',') for line in f.readlines()]
        with log3.open("r", encoding="utf-8") as f:
            data3 = [line.strip().split(',') for line in f.readlines()]
        with log4.open("r", encoding="utf-8") as f:
            data4 = [line.strip().split(',') for line in f.readlines()]
        with log5.open("r", encoding="utf-8") as f:
            data5 = [line.strip().split(',') for line in f.readlines()]

        item1 = [item[-1] for item in data1]
        item2 = [item[-1] for item in data2]
        item3 = [item[-1] for item in data3]
        item4 = [item[-1] for item in data4]
        item5 = [item[-1] for item in data5]

        for idx in range(len(item1)):
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

        log = [i1[:7] + [i2[7]] + [i3[7]] + [i4[7]] + [i5[7]] + i1[7:] for i1, i2, i3, i4, i5 in zip(data1, data2, data3, data4, data5)]
        log = pd.DataFrame(log, columns=["arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sp2", "sp3", "sp4", "sp5", "sentence", "conflict"])

        item1 = log.sp1.values
        item2 = log.sp2.values
        item3 = log.sp3.values
        item4 = log.sp4.values
        item5 = log.sp5.values

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
    if tag in {'jsaispnntcglove', 'paclingspnntcglove', 'acmspnntcglove', 'acmspnntcbert', 'acmspnbccwjglove', 'acmspnbccwjbert'}:
        log.assign(conflict=summary_conflict)
        print(log.query('conflict == True').shape[0])

    ret_path = str(Path('../../results/packed/detaillog_{}.txt'.format(tag)).resolve())
    log.to_csv(ret_path, header=True, index=False, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PASA results combination')
    parser.add_argument('--event', default=None, type=str, choices=['jsai', 'pacling', 'acm'])
    parser.add_argument('--model', default=None, type=str, choices=['sl', 'spg', 'spl', 'spn'])
    parser.add_argument('--corpus', default=None, type=str, choices=['ntc', 'bccwj'])
    parser.add_argument('--emb', default=None, type=str, choices=['glove', 'bert'])
    arguments = parser.parse_args()

    tag = arguments.event + arguments.model + arguments.corpus + arguments.emb
    print(tag)
    main(tag)
