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
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-172255-856357/detaillog_lstm_20200329-172255.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-172256-779451/detaillog_lstm_20200329-172256.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-172259-504777/detaillog_lstm_20200329-172259.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-172333-352525/detaillog_lstm_20200329-172333.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-172320-621931/detaillog_lstm_20200329-172320.txt")
        elif tag == 'acmslbccwjglove':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-003503-990662/detaillog_lstm_20200329-003503.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-003529-533233/detaillog_lstm_20200329-003529.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-003625-441811/detaillog_lstm_20200329-003625.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-003702-744631/detaillog_lstm_20200329-003702.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-lstm-20200329-003720-611158/detaillog_lstm_20200329-003720.txt")

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
                ref = Path("../../results/packed/sentences_bccwj.txt")
            with ref.open("r", encoding="utf-8") as f:
                sentences = [line.strip().split(',') for line in f.readlines()]
            log = [[i] + i1[:8] + [i2[7], i3[7], i4[7], i5[7], [i1[7], i2[7], i3[7], i4[7], i5[7]].count(i1[6])] + i1[8:] + [s[-2]] + ['False'] for i, (i1, i2, i3, i4, i5, s) in
                   enumerate(zip(data1, data2, data3, data4, data5, sentences))]
        else:
            log = [[i] + i1[:8] + [i2[7], i3[7], i4[7], i5[7], [i1[7], i2[7], i3[7], i4[7], i5[7]].count(i1[6])] + i1[8:] for
                   i, (i1, i2, i3, i4, i5) in
                   enumerate(zip(data1, data2, data3, data4, data5))]
        log = pd.DataFrame(log, columns=["index", "arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1", "sl2", "sl3", "sl4", "sl5", "counts", "sentence", "conflict"])

        item1 = log.sl1.values
        item2 = log.sl2.values
        item3 = log.sl3.values
        item4 = log.sl4.values
        item5 = log.sl5.values

    elif tag in {'acmslntcbert', 'acmslbccwjbert'}:
        # For acm 2020
        if tag == 'acmslntcbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-123819-415751/detaillog_bertsl_20200306-123819.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-123818-814117/detaillog_bertsl_20200306-123818.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-123820-333582/detaillog_bertsl_20200306-123820.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-123820-545980/detaillog_bertsl_20200306-123820.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-201956-237530/detaillog_bertsl_20200306-201956.txt")
        elif tag == 'acmslbccwjbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200403-112105-536009/detaillog_bertsl_20200403-112105.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-225320-031641/detaillog_bertsl_20200402-225320.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-225138-903629/detaillog_bertsl_20200402-225138.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-230314-149516/detaillog_bertsl_20200402-230314.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertsl-20200402-230524-638769/detaillog_bertsl_20200402-230524.txt")

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

        log = [[i] + i1[:8] + [i2[7], i3[7], i4[7], i5[7], [i1[7], i2[7], i3[7], i4[7], i5[7]].count(i1[6])] + i1[8:]
               for
               i, (i1, i2, i3, i4, i5) in
               enumerate(zip(data1, data2, data3, data4, data5))]
        log = pd.DataFrame(log, columns=["index", "arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sl1", "sl2", "sl3", "sl4", "sl5", "counts", "sentence", "conflict"])

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
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-220620-026555/detaillog_pointer_20200328-220620.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-220701-953235/detaillog_pointer_20200328-220701.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-220650-498845/detaillog_pointer_20200328-220650.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-220618-338695/detaillog_pointer_20200328-220618.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-220642-006275/detaillog_pointer_20200328-220642.txt")
        elif tag == 'acmsplntcglove':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-181219-050793/detaillog_pointer_20200329-181219.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-181242-757471/detaillog_pointer_20200329-181242.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-181255-253679/detaillog_pointer_20200329-181255.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-181329-741718/detaillog_pointer_20200329-181329.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-181405-914906/detaillog_pointer_20200329-181405.txt")
        elif tag == 'acmspgntcbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-134057-799938/detaillog_bertptr_20200402-134057.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-134057-825245/detaillog_bertptr_20200402-134057.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-134057-738238/detaillog_bertptr_20200402-134057.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-134057-896365/detaillog_bertptr_20200402-134057.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-134106-778681/detaillog_bertptr_20200402-134106.txt")
        elif tag == 'acmsplntcbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-195131-152329/detaillog_bertptr_20200402-195131.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-195230-748475/detaillog_bertptr_20200402-195230.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-195441-889702/detaillog_bertptr_20200402-195441.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-200529-393340/detaillog_bertptr_20200402-200529.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-200821-141107/detaillog_bertptr_20200402-200821.txt")
        elif tag == 'acmspgbccwjglove':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-022106-069150/detaillog_pointer_20200329-022106.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-022109-056568/detaillog_pointer_20200329-022109.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-022128-955906/detaillog_pointer_20200329-022128.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-022222-724719/detaillog_pointer_20200329-022222.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-022313-903459/detaillog_pointer_20200329-022313.txt")
        elif tag == 'acmsplbccwjglove':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-053317-066569/detaillog_pointer_20200329-053317.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-053420-334813/detaillog_pointer_20200329-053420.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-053658-852976/detaillog_pointer_20200329-053658.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-053744-584854/detaillog_pointer_20200329-053744.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-053954-847594/detaillog_pointer_20200329-053954.txt")
        elif tag == 'acmspgbccwjbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-025538-564688/detaillog_bertptr_20200403-025538.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-025740-192547/detaillog_bertptr_20200403-025740.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-025725-718275/detaillog_bertptr_20200403-025725.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-030515-753190/detaillog_bertptr_20200403-030515.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-030648-760108/detaillog_bertptr_20200403-030648.txt")
        elif tag == 'acmsplbccwjbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-045040-398629/detaillog_bertptr_20200403-045040.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-045320-503212/detaillog_bertptr_20200403-045320.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-045346-565331/detaillog_bertptr_20200403-045346.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-050141-426441/detaillog_bertptr_20200403-050141.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-050204-373548/detaillog_bertptr_20200403-050204.txt")

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

        log = [[i] + i1[:8] + [i2[7], i3[7], i4[7], i5[7], [i1[7], i2[7], i3[7], i4[7], i5[7]].count(i1[6])] + i1[8:]
               for
               i, (i1, i2, i3, i4, i5) in
               enumerate(zip(data1, data2, data3, data4, data5))]
        log = pd.DataFrame(log, columns=["index", "arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sp2", "sp3", "sp4", "sp5", "counts", "sentence", "conflict"])

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
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-224527-671480/detaillog_pointer_20200328-224527.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-224523-646336/detaillog_pointer_20200328-224523.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-224545-172444/detaillog_pointer_20200328-224545.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-224538-434833/detaillog_pointer_20200328-224538.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200328-224530-441394/detaillog_pointer_20200328-224530.txt")
        elif tag == 'acmspnntcbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-165144-157728/detaillog_bertptr_20200402-165144.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-165338-628976/detaillog_bertptr_20200402-165338.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-165557-747882/detaillog_bertptr_20200402-165557.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-170544-734496/detaillog_bertptr_20200402-170544.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200402-170813-804379/detaillog_bertptr_20200402-170813.txt")
        elif tag == 'acmspnbccwjglove':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-035820-082435/detaillog_pointer_20200329-035820.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-035828-731188/detaillog_pointer_20200329-035828.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-035921-430627/detaillog_pointer_20200329-035921.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-040037-823312/detaillog_pointer_20200329-040037.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-pointer-20200329-040155-312838/detaillog_pointer_20200329-040155.txt")
        elif tag == 'acmspnbccwjbert':
            log1 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-010141-686124/detaillog_bertptr_20200403-010141.txt")
            log2 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-010141-667945/detaillog_bertptr_20200403-010141.txt")
            log3 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-010141-341382/detaillog_bertptr_20200403-010141.txt")
            log4 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-011035-863656/detaillog_bertptr_20200403-011035.txt")
            log5 = Path("/clwork/keigo/JapanesePASAasaSequencePointingTask/results/pasa-bertptr-20200403-011130-170880/detaillog_bertptr_20200403-011130.txt")

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

        log = [[i] + i1[:8] + [i2[7], i3[7], i4[7], i5[7], [i1[7], i2[7], i3[7], i4[7], i5[7]].count(i1[6])] + i1[8:]
               for
               i, (i1, i2, i3, i4, i5) in
               enumerate(zip(data1, data2, data3, data4, data5))]
        log = pd.DataFrame(log, columns=["index", "arg", "pred", "prop", "word_distance", "ku_distance", "pred_or_not", "label", "sp1", "sp2", "sp3", "sp4", "sp5", "counts", "sentence", "conflict"])

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
