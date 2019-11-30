import pandas as pd
from pathlib import Path

# lstm_log_path = Path("../../results/packed/lstm/detaillog_lstm.txt")
# global_argmax_log_path = Path("../../results/packed/global_argmax/detaillog_global_argmax.txt")
lstm_log_path = Path("../../results/testresult_acl2019/detaillog_lstm.txt")
global_argmax_log_path = Path("../../results/testresult_acl2019/detaillog_global_argmax.txt")
ordered_log_path = Path("../../results/packed/ordered/detaillog_ordered.txt")
no_decode_log_path = Path("../../results/packed/no_decode/detaillog_no_decode.txt")

with lstm_log_path.open("r", encoding="utf-8") as f:
    lstm_log = pd.read_csv(f)
with global_argmax_log_path.open("r", encoding="utf-8") as f:
    global_argmax_log = pd.read_csv(f)
with ordered_log_path.open("r", encoding="utf-8") as f:
    ordered_log = pd.read_csv(f)
with no_decode_log_path.open("r", encoding="utf-8") as f:
    no_decode_log = pd.read_csv(f)

lstm_log = lstm_log.drop(["lstm1", "lstm2", "lstm3", "lstm4", "lstm5", "sentence"], axis=1)
lstm_log = lstm_log.rename(columns={'combined': 'lstm'})
lstm_log = lstm_log.join(global_argmax_log.rename(columns={'combined': 'global_argmax'})["global_argmax"])
lstm_log = lstm_log.join(ordered_log.rename(columns={'combined': 'ordered'})["ordered"])
lstm_log = lstm_log.join(no_decode_log.rename(columns={'combined': 'no_decode'})["no_decode"])
logs = lstm_log.join(ordered_log["sentence"])


if __name__ == "__main__":
    args = logs.arg.values
    preds = logs.pred.values
    props = logs.prop.values
    wdsts = logs.word_distance.values
    kdsts = logs.ku_distance.values
    pons = logs.pred_or_not.values
    labels = logs.label.values
    lstms = logs.lstm.values
    odrs = logs.ordered.values
    gss = logs.global_argmax.values
    nds = logs.no_decode.values
    sentences = logs.sentence.values

    rets = {}

    ret_arg = []
    ret_pred = []
    ret_prop = []
    ret_wdst = []
    ret_kdst = []
    ret_pon = []
    ret_label = []
    ret_lstm = []
    ret_odr = []
    ret_gs = []
    ret_nd = []
    ret_sentence = []

    bef_pred = preds[0]
    bef_sentence = sentences[0]
    start_i = 0

    for idx in range(logs.shape[0]):
        if bef_pred != preds[idx] or bef_sentence != sentences[idx]:
            rets[start_i] = {'arg': ret_arg,
                             'pred': bef_pred,
                             'prop': ret_prop,
                             'wdst': ret_wdst,
                             'kdst': ret_kdst,
                             'pon': ret_pon,
                             'label': ret_label,
                             'lstm': ret_lstm,
                             'odr': ret_odr,
                             'gs': ret_gs,
                             'nd': ret_nd,
                             'sent': bef_sentence
                             }

            ret_arg = [args[idx]]
            ret_pred = [preds[idx]]
            ret_prop = [props[idx]]
            ret_wdst = [wdsts[idx]]
            ret_kdst = [kdsts[idx]]
            ret_pon = [pons[idx]]
            ret_label = [labels[idx]]
            ret_lstm = [lstms[idx]]
            ret_odr = [odrs[idx]]
            ret_gs = [gss[idx]]
            ret_nd = [nds[idx]]
            ret_sentence = [sentences[idx]]

            bef_pred = preds[idx]
            bef_sentence = sentences[idx]
            start_i = idx
        else:
            ret_arg.append(args[idx])
            ret_pred.append(preds[idx])
            ret_prop.append(props[idx])
            ret_wdst.append(wdsts[idx])
            ret_kdst.append(kdsts[idx])
            ret_pon.append(pons[idx])
            ret_label.append(labels[idx])
            ret_lstm.append(lstms[idx])
            ret_odr.append(odrs[idx])
            ret_gs.append(gss[idx])
            ret_nd.append(nds[idx])
            ret_sentence.append(sentences[idx])
    else:
        rets[start_i] = {'arg': ret_arg,
                          'pred': ret_pred,
                          'prop': ret_prop,
                          'wdst': ret_wdst,
                          'kdst': ret_kdst,
                          'pon': ret_pon,
                          'label': ret_label,
                          'lstm': ret_lstm,
                          'odr': ret_odr,
                          'gs': ret_gs,
                          'nd': ret_nd,
                          'sent': ret_sentence
                          }

    pass
