# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import os
import numpy as np
from pathlib import Path
import pickle

import torch
import torch.nn as nn

sys.path.append(os.pardir)
from utils.Datasets import get_datasets, get_datasets_in_sentences, get_datasets_in_sentences_test
from utils.Vocab import Vocab
from utils.LineNotifier import LineNotifier
from utils.StopWatch import StopWatch
from utils.Indexer import Indexer
from utils.ValueWatcher import ValueWatcher
from utils.MemoryWatcher import MemoryWatcher
from utils.GitManager import GitManager
from utils.TensorBoardLogger import TensorBoardLogger
from utils.ServerManager import ServerManager
from utils.HelperFunctions import get_argparser, get_pasa, get_now, get_save_dir, add_null, get_pointer_label, concat_labels, get_cuda_id
from utils.GoogleSpreadSheet import write_spreadsheet

from Loss import *
from Batcher import SequenceBatcher
from Validation import get_pr_numbers, get_f_score
from Decoders import get_restricted_prediction, get_ordered_prediction, get_no_decode_prediction

from PretrainedEmbeddings import PretrainedEmbedding
from PointerNetworksModel import PointerNetworksModel

arguments = get_argparser()

device = torch.device("cpu")
gpu = False
if arguments.device != "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        gpu = True

TRAIN = "train"
DEV = "dev"
TEST = "test"

if arguments.test:
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test(TRAIN, with_bccwj=arguments.with_bccwj, with_bert=False)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test(DEV, with_bccwj=arguments.with_bccwj, with_bert=False)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test(TEST, with_bccwj=arguments.with_bccwj, with_bert=False)
else:
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN, with_bccwj=arguments.with_bccwj, with_bert=False)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV, with_bccwj=arguments.with_bccwj, with_bert=False)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST, with_bccwj=arguments.with_bccwj, with_bert=False)

vocab = Vocab()
vocab.fit(train_vocab, arguments.vocab_thresh)
vocab.fit(dev_vocab, arguments.vocab_thresh)
vocab.fit(test_vocab, arguments.vocab_thresh)
if arguments.embed != 'original':
    vocab = PretrainedEmbedding(arguments.embed, vocab)
test_arg_id, test_pred_id = vocab.transform_sentences(test_args, test_preds)

word_pos_indexer = Indexer()
word_pos_id = np.concatenate([train_word_pos_id, dev_word_pos_id, test_word_pos_id])
word_pos_indexer.fit(word_pos_id)
test_word_pos = word_pos_indexer.transform_sentences(test_word_pos)

ku_pos_indexer = Indexer()
ku_pos_id = np.concatenate([train_ku_pos_id, dev_ku_pos_id, test_ku_pos_id])
ku_pos_indexer.fit(ku_pos_id)
test_ku_pos = ku_pos_indexer.transform_sentences(test_ku_pos)

mode_indexer = Indexer()
modes_id = np.concatenate([train_modes_id, dev_modes_id, test_modes_id])
mode_indexer.fit(modes_id)
test_modes = mode_indexer.transform_sentences(test_modes)

embedding_dim = arguments.embed_size
hidden_size = arguments.hidden_size

tag = get_pasa() + "-" + arguments.model
now = get_now()
save_dir_base, _ = get_save_dir(tag, now)

ln = LineNotifier()
sw = StopWatch()
vw_test_loss = ValueWatcher()
mw = MemoryWatcher()
gm = GitManager()
sm = ServerManager(arguments.device)
if arguments.tensorboard:
    tl = TensorBoardLogger(str(save_dir_base.joinpath("runs").resolve()))


# @profile
def eval(batch_size=1, null_weight=None, loss_weight=None):
    batch_size = int(batch_size)
    fc1_size = arguments.fc1_size
    if arguments.without_linear:
        fc1_size = 0
    fc2_size = 0

    if arguments.embed == 'original':
        model = PointerNetworksModel(target_size=1,
                                     l1_size=fc1_size,
                                     dropout_ratio=0.0,
                                     vocab_size=len(vocab),
                                     word_pos_size=len(word_pos_indexer),
                                     ku_pos_size=len(ku_pos_indexer),
                                     mode_size=len(mode_indexer),
                                     vocab_padding_idx=vocab.get_pad_id(),
                                     word_pos_padding_idx=word_pos_indexer.get_pad_id(),
                                     ku_pos_padding_idx=ku_pos_indexer.get_pad_id(),
                                     mode_padding_idx=mode_indexer.get_pad_id(),
                                     device=device,
                                     add_null_word=arguments.add_null_word,
                                     without_linear=arguments.without_linear)
    else:
        model = PointerNetworksModel(target_size=1,
                                     l1_size=fc1_size,
                                     dropout_ratio=0.0,
                                     vocab_size=0,
                                     word_pos_size=len(word_pos_indexer),
                                     ku_pos_size=len(ku_pos_indexer),
                                     mode_size=len(mode_indexer),
                                     vocab_padding_idx=vocab.get_pad_id(),
                                     word_pos_padding_idx=word_pos_indexer.get_pad_id(),
                                     ku_pos_padding_idx=ku_pos_indexer.get_pad_id(),
                                     mode_padding_idx=mode_indexer.get_pad_id(),
                                     pretrained_embedding=arguments.embed,
                                     pretrained_weights=vocab.weights,
                                     device=device,
                                     add_null_word=arguments.add_null_word,
                                     without_linear=arguments.without_linear)
    embedding_dim = model.embedding_dim
    hidden_size = model.hidden_size

    if arguments.init_checkpoint != '':
        state_dict_path = Path(arguments.init_checkpoint)
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        print("Load model: {}".format(arguments.init_checkpoint))

    print(model)
    num_params = 0
    for parameter in model.parameters():
        num_params += len(parameter)
    print(num_params)

    if arguments.device != "cpu":
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, device_ids=get_cuda_id(arguments.device))
    model.to(device)

    test_batcher = SequenceBatcher(batch_size,
                                   test_arg_id,
                                   test_pred_id,
                                   test_label,
                                   test_prop,
                                   test_word_pos,
                                   test_ku_pos,
                                   test_modes,
                                   vocab_pad_id=vocab.get_pad_id(),
                                   word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                   ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                   mode_pad_id=mode_indexer.get_pad_id(), shuffle=True)

    if null_weight is not None:
        null_weight = [null_weight['null_weight_ga'],
                       null_weight['null_weight_ni'],
                       null_weight['null_weight_wo']]

    _params = ['server: {} {}'.format(sm.server_name, sm.device_name),
               'init_checkpoint: {}'.format(arguments.init_checkpoint),
               'embed_type: {}'.format(arguments.embed),
               'test_batch_size: {}'.format(batch_size),
               'fc1_size: {}'.format(fc1_size),
               'fc2_size: {}'.format(fc2_size),
               'embedding_dim: {}'.format(embedding_dim),
               'hidden_size: {}'.format(hidden_size),
               'loss_weight: {}'.format(loss_weight),
               'null_weight: {}'.format(null_weight),
               'add_null_word: {}'.format(arguments.add_null_word),
               'add_null_weight: {}'.format(arguments.add_null_weight),
               'add_loss_weight: {}'.format(arguments.add_loss_weight),
               'git sha: {}'.format(gm.sha),
               "with_bccwj: {}".format(arguments.with_bccwj)
               ]

    sw.start()

    # Make sure network is in eval mode for inference
    model.eval()

    # Turn off gradients for validation, saves memory and computations
    tp_history = []
    fp_history = []
    fn_history = []
    data_history = []
    with torch.no_grad():
        for t_batch in range(len(test_batcher)):
            # output shape: Batch, Sentence_length
            t_args, t_preds, t_labels, t_props, t_word_pos, t_ku_pos, t_mode = test_batcher.get_batch()

            if arguments.add_null_word:
                t_args = add_null(t_args, vocab.get_null_id())
                t_preds = add_null(t_preds, vocab.get_null_id())
                t_word_pos = add_null(t_word_pos, word_pos_indexer.get_null_id())
                t_ku_pos = add_null(t_ku_pos, ku_pos_indexer.get_null_id())
                t_mode = add_null(t_mode, mode_indexer.get_null_id())

            # output shape: Batch, Sentence_length
            t_args = torch.from_numpy(t_args).long().to(device)
            t_preds = torch.from_numpy(t_preds).long().to(device)
            t_word_pos = torch.from_numpy(t_word_pos).long().to(device)
            t_ku_pos = torch.from_numpy(t_ku_pos).long().to(device)
            t_mode = torch.from_numpy(t_mode).long().to(device)

            # output shape: 3, Batch, Sentence_length+1
            ga_prediction, ni_prediction, wo_prediction = model(t_args, t_preds, t_word_pos, t_ku_pos, t_mode)

            # output shape: Batch, Sentence_length+1
            b_size = ga_prediction.shape[0]
            s_size = ga_prediction.shape[1]
            ga_prediction = ga_prediction.view(b_size, s_size)
            ni_prediction = ni_prediction.view(b_size, s_size)
            wo_prediction = wo_prediction.view(b_size, s_size)

            # output shape: 3, Batch
            t_loss_labels = get_pointer_label(t_labels)

            # output shape: Batch
            t_ga_labels = torch.from_numpy(t_loss_labels[0]).long().to(device)
            t_ni_labels = torch.from_numpy(t_loss_labels[1]).long().to(device)
            t_wo_labels = torch.from_numpy(t_loss_labels[2]).long().to(device)

            data_history.append([ga_prediction.tolist()[0], ni_prediction.tolist()[0], wo_prediction.tolist()[0], t_props.tolist()[0], t_labels.tolist()[0]])

            criterion_ga = nn.CrossEntropyLoss(ignore_index=-1)
            criterion_ni = nn.CrossEntropyLoss(ignore_index=-1)
            criterion_wo = nn.CrossEntropyLoss(ignore_index=-1)
            if null_weight is not None:
                criterion_ga = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(
                    [null_weight[0]] + [1.0] * (s_size - 1)).to(device))
                criterion_ni = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(
                    [null_weight[1]] + [1.0] * (s_size - 1)).to(device))
                criterion_wo = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(
                    [null_weight[2]] + [1.0] * (s_size - 1)).to(device))

            test_loss_ga = criterion_ga(ga_prediction, t_ga_labels)
            test_loss_ni = criterion_ni(ni_prediction, t_ni_labels)
            test_loss_wo = criterion_wo(wo_prediction, t_wo_labels)

            test_loss = test_loss_ga + test_loss_ni + test_loss_wo
            if loss_weight is not None:
                test_loss = (loss_weight[0] * test_loss_ga + loss_weight[1] * test_loss_ni + loss_weight[2] * test_loss_wo)
            vw_test_loss.add(float(test_loss))

            # output shape: Batch
            if arguments.decode == 'ordered':
                ga_prediction, ni_prediction, wo_prediction = get_ordered_prediction(ga_prediction, ni_prediction,
                                                                                     wo_prediction)
            elif arguments.decode == 'global_argmax':
                ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(ga_prediction, ni_prediction,
                                                                                     wo_prediction)
            elif arguments.decode == 'no_decoder':
                ga_prediction, ni_prediction, wo_prediction = get_no_decode_prediction(ga_prediction, ni_prediction,
                                                                                     wo_prediction)

            # output shape: Batch, Sentence_length+1
            ga_prediction = np.identity(s_size)[ga_prediction].astype(np.int64)
            ni_prediction = np.identity(s_size)[ni_prediction].astype(np.int64)
            wo_prediction = np.identity(s_size)[wo_prediction].astype(np.int64)

            if len(ga_prediction.shape) == 1:
                ga_prediction = np.expand_dims(ga_prediction, axis=0)
            if len(ni_prediction.shape) == 1:
                ni_prediction = np.expand_dims(ni_prediction, axis=0)
            if len(wo_prediction.shape) == 1:
                wo_prediction = np.expand_dims(wo_prediction, axis=0)

            # output shape: Batch, Sentence_length
            t_prediction = concat_labels(ga_prediction, ni_prediction, wo_prediction)
            tp, fp, fn = get_pr_numbers(t_prediction, t_labels, t_props)

            _log_path = save_dir_base.joinpath('detaillog_{0}_{1:%Y%m%d-%H%M%S}.txt'.format(arguments.model, now))
            with _log_path.open(mode='a', encoding="utf-8") as f:
                sentence = [vocab.id2word(item) for item in t_args[0]]
                sentence = ' '.join(sentence)
                for arg, pred, prop, word_pos, ku_pos, mode, label, predict in zip(t_args[0], t_preds[0], t_props[0], t_word_pos[0], t_ku_pos[0], t_mode[0], t_labels[0], t_prediction[0]):
                    conflict = False
                    if type(predict) == list:
                        ret = ''
                        for item in predict:
                            ret += str(item)
                        predict = ret
                        conflict = True
                    _line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}' \
                        .format(vocab.id2word(arg),
                                vocab.id2word(pred),
                                prop,
                                word_pos_indexer.id2word(word_pos),
                                ku_pos_indexer.id2word(ku_pos),
                                mode_indexer.id2word(mode),
                                label,
                                predict,
                                sentence,
                                conflict)
                    f.write(_line + '\n')

            tp_history.append(tp)
            fp_history.append(fp)
            fn_history.append(fn)
            if arguments.overfit:
                break

    num_tp = np.sum(tp_history, axis=0)
    num_fp = np.sum(fp_history, axis=0)
    num_fn = np.sum(fn_history, axis=0)
    all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)

    precisions = []
    recalls = []
    f1s = []
    num_tn = np.array([0] * len(num_tp))
    for _tp, _fp, _fn, _tn in zip(num_tp, num_fp, num_fn, num_tn):
        precision = 0.0
        if _tp + _fp != 0:
            precision = _tp / (_tp + _fp)
        precisions.append(precision)

        recall = 0.0
        if _tp + _fn != 0:
            recall = _tp / (_tp + _fn)
        recalls.append(recall)

        f1 = 0.0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)

    if arguments.tensorboard:
        tl.writer.add_pr_curve_raw('data/pr curve', num_tp, num_fp, num_tn, num_fn, precisions, recalls, 1)
        tl.writer.add_scalar('data/f1', all_score, 1)
        tl.writer.add_scalars('data/losses', {'test': vw_test_loss.get_ave()}, 1)
        tl.writer.add_scalars('data/gpu memory', {'memory': mw.get_current_memory(),
                                                  'cache': mw.get_current_cache()}, 1)

    print_text = ["Current: {:%Y%m%d-%H%M%S} ".format(get_now()),
                  "Lap: {:.2f}s ".format(sw.stop()),
                  "Data: {:06}/{:06} ".format(test_batcher.get_current_index(), test_batcher.get_max_index()),
                  "Test Loss: {:.4f} ".format(vw_test_loss.get_ave()),
                  "F score: {:.5f}/{:.5f}/{:.5f} ".format(all_score, dep_score, zero_score),
                  "f1: {} ".format(f1s),
                  "TP: {0} / FP: {1} / FN: {2} ".format(num_tp, num_fp, num_fn)]
    sw.start()

    print(''.join(print_text))
    if arguments.line:
        ln.send_message("\nStart: {0:%Y%m%d-%H%M%S}\n".format(now) +
                        arguments.model + "\n" +
                        "\n".join(print_text + _params))

    _log_path = save_dir_base.joinpath('testlog_{0:%Y%m%d-%H%M%S}.txt'.format(now))
    with _log_path.open(mode='a') as f:
        _line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}' \
            .format(get_now(), vw_test_loss.get_ave(), all_score, dep_score, zero_score, num_tp, num_fp, num_fn, arguments.init_checkpoint)
        f.write(_line + '\n')

    if arguments.save_output:
        items = arguments.init_checkpoint.split("/")
        tmp = '{0:%Y%m%d-%H%M%S}'.format(now)
        if len(items) >= 2:
            _log_path = save_dir_base.joinpath('ptr_{}_{}_{}.pkl'.format(tmp, items[-2], items[-1]))
        else:
            _log_path = save_dir_base.joinpath('ptr_{}.pkl'.format(tmp))
        with _log_path.open('wb') as f:
            pickle.dump(data_history, f)

    if arguments.spreadsheet:
        str_num_tp = list(map(str, num_tp))
        str_num_fp = list(map(str, num_fp))
        str_num_fn = list(map(str, num_fn))
        str_f1s = list(map(str, f1s))
        if null_weight is None:
            null_weight = [0] * 4
        str_null_weight = list(map(str, null_weight))
        if loss_weight is None:
            str_loss_weight = list(map(str, [0] * 4))
        else:
            str_loss_weight = [0] + list(map(str, loss_weight.values()))
        _file = Path(__file__).name
        _spreadline = ["{:%Y%m%d-%H%M%S} ".format(now),
                       "{:%Y%m%d-%H%M%S} ".format(get_now()),
                       _file,
                       '{} {}'.format(sm.server_name, sm.device_name),
                       all_score,
                       dep_score,
                       zero_score]\
                      + str_num_tp\
                      + str_num_fp\
                      + str_num_fn\
                      + str_f1s\
                      + [batch_size,
                         fc1_size,
                         fc2_size,
                         embedding_dim,
                         hidden_size]\
                      + str_null_weight\
                      + str_loss_weight\
                      + [arguments.init_checkpoint] \
                      + [gm.sha] \
                      + [arguments.without_linear] \
                      + [num_params]\
                      + [arguments.decode]\
                      + [arguments.with_bccwj]
        write_spreadsheet(_spreadline, type="test")


if __name__ == "__main__":
    eval()

