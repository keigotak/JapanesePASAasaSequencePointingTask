# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import os
import numpy as np
import random
import gc
import copy
from pathlib import Path
import pickle

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from memory_profiler import profile

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
import adabound

sys.path.append(os.pardir)
from utils.Datasets import get_datasets, get_datasets_in_sentences, get_datasets_in_sentences_test
from utils.Vocab import Vocab
from utils.LineNotifier import LineNotifier
from utils.StopWatch import StopWatch
from utils.CheckPoint import CheckPoint
from utils.DevChecker import DevChecker
from utils.EarlyStop import EarlyStop
from utils.OptCallbacks import OptCallbacks
from utils.Indexer import Indexer
from utils.ValueWatcher import ValueWatcher
from utils.MemoryWatcher import MemoryWatcher
from utils.GitManager import GitManager
from utils.TensorBoardLogger import TensorBoardLogger
from utils.ServerManager import ServerManager
from utils.HelperFunctions import get_argparser, get_pasa, get_now, get_save_dir, add_null, get_pointer_label, concat_labels, get_cuda_id, translate_score_and_loss, print_b
from utils.GoogleSpreadSheet import write_spreadsheet
from utils.ParallelTrials import ParallelTrials

from Loss import *
from Batcher import SequenceBatcherBert
from Validation import get_pr_numbers, get_f_score
from Decoders import get_restricted_prediction, get_ordered_prediction, get_no_decode_prediction

from BertSequencePointingModel import BertSequencePointingModel
from BertSequencePointingModelNoRnn import BertSequencePointingModelNoRnn

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
PADDING_ID = 4

if arguments.test:
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test(TRAIN, with_bccwj=arguments.with_bccwj, with_bert=True)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test(DEV, with_bccwj=arguments.with_bccwj, with_bert=True)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test(TEST, with_bccwj=arguments.with_bccwj, with_bert=True)
else:
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN, with_bccwj=arguments.with_bccwj, with_bert=True)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV, with_bccwj=arguments.with_bccwj, with_bert=True)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST, with_bccwj=arguments.with_bccwj, with_bert=True)

if arguments.num_data != -1 and arguments.num_data < len(train_label):
    np.random.seed(71)
    random.seed(71)
    items = range(len(train_label))
    items = random.sample(items, k=arguments.num_data)
    train_label = [item for i, item in enumerate(train_label) if i in items]
    train_args = [item for i, item in enumerate(train_args) if i in items]
    train_preds = [item for i, item in enumerate(train_preds) if i in items]
    train_prop = [item for i, item in enumerate(train_prop) if i in items]
    train_word_pos = [item for i, item in enumerate(train_word_pos) if i in items]
    train_ku_pos = [item for i, item in enumerate(train_ku_pos) if i in items]
    train_modes = [item for i, item in enumerate(train_modes) if i in items]
    train_word_pos_id = [item for i, item in enumerate(train_word_pos_id) if i in items]
    train_ku_pos_id = [item for i, item in enumerate(train_ku_pos_id) if i in items]
    train_modes_id = [item for i, item in enumerate(train_modes_id) if i in items]

vocab = Vocab()
vocab.fit(train_vocab, arguments.vocab_thresh)
vocab.fit(dev_vocab, arguments.vocab_thresh)
vocab.fit(test_vocab, arguments.vocab_thresh)

train_arg_id, train_pred_id = vocab.transform_sentences(train_args, train_preds)
dev_arg_id, dev_pred_id = vocab.transform_sentences(dev_args, dev_preds)
test_arg_id, test_pred_id = vocab.transform_sentences(test_args, test_preds)

word_pos_indexer = Indexer()
word_pos_id = np.concatenate([train_word_pos_id, dev_word_pos_id, test_word_pos_id])
word_pos_indexer.fit(word_pos_id)
train_word_pos = word_pos_indexer.transform_sentences(train_word_pos)
dev_word_pos = word_pos_indexer.transform_sentences(dev_word_pos)
test_word_pos = word_pos_indexer.transform_sentences(test_word_pos)

ku_pos_indexer = Indexer()
ku_pos_id = np.concatenate([train_ku_pos_id, dev_ku_pos_id, test_ku_pos_id])
ku_pos_indexer.fit(ku_pos_id)
train_ku_pos = ku_pos_indexer.transform_sentences(train_ku_pos)
dev_ku_pos = ku_pos_indexer.transform_sentences(dev_ku_pos)
test_ku_pos = ku_pos_indexer.transform_sentences(test_ku_pos)

mode_indexer = Indexer()
modes_id = np.concatenate([train_modes_id, dev_modes_id, test_modes_id])
mode_indexer.fit(modes_id)
train_modes = mode_indexer.transform_sentences(train_modes)
dev_modes = mode_indexer.transform_sentences(dev_modes)
test_modes = mode_indexer.transform_sentences(test_modes)

embedding_dim = arguments.embed_size
hidden_size = arguments.hidden_size

tag = get_pasa() + "-" + arguments.model
now = get_now()
save_dir_base, _ = get_save_dir(tag, now)

op = OptCallbacks()
ln = LineNotifier()
sw_lap = StopWatch()
sw_total = StopWatch()
cp = CheckPoint()
dc = DevChecker(arguments.printevery)
vw_train_loss = ValueWatcher()
vw_dev_loss = ValueWatcher()
mw = MemoryWatcher()
gm = GitManager()
sm = ServerManager(arguments.device)
if arguments.tensorboard:
    tl = TensorBoardLogger(str(save_dir_base.joinpath("runs").resolve()))
hyp_max_score = ValueWatcher()
trials = Trials()


# @profile
def train(batch_size, learning_rate=1e-3, fc1_size=128, optim="adam",  dropout_ratio=0.4, null_weight=None, loss_weight=None, norm_type=None):
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)

    batch_size = int(batch_size)
    fc1_size = 0
    fc2_size = 0
    dropout_ratio = round(dropout_ratio, 2)

    if arguments.model == "bertptrnr":
        model = BertSequencePointingModelNoRnn(target_size=1,
                                               dropout_ratio=dropout_ratio,
                                               word_pos_size=len(word_pos_indexer),
                                               ku_pos_size=len(ku_pos_indexer),
                                               mode_size=len(mode_indexer),
                                               word_pos_pred_idx=word_pos_indexer.word2id(0),
                                               vocab_padding_idx=vocab.get_pad_id(),
                                               word_pos_padding_idx=word_pos_indexer.get_pad_id(),
                                               ku_pos_padding_idx=ku_pos_indexer.get_pad_id(),
                                               mode_padding_idx=mode_indexer.get_pad_id(),
                                               device=device,
                                               seed=arguments.seed)
    else:
        model = BertSequencePointingModel(target_size=1,
                                          dropout_ratio=dropout_ratio,
                                          word_pos_size=len(word_pos_indexer),
                                          ku_pos_size=len(ku_pos_indexer),
                                          mode_size=len(mode_indexer),
                                          word_pos_pred_idx=word_pos_indexer.word2id(0),
                                          vocab_padding_idx=vocab.get_pad_id(),
                                          word_pos_padding_idx=word_pos_indexer.get_pad_id(),
                                          ku_pos_padding_idx=ku_pos_indexer.get_pad_id(),
                                          mode_padding_idx=mode_indexer.get_pad_id(),
                                          device=device,
                                          seed=arguments.seed)

    embedding_dim = model.embedding_dim
    hidden_size = model.hidden_size

    if arguments.init_checkpoint != '':
        state_dict_path = Path(arguments.init_checkpoint)
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        print("Load model: {}".format(arguments.init_checkpoint))

    print_b(str(model))
    num_params = 0
    for parameter in model.parameters():
        num_params += len(parameter)
    print_b(num_params)

    if arguments.device != "cpu":
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, device_ids=get_cuda_id(arguments.device))
    model.to(device)

    weight_decay = norm_type['weight_decay']
    clip = int(norm_type['clip'])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr=0.1)

    train_batcher = SequenceBatcherBert(batch_size,
                                        train_args,
                                        train_preds,
                                        train_label,
                                        train_prop,
                                        train_word_pos,
                                        train_ku_pos,
                                        train_modes,
                                        vocab_pad_id=vocab.get_pad_id(),
                                        word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                        ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                        mode_pad_id=mode_indexer.get_pad_id(), shuffle=True)
    if arguments.overfit:
        dev_batcher = SequenceBatcherBert(arguments.dev_size,
                                          train_args,
                                          train_preds,
                                          train_label,
                                          train_prop,
                                          train_word_pos,
                                          train_ku_pos,
                                          train_modes,
                                          vocab_pad_id=vocab.get_pad_id(),
                                          word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                          ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                          mode_pad_id=mode_indexer.get_pad_id(), shuffle=True)
    else:
        dev_batcher = SequenceBatcherBert(arguments.dev_size,
                                          dev_args,
                                          dev_preds,
                                          dev_label,
                                          dev_prop,
                                          dev_word_pos,
                                          dev_ku_pos,
                                          dev_modes,
                                          vocab_pad_id=vocab.get_pad_id(),
                                          word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                          ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                          mode_pad_id=mode_indexer.get_pad_id(), shuffle=True)

    if len(trials) != 1 and not arguments.hyp:
        hyp_max_score.set(translate_score_and_loss(trials.best_trial['result']['loss']))
    max_all_score = 0.0
    best_dep_score = 0.0
    best_zero_score = 0.0
    best_num_tp = [0] * 6
    best_num_fp = [0] * 6
    best_num_fn = [0] * 6
    best_e = 0
    best_f1s = [0] * 6

    es = EarlyStop(arguments.earlystop, go_minus=False)

    _params = ['server: {} {}'.format(sm.server_name, sm.device_name),
               'init_checkpoint: {}'.format(arguments.init_checkpoint),
               'embed_type: {}'.format(arguments.embed),
               'train_batch_size: {}'.format(batch_size),
               'dev_batch_size: {}'.format(arguments.dev_size),
               'learning_rate: {}'.format(learning_rate),
               'fc1_size: {}'.format(fc1_size),
               'fc2_size: {}'.format(fc2_size),
               'embedding_dim: {}'.format(embedding_dim),
               'hidden_size: {}'.format(hidden_size),
               'clip: {}'.format(clip),
               'weight_decay: {}'.format(weight_decay),
               'dropout_ratio: {}'.format(dropout_ratio),
               'optim: {}'.format(optim),
               'git sha: {}'.format(gm.sha),
               'seed: {}'.format(arguments.seed),
               "with_bccwj: {}".format(arguments.with_bccwj)
               ]

    model_dir, _ = op.get_model_save_dir(tag, now)
    best_model_path = ''
    op.save_model_info(_params)
    sw_lap.start()
    sw_total.start()

    for e in range(arguments.epochs):
        vw_train_loss.reset()
        vw_dev_loss.reset()
        for t_batch in range(len(train_batcher)):
            model.train()

            optimizer.zero_grad()
            # output shape: Batch, Sentence_length
            t_args, t_preds, t_labels, t_props, t_word_pos, t_ku_pos, t_mode = train_batcher.get_batch()

            # output shape: Batch, Sentence_length, 1
            t_word_pos = torch.from_numpy(t_word_pos).long().to(device)
            t_ku_pos = torch.from_numpy(t_ku_pos).long().to(device)
            t_mode = torch.from_numpy(t_mode).long().to(device)

            # output shape: 3, Batch, Sentence_length
            output = model(t_args, t_preds, t_word_pos, t_ku_pos, t_mode)

            # output shape: 3, Batch
            t_ga_labels, t_ni_labels, t_wo_labels = get_pointer_label(t_labels)

            # output shape: Batch
            t_ga_labels = torch.from_numpy(t_ga_labels).long().to(device)
            t_ni_labels = torch.from_numpy(t_ni_labels).long().to(device)
            t_wo_labels = torch.from_numpy(t_wo_labels).long().to(device)

            b_size = output[0].shape[0]
            s_size = output[0].shape[1]
            criterion_ga = nn.CrossEntropyLoss()
            criterion_ni = nn.CrossEntropyLoss()
            criterion_wo = nn.CrossEntropyLoss()

            loss_ga = criterion_ga(output[0].view(b_size, s_size), t_ga_labels)
            loss_ni = criterion_ni(output[1].view(b_size, s_size), t_ni_labels)
            loss_wo = criterion_wo(output[2].view(b_size, s_size), t_wo_labels)
            loss = loss_ga + loss_ni + loss_wo
            if loss_weight is not None:
                loss = (loss_weight[0] * loss_ga + loss_weight[1] * loss_ni + loss_weight[2] * loss_wo)

            loss.backward()

            if clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            vw_train_loss.add(float(loss))

        else:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            tp_history = []
            fp_history = []
            fn_history = []
            with torch.no_grad():
                for d_batch in range(len(dev_batcher)):
                    # output shape: Batch, Sentence_length
                    d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode = dev_batcher.get_batch()

                    # output shape: Batch, Sentence_length
                    d_word_pos = torch.from_numpy(d_word_pos).long().to(device)
                    d_ku_pos = torch.from_numpy(d_ku_pos).long().to(device)
                    d_mode = torch.from_numpy(d_mode).long().to(device)

                    # output shape: 3, Batch, Sentence_length+1
                    ga_prediction, ni_prediction, wo_prediction = model(d_args, d_preds, d_word_pos, d_ku_pos, d_mode)

                    # output shape: Batch, Sentence_length+1
                    b_size = ga_prediction.shape[0]
                    s_size = ga_prediction.shape[1]
                    ga_prediction = ga_prediction.view(b_size, s_size)
                    ni_prediction = ni_prediction.view(b_size, s_size)
                    wo_prediction = wo_prediction.view(b_size, s_size)

                    # output shape: 3, Batch
                    d_loss_labels = get_pointer_label(d_labels)

                    # output shape: Batch
                    d_ga_labels = torch.from_numpy(d_loss_labels[0]).long().to(device)
                    d_ni_labels = torch.from_numpy(d_loss_labels[1]).long().to(device)
                    d_wo_labels = torch.from_numpy(d_loss_labels[2]).long().to(device)

                    criterion_ga = nn.CrossEntropyLoss()
                    criterion_ni = nn.CrossEntropyLoss()
                    criterion_wo = nn.CrossEntropyLoss()

                    dev_loss_ga = criterion_ga(ga_prediction, d_ga_labels)
                    dev_loss_ni = criterion_ni(ni_prediction, d_ni_labels)
                    dev_loss_wo = criterion_wo(wo_prediction, d_wo_labels)

                    dev_loss = dev_loss_ga + dev_loss_ni + dev_loss_wo
                    vw_dev_loss.add(float(dev_loss))

                    # output shape: Batch
                    if arguments.decode == 'ordered':
                        ga_prediction, ni_prediction, wo_prediction = get_ordered_prediction(ga_prediction,
                                                                                             ni_prediction,
                                                                                             wo_prediction)
                    elif arguments.decode == 'global_argmax':
                        ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(ga_prediction,
                                                                                                ni_prediction,
                                                                                                wo_prediction)
                    elif arguments.decode == 'no_decoder':
                        ga_prediction, ni_prediction, wo_prediction = get_no_decode_prediction(ga_prediction,
                                                                                               ni_prediction,
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
                    d_prediction = concat_labels(ga_prediction, ni_prediction, wo_prediction)
                    tp, fp, fn = get_pr_numbers(d_prediction, d_labels, d_props)

                    tp_history.append(tp)
                    fp_history.append(fp)
                    fn_history.append(fn)
                    if arguments.overfit:
                        break
                dev_batcher.reset()

            num_tp = np.sum(tp_history, axis=0)
            num_fp = np.sum(fp_history, axis=0)
            num_fn = np.sum(fn_history, axis=0)
            all_score, dep_score, zero_score = get_f_score(num_tp, num_fp, num_fn)
            is_best = hyp_max_score.maximum(all_score)

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

            if max_all_score < all_score:
                max_all_score = all_score
                best_dep_score = dep_score
                best_zero_score = zero_score
                best_num_tp = num_tp
                best_num_fp = num_fp
                best_num_fn = num_fn
                best_e = e
                best_f1s = f1s

            if arguments.tensorboard:
                tb_base_dir = model_dir.parts[-1] + '/'
                tl.writer.add_pr_curve_raw(tb_base_dir + 'pr curve', num_tp, num_fp, num_tn, num_fn, precisions, recalls, e)
                tl.writer.add_scalars(tb_base_dir + 'f1', {'all': all_score, 'dep': dep_score, 'zero': zero_score}, e)
                tl.writer.add_scalars(tb_base_dir + 'losses', {'train': vw_train_loss.get_ave(),
                                                               'dev': vw_dev_loss.get_ave()}, e)
                tl.writer.add_scalars(tb_base_dir + 'gpu memory', {'memory': mw.get_current_memory(),
                                                                   'cache': mw.get_current_cache()}, e)

            print_text = ["Current: {:%Y%m%d-%H%M%S} ".format(get_now()),
                          "Lap: {:.2f}s/{:.2f}s ".format(sw_lap.stop(), sw_total.stop()),
                          "Hyp: {}/{} ".format(op.get_count(), arguments.max_eval),
                          "Epoch: {:03}/{:03} ".format(e + 1, arguments.epochs),
                          "Data: {:06}/{:06} ".format(train_batcher.get_current_index(), train_batcher.get_max_index()),
                          "Train Loss: {:.4f} ".format(vw_train_loss.get_ave()),
                          "Dev Loss: {:.4f} ".format(vw_dev_loss.get_ave()),
                          "F score: {:.4f}/{:.4f}/{:.4f} ".format(all_score, dep_score, zero_score),
                          "max F score: {:.4f} ".format(max_all_score),
                          "hyp max F score: {:.4f} ".format(hyp_max_score.value),
                          "f1: {} ".format(f1s),
                          "TP: {0}/FP: {1}/FN: {2} ".format(num_tp, num_fp, num_fn),
                          "docode: {}".format(arguments.decode)]
            sw_lap.start()

            es.is_maximum_delay(all_score)

            print_b(''.join(print_text))
            if arguments.line and (is_best or es.is_over() or e == arguments.epochs - 1):
                ln.send_message("\nStart: {0:%Y%m%d-%H%M%S}\n".format(now) +
                                arguments.model + "\n" +
                                "\n".join(print_text + _params))

            _log_path = save_dir_base.joinpath('trainlog_{0:%Y%m%d-%H%M%S}.txt'.format(now))
            with _log_path.open(mode='a') as f:
                _line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}' \
                    .format(get_now(), e + 1, vw_train_loss.get_ave(), vw_dev_loss.get_ave(), all_score, max_all_score, num_tp, num_fp, num_fn)
                f.write(_line + '\n')
            vw_train_loss.reset()

            if cp.is_maximum(max_all_score) and arguments.save_model:
                model_path = model_dir.joinpath('epoch{0}-f{1:.4f}.h5'.format(e, max_all_score))
                torch.save(model.state_dict(), model_path)
                model_path = model_dir.joinpath('epoch{0}-f{1:.4f}_bert.h5'.format(e, max_all_score))
                torch.save(model.word_embeddings.state_dict(), model_path)
                best_model_path = model_path

            if es.is_over():
                break

        train_batcher.reshuffle()
        dev_batcher.reset()

        if es.is_over():
            print('Stop learning with early stopping.')
            break

    _log_path = save_dir_base.joinpath('optlog_{0:%Y%m%d-%H%M%S}.txt'.format(now))
    with _log_path.open(mode='a') as f:
        _line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}' \
            .format("{:%Y%m%d-%H%M%S} ".format(get_now()), best_e, max_all_score, best_dep_score, best_zero_score, best_num_tp, best_num_fp, best_num_fn, best_f1s, optim, learning_rate, batch_size, fc1_size, fc2_size,
                    embedding_dim, hidden_size, clip, weight_decay, dropout_ratio, null_weight, loss_weight, best_model_path, arguments.seed)
        f.write(_line + '\n')

    if arguments.spreadsheet:
        str_best_num_tp = list(map(str, best_num_tp))
        str_best_num_fp = list(map(str, best_num_fp))
        str_best_num_fn = list(map(str, best_num_fn))
        str_best_f1s = list(map(str, best_f1s))
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
                       best_e,
                       max_all_score,
                       best_dep_score,
                       best_zero_score]\
                      + str_best_num_tp\
                      + str_best_num_fp\
                      + str_best_num_fn\
                      + str_best_f1s\
                      + [optim,
                         learning_rate,
                         batch_size,
                         fc1_size,
                         fc2_size,
                         embedding_dim,
                         hidden_size,
                         clip,
                         weight_decay,
                         dropout_ratio]\
                      + str_null_weight\
                      + str_loss_weight\
                      + [best_model_path]\
                      + [arguments.seed]\
                      + [gm.sha]\
                      + [arguments.with_linear]\
                      + [num_params]\
                      + [arguments.num_data]\
                      + [arguments.decode] \
                      + [arguments.model]\
                      + [arguments.with_bccwj]
        write_spreadsheet(_spreadline)

    op.count_up()

    del model
    del optimizer
    del train_batcher
    del dev_batcher
    gc.collect()
    if arguments.device != "cpu":
        torch.cuda.empty_cache()

    return max_all_score, best_dep_score, best_zero_score


if __name__ == "__main__":
    if arguments.hyp:
        train(batch_size=2,
              learning_rate=0.2, fc1_size=88, optim="sgd", dropout_ratio=0.4,
              norm_type={'clip': 2, 'weight_decay': 0.0})
    else:
        def objective(args):
            print(args)
            score_all, score_dep, score_zero = train(**args)
            return {
                'loss': translate_score_and_loss(score_all),
                'status': STATUS_OK,
                'attachments': {'all': score_all, 'dep': score_dep, 'zero': score_zero}
            }

        def get_parameter_space_pointer_no_null_loss_weight():
            return {'optim': hp.choice('optim', ['adam', 'sgd', 'rmsprop', 'adabound']),
                    'learning_rate': hp.choice('learning_rate', [0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]),
                    'batch_size': hp.quniform('batch_size', 2, 200, 2),
                    'dropout_ratio': hp.quniform('dropout_ratio', 0.0, 0.7, 0.1),
                    'norm_type': hp.choice('norm_type', [
                        {
                            'type': 'clip_only',
                            'weight_decay': 0.0,
                            'clip': hp.quniform('clip', 0, 20, 1)
                        }])
            }

        space = get_parameter_space_pointer_no_null_loss_weight()

        _trials_path = Path('../../results/trials').joinpath('{0:%Y%m%d-%H%M%S}.pkl'.format(now))
        if arguments.trials_key != '':
            _pallalel_trials_path = Path('../../results/trials').joinpath(arguments.trials_key + '.pkl')
            pallalel_trials = ParallelTrials(_pallalel_trials_path)

        for eval in range(arguments.max_eval):
            if arguments.trials_key != '':
                trials = copy.deepcopy(pallalel_trials.trials)

            best = fmin(objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=len(trials.tids) + 1,
                        trials=trials)

            _log_path = save_dir_base.joinpath('bestlog_{0:%Y%m%d-%H%M%S}.txt'.format(now))
            with _log_path.open(mode='a') as f:
                f.write("best: " + ", ".join(["{0}={1}".format(key, value) for (key, value) in space_eval(space, trials.argmin).items()]))

            best_score = translate_score_and_loss(trials.best_trial['result']['loss'])
            if arguments.trials_key != '':
                best_score = translate_score_and_loss(pallalel_trials.get_best_score())
                pallalel_trials.add(trials, save=True)

            with _trials_path.open(mode="wb") as f:
                pickle.dump(trials, f)

