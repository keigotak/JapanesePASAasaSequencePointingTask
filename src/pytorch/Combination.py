# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import os
import numpy as np
import pickle
import gc
import copy
from pathlib import Path
import pickle

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from memory_profiler import profile

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

sys.path.append(os.pardir)
from utils.Datasets import get_datasets, get_datasets_in_sentences, get_datasets_in_sentences_rework, get_datasets_in_sentences_test, get_datasets_in_sentences_test_rework
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
from utils.HelperFunctions import get_argparser, get_pasa, get_now, get_save_dir, add_null, get_pointer_label, concat_labels, get_cuda_id, get_restricted_prediction
from utils.GoogleSpreadSheet import write_spreadsheet

from Loss import *
from Batcher import PairBatcher, SequenceBatcher
from Validation import get_score

from PretrainedEmbeddings import PretrainedEmbedding
from CombinationModel import CombinationModel

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
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test_rework(TRAIN)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test_rework(DEV)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test_rework(TEST)
else:
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_rework(TRAIN)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_rework(DEV)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_rework(TEST)

vocab = Vocab()
vocab.fit(train_vocab, arguments.vocab_thresh)
vocab.fit(dev_vocab, arguments.vocab_thresh)
vocab.fit(test_vocab, arguments.vocab_thresh)
if arguments.embed != 'original':
    vocab = PretrainedEmbedding(arguments.embed, vocab)
train_arg_id, train_pred_id = vocab.transform_sentences(train_args, train_preds)
dev_arg_id, dev_pred_id = vocab.transform_sentences(dev_args, dev_preds)
test_arg_id, test_pred_id = vocab.transform_sentences(test_args, test_preds)

train_lens = max([len(item) for item in train_args])
dev_lens = max([len(item) for item in dev_args])
test_lens = max([len(item) for item in test_args])
fc1_size = max(train_lens, dev_lens, test_lens)

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
sw = StopWatch()
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


# @profile
def train(batch_size, learning_rate=1e-3, optim="adam", fc2_size=128, dropout_ratio=0.4, null_weight=None, loss_weight=None, norm_type=None):
    batch_size = int(batch_size)
    fc2_size = int(fc2_size)
    dropout_ratio = round(dropout_ratio, 2)

    if arguments.embed == 'original':
        model = CombinationModel(l1_size=fc1_size,
                                 l2_size=fc2_size,
                                 dropout_ratio=dropout_ratio,
                                 vocab_size=len(vocab),
                                 word_pos_size=len(word_pos_indexer),
                                 ku_pos_size=len(ku_pos_indexer),
                                 mode_size=len(mode_indexer),
                                 vocab_padding_idx=vocab.get_pad_id(),
                                 word_pos_padding_idx=word_pos_indexer.get_pad_id(),
                                 ku_pos_padding_idx=ku_pos_indexer.get_pad_id(),
                                 mode_padding_idx=mode_indexer.get_pad_id(),
                                 device=device,
                                 add_null_word=arguments.add_null_word)
    else:
        model = CombinationModel(l1_size=fc1_size,
                                 l2_size=fc2_size,
                                 dropout_ratio=dropout_ratio,
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
                                 add_null_word=arguments.add_null_word)
    embedding_dim = model.embedding_dim
    hidden_size = model.hidden_size

    if arguments.init_checkpoint != '':
        state_dict_path = Path(arguments.init_checkpoint)
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        print("Load model: {}".format(arguments.init_checkpoint))

    print(model)

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

    train_batcher = SequenceBatcher(batch_size,
                                    train_arg_id,
                                    train_pred_id,
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
        dev_batcher = SequenceBatcher(arguments.dev_size,
                                      train_arg_id,
                                      train_pred_id,
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
        dev_batcher = SequenceBatcher(arguments.dev_size,
                                      dev_arg_id,
                                      dev_pred_id,
                                      dev_label,
                                      dev_prop,
                                      dev_word_pos,
                                      dev_ku_pos,
                                      dev_modes,
                                      vocab_pad_id=vocab.get_pad_id(),
                                      word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                      ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                      mode_pad_id=mode_indexer.get_pad_id(), shuffle=True)

    max_all_score = 0.0
    best_dep_score = 0.0
    best_zero_score = 0.0
    best_num_tp = [0] * 6
    best_num_fp = [0] * 6
    best_num_fn = [0] * 6
    best_e = 0
    best_f1s = [0] * 6

    es = EarlyStop(arguments.earlystop, go_minus=False)
    if loss_weight is not None:
        total = sum(loss_weight.values())
        loss_weight = [loss_weight['loss_weight_seq'] / total,
                       loss_weight['loss_weight_ptr_ga'] / total,
                       loss_weight['loss_weight_ptr_ni'] / total,
                       loss_weight['loss_weight_ptr_wo'] / total]
    if null_weight is not None:
        null_weight = [null_weight['null_weight_ga'],
                       null_weight['null_weight_ni'],
                       null_weight['null_weight_wo'],
                       null_weight['null_weight_el']]
    criterion = nn.NLLLoss(ignore_index=PADDING_ID)
    if null_weight is not None:
        class_weight = torch.FloatTensor(null_weight).to(device)
        criterion = nn.NLLLoss(ignore_index=PADDING_ID, weight=class_weight)

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
               'loss_weight: {}'.format(loss_weight),
               'null_weight: {}'.format(null_weight),
               'optim: {}'.format(optim),
               'add_null_word: {}'.format(arguments.add_null_word),
               'add_null_weight: {}'.format(arguments.add_null_weight),
               'add_loss_weight: {}'.format(arguments.add_loss_weight),
               'git sha: {}'.format(gm.sha)
               ]

    model_dir, _ = op.get_model_save_dir(tag, now)
    best_model_path = ''
    op.save_model_info(_params)
    sw.start()

    for e in range(arguments.epochs):
        vw_train_loss.reset()
        vw_dev_loss.reset()
        for t_batch in range(len(train_batcher)):
            model.train()

            optimizer.zero_grad()
            # output shape: Batch, Sentence_length
            t_args, t_preds, t_labels, t_props, t_word_pos, t_ku_pos, t_mode = train_batcher.get_batch()

            if arguments.add_null_word:
                t_args = add_null(t_args, vocab.get_null_id())
                t_preds = add_null(t_preds, vocab.get_null_id())
                t_word_pos = add_null(t_word_pos, word_pos_indexer.get_null_id())
                t_ku_pos = add_null(t_ku_pos, ku_pos_indexer.get_null_id())
                t_mode = add_null(t_mode, mode_indexer.get_null_id())

            # output shape: Batch, Sentence_length, 1
            t_args = torch.from_numpy(t_args).long().to(device)
            t_preds = torch.from_numpy(t_preds).long().to(device)
            t_word_pos = torch.from_numpy(t_word_pos).long().to(device)
            t_ku_pos = torch.from_numpy(t_ku_pos).long().to(device)
            t_mode = torch.from_numpy(t_mode).long().to(device)

            # output shape: 3, Batch, Sentence_length
            output_seq, output_ptr_ga, output_ptr_ni, output_ptr_wo = model(t_args, t_preds, t_word_pos, t_ku_pos, t_mode)
            if arguments.add_null_word:
                output_seq = output_seq[:, 1:, :].contiguous()

            t_size = output_seq.shape[2]
            t_labels_seq = torch.from_numpy(t_labels).long().to(device)
            loss_seq = criterion(output_seq.view(-1, t_size), t_labels_seq.view(-1))

            # output shape: 3, Batch
            t_ga_labels, t_ni_labels, t_wo_labels = get_pointer_label(t_labels)

            # output shape: Batch
            t_ga_labels = torch.from_numpy(t_ga_labels).long().to(device)
            t_ni_labels = torch.from_numpy(t_ni_labels).long().to(device)
            t_wo_labels = torch.from_numpy(t_wo_labels).long().to(device)

            b_size = output_ptr_ga.shape[0]
            s_size = output_ptr_ga.shape[1]
            criterion_ga = nn.NLLLoss(ignore_index=PADDING_ID)
            criterion_ni = nn.NLLLoss(ignore_index=PADDING_ID)
            criterion_wo = nn.NLLLoss(ignore_index=PADDING_ID)
            if null_weight is not None:
                criterion_ga = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor([null_weight[0]] + [1.0] * (s_size - 1)).to(device))
                criterion_ni = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor([null_weight[1]] + [1.0] * (s_size - 1)).to(device))
                criterion_wo = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor([null_weight[2]] + [1.0] * (s_size - 1)).to(device))
            loss_ptr_ga = criterion_ga(output_ptr_ga.view(b_size, s_size), t_ga_labels)
            loss_ptr_ni = criterion_ni(output_ptr_ni.view(b_size, s_size), t_ni_labels)
            loss_ptr_wo = criterion_wo(output_ptr_wo.view(b_size, s_size), t_wo_labels)

            loss = loss_seq + loss_ptr_ga + loss_ptr_ni + loss_ptr_wo
            if loss_weight is not None:
                loss = (loss_weight[0] * loss_seq + loss_weight[1] * loss_ptr_ga + loss_weight[2] * loss_ptr_ni + loss_weight[3] * loss_ptr_wo)

            loss.backward()

            if clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            vw_train_loss.add(float(loss))

        else:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            all_score_history = []
            dep_score_history = []
            zero_score_history = []
            tp_history = []
            fp_history = []
            fn_history = []
            with torch.no_grad():
                for d_batch in range(len(dev_batcher)):
                    # output shape: Batch, Sentence_length
                    d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode = dev_batcher.get_batch()

                    if arguments.add_null_word:
                        d_args = add_null(d_args, vocab.get_null_id())
                        d_preds = add_null(d_preds, vocab.get_null_id())
                        d_word_pos = add_null(d_word_pos, word_pos_indexer.get_null_id())
                        d_ku_pos = add_null(d_ku_pos, ku_pos_indexer.get_null_id())
                        d_mode = add_null(d_mode, mode_indexer.get_null_id())

                    # output shape: Batch, Sentence_length
                    d_args = torch.from_numpy(d_args).long().to(device)
                    d_preds = torch.from_numpy(d_preds).long().to(device)
                    d_word_pos = torch.from_numpy(d_word_pos).long().to(device)
                    d_ku_pos = torch.from_numpy(d_ku_pos).long().to(device)
                    d_mode = torch.from_numpy(d_mode).long().to(device)

                    # output shape: 3, Batch, Sentence_length+1
                    seq_prediction, ptr_ga_prediction, ptr_ni_prediction, ptr_wo_prediction = model(d_args, d_preds, d_word_pos, d_ku_pos, d_mode)
                    if arguments.add_null_word:
                        seq_prediction = seq_prediction[:, 1:, :].contiguous()

                    # output shape: Batch, Sentence_length+1
                    t_size = seq_prediction.shape[2]
                    d_labels_seq = torch.from_numpy(d_labels).long().to(device)
                    dev_loss_seq = criterion(seq_prediction.view(-1, t_size), d_labels_seq.view(-1))

                    # output shape: Batch, Sentence_length+1
                    b_size = ptr_ga_prediction.shape[0]
                    s_size = ptr_ga_prediction.shape[1]
                    ptr_ga_prediction = ptr_ga_prediction.view(b_size, s_size)
                    ptr_ni_prediction = ptr_ni_prediction.view(b_size, s_size)
                    ptr_wo_prediction = ptr_wo_prediction.view(b_size, s_size)

                    # output shape: 3, Batch
                    d_labels_ptr = get_pointer_label(d_labels)

                    # output shape: Batch
                    d_ga_labels = torch.from_numpy(d_labels_ptr[0]).long().to(device)
                    d_ni_labels = torch.from_numpy(d_labels_ptr[1]).long().to(device)
                    d_wo_labels = torch.from_numpy(d_labels_ptr[2]).long().to(device)

                    criterion_ga = nn.NLLLoss(ignore_index=PADDING_ID)
                    criterion_ni = nn.NLLLoss(ignore_index=PADDING_ID)
                    criterion_wo = nn.NLLLoss(ignore_index=PADDING_ID)
                    if null_weight is not None:
                        criterion_ga = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor(
                            [null_weight[0]] + [1.0] * (s_size - 1)).to(device))
                        criterion_ni = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor(
                            [null_weight[1]] + [1.0] * (s_size - 1)).to(device))
                        criterion_wo = nn.NLLLoss(ignore_index=PADDING_ID, weight=torch.FloatTensor(
                            [null_weight[2]] + [1.0] * (s_size - 1)).to(device))

                    dev_loss_ptr_ga = criterion_ga(ptr_ga_prediction, d_ga_labels)
                    dev_loss_ptr_ni = criterion_ni(ptr_ni_prediction, d_ni_labels)
                    dev_loss_ptr_wo = criterion_wo(ptr_wo_prediction, d_wo_labels)

                    dev_loss = dev_loss_seq + dev_loss_ptr_ga + dev_loss_ptr_ni + dev_loss_ptr_wo
                    if loss_weight is not None:
                        dev_loss = (loss_weight[0] * dev_loss_seq + loss_weight[1] * dev_loss_ptr_ga + loss_weight[2] * dev_loss_ptr_ni + loss_weight[3] * dev_loss_ptr_wo)
                    vw_dev_loss.add(float(dev_loss))

                    # output shape: Batch
                    ga_prediction, ni_prediction, wo_prediction = get_restricted_prediction(ptr_ga_prediction, ptr_ni_prediction, ptr_wo_prediction)

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
                    all_score, dep_score, zero_score, tp, fp, fn = get_score(d_prediction, d_labels, d_props)

                    all_score_history.append(all_score)
                    dep_score_history.append(dep_score)
                    zero_score_history.append(zero_score)
                    tp_history.append(tp)
                    fp_history.append(fp)
                    fn_history.append(fn)
                    if arguments.overfit:
                        break
                dev_batcher.reset()

            all_score = np.mean(all_score_history, axis=0)
            dep_score = np.mean(dep_score_history, axis=0)
            zero_score = np.mean(zero_score_history, axis=0)
            num_tp = np.sum(tp_history, axis=0)
            num_fp = np.sum(fp_history, axis=0)
            num_fn = np.sum(fn_history, axis=0)
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
                tl.writer.add_pr_curve_raw('data/pr curve', num_tp, num_fp, num_tn, num_fn, precisions, recalls, e)
                tl.writer.add_scalar('data/f1', all_score, e)
                tl.writer.add_scalars('data/losses', {'train': vw_train_loss.get_ave(),
                                                      'dev': vw_dev_loss.get_ave()}, e)
                tl.writer.add_scalars('data/gpu memory', {'memory': mw.get_current_memory(),
                                                          'cache': mw.get_current_cache()}, e)

            print_text = ["Current: {:%Y%m%d-%H%M%S} ".format(get_now()),
                          "Lap: {:.2f}s ".format(sw.stop()),
                          "Hyp: {}/{} ".format(op.get_count(), arguments.max_eval),
                          "Epoch: {:03}/{:03} ".format(e + 1, arguments.epochs),
                          "Data: {:06}/{:06} ".format(train_batcher.get_current_index(), train_batcher.get_max_index()),
                          "Train Loss: {:.4f} ".format(vw_train_loss.get_ave()),
                          "Dev Loss: {:.4f} ".format(vw_dev_loss.get_ave()),
                          "F score: {:.4f}/{:.4f}/{:.4f} ".format(all_score, dep_score, zero_score),
                          "max F score: {:.4f} ".format(max_all_score),
                          "hyp max F score: {:.4f} ".format(hyp_max_score.value),
                          "f1: {} ".format(f1s),
                          "TP: {0} / FP: {1} / FN: {2} ".format(num_tp, num_fp, num_fn)]
            sw.start()

            es.is_minimum_delay(vw_dev_loss.get_ave())

            print(''.join(print_text))
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

            if cp.is_minimum(vw_dev_loss.get_ave()) and arguments.save_model:
                model_path = model_dir.joinpath('epoch{0}-f{1:.4f}.h5'.format(e, max_all_score))
                torch.save(model.state_dict(), model_path)
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
        _line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}' \
            .format("{:%Y%m%d-%H%M%S} ".format(get_now()), best_e, max_all_score, best_dep_score, best_zero_score, best_num_tp, best_num_fp, best_num_fn, best_f1s, optim, learning_rate, batch_size, fc1_size, fc2_size,
                    embedding_dim, hidden_size, clip, weight_decay, dropout_ratio, null_weight, loss_weight, best_model_path)
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
            str_loss_weight = [0] + list(map(str, loss_weight))
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
                      + [best_model_path]
        write_spreadsheet(_spreadline)

    op.count_up()

    del model
    del optimizer
    del train_batcher
    del dev_batcher
    gc.collect()
    if arguments.device != "cpu":
        torch.cuda.empty_cache()

    return 1 - max_all_score


if __name__ == "__main__":
    if arguments.hyp:
        train(batch_size=arguments.batch_size,
              norm_type={'clip': arguments.clip, 'weight_decay': 0.0})
    else:
        def objective(args):
            print(args)
            return {
                'loss': train(**args),
                'status': STATUS_OK
            }

        def get_parameter_space_pointer():
            ret = get_parameter_space_pointer_no_null_loss_weight()
            ret = get_parameter_space_pointer_add_loss_weight(ret)
            ret = get_parameter_space_pointer_add_null_weight(ret)
            return ret

        def get_parameter_space_pointer_no_null_loss_weight():
            return {'optim': hp.choice('optim', ['adadelta', 'adam', 'sgd']),
                    'learning_rate': hp.choice('learning_rate', [0.005, 0.001, 0.0001]),
                    'batch_size': hp.quniform('batch_size', 4, 64, 2),
                    'fc2_size': hp.quniform('fc2_size', 64, 256, 16),
                    'dropout_ratio': hp.quniform('dropout_ratio', 0.0, 0.5, 0.1),
                    'norm_type': hp.choice('norm_type', [
                        {
                            'type': 'clip_only',
                            'weight_decay': 0.0,
                            'clip': hp.quniform('clip', 0, 10, 1)
                        }])
            }

        def get_parameter_space_pointer_add_loss_weight(ret):
            ret['loss_weight'] = {
                'loss_weight_seq': hp.quniform('loss_weight_seq', 0, 1000, 1),
                'loss_weight_ptr_ga': hp.quniform('loss_weight_ptr_ga', 0, 1000, 1),
                'loss_weight_ptr_ni': hp.quniform('loss_weight_ptr_ni', 0, 1000, 1),
                'loss_weight_ptr_wo': hp.quniform('loss_weight_ptr_wo', 0, 1000, 1)
            }
            return ret

        def get_parameter_space_pointer_add_null_weight(ret):
            ret['null_weight'] = {
                'null_weight_ga': hp.loguniform('null_weight_ga', -5, 0),
                'null_weight_ni': hp.loguniform('null_weight_ni', -5, 0),
                'null_weight_wo': hp.loguniform('null_weight_wo', -5, 0),
                'null_weight_el': hp.loguniform('null_weight_el', -5, 0)
            }
            return ret

        space = get_parameter_space_pointer_no_null_loss_weight()
        if arguments.add_null_weight and arguments.add_loss_weight:
            space = get_parameter_space_pointer()
        elif not arguments.add_null_weight and arguments.add_loss_weight:
            space = get_parameter_space_pointer_add_loss_weight(space)
        elif arguments.add_null_weight and not arguments.add_loss_weight:
            space = get_parameter_space_pointer_add_null_weight(space)
        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=arguments.max_eval,
                    trials=trials,
                    verbose=1)

        _log_path = save_dir_base.joinpath('optlog_{0:%Y%m%d-%H%M%S}.txt'.format(now))
        with _log_path.open(mode='a') as f:
            f.write("best: " + ", ".join(["{0}={1}".format(key, value) for (key, value) in space_eval(space, trials.argmin).items()]))

        _pickle_path = save_dir_base.joinpath('trials_{0:%Y%m%d-%H%M%S}.pkl'.format(now))
        with _pickle_path.open(mode="wb") as f:
            pickle.dump(trials, f)

