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
from torch.utils.data import DataLoader

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
from utils.WSCDataset import WSCGPT2Dataset

from Loss import *
from Batcher import SequenceBatcherBert
from Validation import get_pr_numbers, get_f_score
from Decoders import get_restricted_prediction, get_ordered_prediction, get_no_decode_prediction

from GPT2SequencePointingModel import GPT2SimpleSequencePointingModel

arguments = get_argparser()

device = torch.device("cpu")
gpu = False
if arguments.device != "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        gpu = True

TRAIN = "train2"
DEV = "dev"
TEST = "test"
PADDING_ID = 4

with_bert = True
if "nict" in arguments.model:
    with_bert = False

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
vw_test_loss = ValueWatcher()
mw = MemoryWatcher()
gm = GitManager()
sm = ServerManager(arguments.device)
if arguments.tensorboard:
    tl = TensorBoardLogger(str(save_dir_base.joinpath("runs").resolve()))
hyp_max_score = ValueWatcher()
trials = Trials()

def logits_encoder(token_pooling_method, logits_answer, logits_candidate):
    if token_pooling_method == 'mean':
        logits_answer = torch.mean(torch.stack(logits_answer))
        logits_candidate = torch.mean(torch.stack(logits_candidate))
    elif token_pooling_method == 'max':
        logits_answer = torch.max(torch.stack(logits_answer))
        logits_candidate = torch.max(torch.stack(logits_candidate))
    elif token_pooling_method == 'first_token':
        logits_answer = logits_answer[0]
        logits_candidate = logits_candidate[0]
    elif token_pooling_method == 'last_token':
        logits_answer = logits_answer[-1]
        logits_candidate = logits_candidate[-1]

    return logits_answer, logits_candidate

def get_token_position(answer_logits, candidate_logits):
    answer_pos = [i for i, d in enumerate(answer_logits) if d[0] == '1']
    candidate_pos = [i for i, d in enumerate(candidate_logits) if d[0] == '1']
    return answer_pos, candidate_pos

# @profile
def train(batch_size, learning_rate=1e-3, fc1_size=128, optim="adam",  dropout_ratio=0.4, null_weight=None, loss_weight=None, norm_type=None):
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)

    result_texts = []

    batch_size = int(batch_size)
    fc1_size = 0
    fc2_size = 0
    dropout_ratio = round(dropout_ratio, 2)

    model = GPT2SimpleSequencePointingModel(target_size=1,
                                      dropout_ratio=dropout_ratio,
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

    wsc_train_batcher = DataLoader(WSCGPT2Dataset(mode='train'), batch_size=1, shuffle=True)
    wsc_dev_batcher = DataLoader(WSCGPT2Dataset(mode='dev'), batch_size=1, shuffle=False)
    wsc_test_batcher = DataLoader(WSCGPT2Dataset(mode='test'), batch_size=1, shuffle=False)

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
               'decode: {}'.format(arguments.decode),
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
               "with_bccwj: {}".format(arguments.with_bccwj),
               "trainbert: {}".format(arguments.trainbert),
               "with_db: {}".format(arguments.with_db)
               ]

    model_dir, _ = op.get_model_save_dir(tag, now)
    best_model_path = ''
    op.save_model_info(_params)
    sw_lap.start()
    sw_total.start()
    best_score = 0.0
    best_model = None

    criterion = nn.CrossEntropyLoss()
    token_pooling_method = 'mean'
    print(token_pooling_method)

    for e in range(arguments.epochs):
        model.train()
        tp, fp, fn, tn = 0, 0, 0, 0
        running_loss = 0.0
        for data in wsc_train_batcher:
            optimizer.zero_grad()

            ga_logits, ni_logits, wo_logits = model(arg=[[int(d[0]) for d in data[0]]],
                                                                pred=[[int(d[0]) for d in data[1]]],
                                                                word_pos=None, ku_pos=None, mode=None, tag='train', is_wsc=True)

            # output shape: Batch, Sentence_length
            t_labels = torch.argmax(torch.LongTensor([[int(d[0]) for d in data[2]]]), dim=1).to(device)

            loss = criterion(ga_logits, t_labels)
            loss.backward()
            if clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            for l, p in zip(t_labels, torch.argmax(ga_logits, dim=1)):
                if l == p:
                    tp += 1
                else:
                    fn += 1
            running_loss += loss.item()
        recall = tp / (tp + fn) if tp + fn != 0 else 0.0
        precision = tp / (tp + fp)  if tp + fp != 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision)  if recall + precision != 0 else 0.0
        acc = tp / (tp + fn) if tp + fn != 0 else 0.0
        print(f'train, {e}, ' + ', '.join([str(acc), str(tp), str(fn), str(running_loss)]))
        result_texts.append(','.join(['train', str(e), str(acc), str(tp), str(fn), str(running_loss)]))

        with torch.no_grad():
            model.eval()
            tp, fp, fn, tn = 0, 0, 0, 0
            for data in wsc_dev_batcher:
                ga_prediction, ni_prediction, wo_prediction = model(arg=[[int(d[0]) for d in data[0]]],
                                                                    pred=[[int(d[0]) for d in data[1]]],
                                                                    word_pos=None, ku_pos=None, mode=None, tag='test', is_wsc=True)

                answer_pos, candidate_pos = get_token_position(data[2], data[3])
                logits_answer = [ga_prediction[0][i] for i in answer_pos]
                logits_candidate = [ga_prediction[0][i] for i in candidate_pos]
                logits_answer, logits_candidate = logits_encoder(token_pooling_method, logits_answer, logits_candidate)

                if logits_answer >= logits_candidate:
                    tp += 1
                else:
                    fn += 1

            recall = tp / (tp + fn) if tp + fn != 0 else 0.0
            precision = tp / (tp + fp)  if tp + fp != 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision)  if recall + precision != 0 else 0.0
            acc = tp / (tp + fn) if tp + fn != 0 else 0.0
            if acc > best_score:
                best_score = acc
                best_model = model
            print(f'dev, {e}, ' + ', '.join([str(acc), str(tp), str(fn)]))
            result_texts.append(','.join(['dev', str(e), str(acc), str(tp), str(fn), str('-')]))

            tp, fp, fn, tn = 0, 0, 0, 0
            for data in wsc_test_batcher:
                ga_prediction, ni_prediction, wo_prediction = model(arg=[[int(d[0]) for d in data[0]]],
                                                                    pred=[[int(d[0]) for d in data[1]]],
                                                                    word_pos=None, ku_pos=None, mode=None, tag='test', is_wsc=True)

                answer_pos, candidate_pos = get_token_position(data[2], data[3])
                logits_answer = [ga_prediction[0][i] for i in answer_pos]
                logits_candidate = [ga_prediction[0][i] for i in candidate_pos]
                logits_answer, logits_candidate = logits_encoder(token_pooling_method, logits_answer, logits_candidate)

                if logits_answer >= logits_candidate:
                    tp += 1
                else:
                    fn += 1

            recall = tp / (tp + fn) if tp + fn != 0 else 0.0
            precision = tp / (tp + fp)  if tp + fp != 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision)  if recall + precision != 0 else 0.0
            acc = tp / (tp + fn) if tp + fn != 0 else 0.0
            if acc > best_score:
                best_score = acc
                best_model = model
            print(f'test, {e}, ' + ', '.join([str(acc), str(tp), str(fn)]))
            result_texts.append(','.join(['test', str(e), str(acc), str(tp), str(fn), str('-')]))

        with Path(f'./wsc_gpt2_{token_pooling_method}_result.txt').open('w') as f:
            for text in result_texts:
                f.write(text)
                f.write('\n')

if __name__ == "__main__":
    if arguments.hyp:
        train(batch_size=2,
              learning_rate=0.01, fc1_size=88, optim="sgd", dropout_ratio=0.4,
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

