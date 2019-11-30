# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class PairBatcher:
    def __init__(self, batch_size, args, preds, labels, props, shuffle=False):
        self.batch_size = batch_size
        self.args = args.copy()
        self.preds = preds.copy()
        self.labels = labels.copy()
        self.props = props.copy()
        self.current = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            np.random.seed(71)
            self.shuffle_index = np.random.permutation(len(self.args))
            self.args = self.args[self.shuffle_index]
            self.preds = self.preds[self.shuffle_index]
            self.labels = self.labels[self.shuffle_index]
            self.props = self.props[self.shuffle_index]

    def get_batch(self, reset=False):
        args = self._batch(self.args)
        preds = self._batch(self.preds)
        labels = self._batch(self.labels)
        props = self._batch(self.props)

        if reset:
            self.reset()
        else:
            self.current += self.batch_size
        return args, preds, labels, props

    def _batch(self, iterable):
        l = len(iterable)
        ret = iterable[self.current - self.batch_size: min(l, self.current)]
        return ret

    def __len__(self):
        l = len(self.args)
        if l % self.batch_size == 0:
            ret = l // self.batch_size
        else:
            ret = l // self.batch_size + 1
        return ret

    def reset(self):
        self.current = self.batch_size


class SequenceBatcher:
    def __init__(self, batch_size, args, preds, labels, props, word_pos=None, ku_pos=None, mode=None, vocab_pad_id=-1, word_pos_pad_id=-1, ku_pos_pad_id=-1, mode_pad_id=-1, shuffle=False, seed=71):
        np.random.seed(seed)
        random.seed(seed)

        self.batch_size = batch_size
        self.args = args
        self.preds = preds
        self.labels = labels
        self.props = props
        self.word_pos = None
        if word_pos is not None:
            self.word_pos = word_pos
        self.ku_pos = None
        if ku_pos is not None:
            self.ku_pos = ku_pos
        if mode is not None:
            self.mode = mode
        self.current_batch_number = 0
        self.current = 0
        self.max_index = len(self.args)
        self.shuffle = shuffle
        self.seq_len = np.array([len(item) for item in self.args])
        self.max_seq_len = max(self.seq_len)

        self.batch_sequence_index = list(range(len(self.args)))
        if self.shuffle:
            random.shuffle(self.batch_sequence_index)

        self.vocab_pad_id = vocab_pad_id
        self.word_pos_pad_id = word_pos_pad_id
        self.ku_pos_pad_id = ku_pos_pad_id
        self.mode_pad_id = mode_pad_id

    def get_batch(self, reset=False):
        args = self._batch(self.args)
        preds = self._batch(self.preds)
        labels = self._batch(self.labels)
        props = self._batch(self.props)
        word_pos = self._batch(self.word_pos)
        ku_pos = self._batch(self.ku_pos)
        mode = self._batch(self.mode)

        max_seq_len = max(self._batch(self.seq_len))

        # padding
        args = self._padding(args, max_seq_len, self.vocab_pad_id)
        preds = self._padding(preds, max_seq_len, self.vocab_pad_id)
        labels = self._padding(labels, max_seq_len, 4)
        props = self._str_padding(props, max_seq_len, 'pad')
        word_pos = self._padding(word_pos, max_seq_len, self.word_pos_pad_id)
        ku_pos = self._padding(ku_pos, max_seq_len, self.ku_pos_pad_id)
        mode = self._padding(mode, max_seq_len, self.mode_pad_id)

        # add null
        args = self._padding(args, max_seq_len, self.vocab_pad_id)

        if reset:
            self.reset()
        else:
            self.current += args.shape[0]
            self.current_batch_number += 1
        return args, preds, labels, props, word_pos, ku_pos, mode

    def _batch(self, iterable):
        max_index = len(iterable)
        batch_index = self.batch_sequence_index[self.current: min(max_index, self.current + self.batch_size)]
        ret = [iterable[item] for item in batch_index]
        return ret

    def _padding(self, iterable, max_seq_len, pad_value):
        ret = []
        for line in iterable:
            if type(line) == np.ndarray:
                line = line.tolist()
            item = line + [pad_value] * (max_seq_len - len(line))
            ret.append(item)
        ret = np.array(ret)
        return ret

    def _str_padding(self, iterable, max_seq_len, pad_value):
        ret = np.array([line.tolist() + [pad_value] * (max_seq_len - len(line)) for line in iterable])
        return ret

    def __len__(self):
        l = len(self.args)
        if l % self.batch_size == 0:
            ret = l // self.batch_size
        else:
            ret = l // self.batch_size + 1
        return ret

    def reset(self):
        self.current = 0
        self.current_batch_number = 0

    def reshuffle(self):
        random.shuffle(self.batch_sequence_index)
        self.reset()

    def get_current_index(self):
        return self.current

    def get_max_index(self):
        return self.max_index

    def get_current_batch_number(self):
        return self.current_batch_number


class SequenceBatcherBert(SequenceBatcher):
    def get_batch(self, reset=False):
        VOCAB_PADDING_WORD = "[PAD]"
        args = self._batch(self.args)
        preds = self._batch(self.preds)
        labels = self._batch(self.labels)
        props = self._batch(self.props)
        word_pos = self._batch(self.word_pos)
        ku_pos = self._batch(self.ku_pos)
        mode = self._batch(self.mode)

        max_seq_len = max(self._batch(self.seq_len))

        # padding
        args = self._padding(args, max_seq_len, VOCAB_PADDING_WORD)
        preds = self._padding(preds, max_seq_len, VOCAB_PADDING_WORD)
        labels = self._padding(labels, max_seq_len, 4)
        props = self._str_padding(props, max_seq_len, 'pad')
        word_pos = self._padding(word_pos, max_seq_len, self.word_pos_pad_id)
        ku_pos = self._padding(ku_pos, max_seq_len, self.ku_pos_pad_id)
        mode = self._padding(mode, max_seq_len, self.mode_pad_id)

        # add null
        args = self._padding(args, max_seq_len, self.vocab_pad_id)

        if reset:
            self.reset()
        else:
            self.current += args.shape[0]
            self.current_batch_number += 1
        return args, preds, labels, props, word_pos, ku_pos, mode
