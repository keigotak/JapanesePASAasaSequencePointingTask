# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict, OrderedDict

ARG = 0
PRED = 1


class Vocab:
    def __init__(self):
        self.vocab = dict()
        self.inverse_vocab = dict()
        self.freq = dict()

    def fit(self, vocab, threshold=0):
        vocab = np.ravel(vocab)
        if len(self.freq) == 0:
            freq = dict()
        else:
            freq = self.freq

        for word in vocab:
            if word in freq.keys():
                freq[word] += 1
            else:
                freq[word] = 1
        self.freq = OrderedDict(sorted(freq.items(), reverse=True, key=lambda x: x[1]))

        number = 0
        for word in self.freq.keys():
            if threshold == -1:
                self.vocab[word] = number
                number += 1
            else:
                if self.freq[word] > threshold:
                    self.vocab[word] = number
                    number += 1
                else:
                    break

        for word, id in self.vocab.items():
            self.inverse_vocab[id] = word

    def transform(self, text):
        arg = np.array([self.vocab[text[i][ARG]] if text[i][ARG] in self.vocab.keys() else self.get_unk_id() for i in range(len(text))])
        pred = np.array([self.vocab[text[i][PRED]] if text[i][PRED] in self.vocab.keys() else self.get_unk_id() for i in range(len(text))])
        return arg, pred

    def transform_sentences(self, args, preds):
        arg_ret = []
        pred_ret = []
        for arg, pred in zip(args, preds):
            arg_ret.append([self.vocab[arg_item] if arg_item in self.vocab.keys() else self.get_unk_id() for arg_item in arg])
            pred_ret.append([self.vocab[pred_item] if pred_item in self.vocab.keys() else self.get_unk_id() for pred_item in pred])

        return np.array(arg_ret), np.array(pred_ret)

    def get_unk_word(self):
        return "<unk>"

    def get_pad_word(self):
        return "<pad>"

    def get_null_word(self):
        return "<null>"

    def get_unk_id(self):
        return len(self.vocab)

    def get_pad_id(self):
        return len(self.vocab) + 1

    def get_null_id(self):
        return len(self.vocab) + 2

    def __len__(self):
        return len(self.vocab) + 3

    def id2word(self, index):
        ret = self.get_unk_word()
        index = int(index)
        if index in self.inverse_vocab.keys():
            ret = self.inverse_vocab[index]
        else:
            if index == self.get_unk_id():
                ret = self.get_unk_word()
            elif index == self.get_pad_id():
                ret = self.get_pad_word()
            elif index == self.get_null_id():
                ret = self.get_null_word()
        return ret


if __name__ == "__main__":
    vocab = Vocab()
    vocab.fit(["私"])

