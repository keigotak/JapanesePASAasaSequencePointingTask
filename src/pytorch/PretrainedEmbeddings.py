# -*- coding: utf-8 -*-
from gensim.models import word2vec
from pathlib import Path
import pickle

import numpy as np
import torch
from joblib import Parallel, delayed


path_to_w2v = Path("../../data/pre-trained-w2v")
path_to_fasttext = Path("../../data/pre-trained-fasttext")
path_to_oomorisan = Path("../../data/embeddings")


class PretrainedEmbedding:
    def __init__(self, model_type='w2v', pre_vocab=None):
        self.model_type = model_type
        self.pre_vocab = pre_vocab
        self.model = None
        self.weights = None
        self.vocab = None
        self.inverse_vocab = dict()
        self.unk_idx = None
        self.pad_idx = None
        self.null_idx = None
        self.pre_unk_idx = None
        self.load()

    def load(self):
        if self.model_type == "w2v":
            model = word2vec.Word2Vec.load(str(path_to_w2v.joinpath('ja.bin').resolve()))
            self.weights = torch.FloatTensor(model.wv.syn0)
            self.vocab = model.wv.vocab
        elif self.model_type == "fasttext":
            with path_to_fasttext.joinpath("model.pkl").open(mode="rb", encoding='utf-8_sig') as f:
                model = f
        else:
            caption = self.model_type + "_weights"
            if path_to_oomorisan.joinpath(caption + ".pkl").exists():
                with path_to_oomorisan.joinpath(caption + ".pkl").open(mode="rb") as fpkl:
                    self.weights = pickle.load(fpkl)
            else:
                with path_to_oomorisan.joinpath(self.model_type + ".txt").open(mode="r", encoding='utf-8_sig') as f:
                    weights = [line.split(" ")[1:] for line in f][1:]
                    self.weights = np.array(weights, dtype=np.float)
                    with path_to_oomorisan.joinpath(caption + ".pkl").open(mode="wb") as fpkl:
                        pickle.dump(self.weights, fpkl)

            caption = self.model_type + "_vocab"
            if path_to_oomorisan.joinpath(caption + ".pkl").exists():
                with path_to_oomorisan.joinpath(caption + ".pkl").open(mode="rb") as fpkl:
                    self.vocab = pickle.load(fpkl)
            else:
                with path_to_oomorisan.joinpath(self.model_type + ".txt").open(mode="r", encoding='utf-8_sig') as f:
                    vocab = [line.split(" ")[0] for line in f][1:]
                    self.vocab = {vocab[i]: i for i in range(len(vocab))}
                    with path_to_oomorisan.joinpath(caption + ".pkl").open(mode="wb") as fpkl:
                        pickle.dump(self.vocab, fpkl)

        self.weights = torch.FloatTensor(self.weights)
        for item in self.pre_vocab.vocab.keys():
            if item not in self.vocab.keys():
                self.vocab[item] = len(self.vocab)
                embedding_vector = torch.zeros(self.weights.shape[1], requires_grad=True)
                embedding_vector.data.uniform_(-0.25, 0.25)
                self.weights = torch.cat((self.weights, embedding_vector.unsqueeze(0)), dim=0)

        self.unk_idx = len(self.vocab)
        self.pad_idx = len(self.vocab) + 1
        self.null_idx = len(self.vocab) + 2
        self.pre_unk_idx = self.vocab["0"]

        self.vocab[self.get_unk_word()] = self.unk_idx
        self.vocab[self.get_pad_word()] = self.pad_idx
        self.vocab[self.get_null_word()] = self.null_idx
        self.vocab[self.get_pre_unk_word()] = self.pre_unk_idx

        for word, id in self.vocab.items():
            self.inverse_vocab[id] = word

        embedding_unk_vector = torch.zeros(self.weights.shape[1], requires_grad=True)
        embedding_unk_vector.data.uniform_(-0.25, 0.25)
        self.weights = torch.cat((self.weights, embedding_unk_vector.unsqueeze(0)), dim=0)

        embedding_pad_vector = torch.zeros(self.weights.shape[1])
        self.weights = torch.cat((self.weights, embedding_pad_vector.unsqueeze(0)), dim=0)

        embedding_null_vector = torch.zeros(self.weights.shape[1], requires_grad=True)
        embedding_null_vector.data.uniform_(-0.25, 0.25)
        self.weights = torch.cat((self.weights, embedding_null_vector.unsqueeze(0)), dim=0)

    def transform_sentences(self, args, preds):
        arg_ret = [[self.get_index(arg_item) for arg_item in arg] for arg in args]
        pred_ret = [[self.get_index(pred_item) for pred_item in pred] for pred in preds]

        return np.array(arg_ret), np.array(pred_ret)

    def get_index(self, key):
        if key == 0:
            return self.pre_unk_idx

        if key in self.vocab.keys():
            if self.model_type == "w2v":
                return self.vocab[key].index
            else:
                return self.vocab[key]
        else:
            return self.unk_idx

    def get_unk_word(self):
        return "<unk>"

    def get_pad_word(self):
        return "<pad>"

    def get_null_word(self):
        return "<null>"

    def get_pre_unk_word(self):
        return "<pre_unk>"

    def get_unk_id(self):
        return self.unk_idx

    def get_pad_id(self):
        return self.pad_idx

    def get_null_id(self):
        return self.null_idx

    def get_pre_unk_id(self):
        return self.pre_unk_idx

    def get_weights(self):
        return self.weights

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
            elif index == self.get_pre_unk_id():
                ret = self.get_pre_unk_word()
        return ret


if __name__ == "__main__":
    import sys
    sys.path.append(os.pardir)
    from utils.Datasets import get_datasets_in_sentences_rework, get_datasets_in_sentences_test_rework
    from utils.Vocab import Vocab

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences_test_rework(TRAIN)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences_test_rework(DEV)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences_test_rework(TEST)

    vocab = Vocab()
    vocab.fit(train_vocab, -1)
    vocab.fit(dev_vocab, -1)
    vocab.fit(test_vocab, -1)
    vocab = PretrainedEmbedding("glove-retrofitting", vocab)

    """
    日本テレビ 系列 各社 が 参加 し 、 日本 の 現実 を 鋭く 0 0 を 送り出し て き た 「 0 ドキュメント 」 が 満 ２ ５ 周年 を 迎え 、 これ を 記念 し て １ ５ 日 深夜 ０ 時 １ ５ 分 から ４ 週 連続 で 大型 企画 「 アジア から の メッセージ ・ おかしい ぞ ！ ニッポン 」 を 放送 する 。
    """
