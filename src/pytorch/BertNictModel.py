# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel
import os
import sys
sys.path.append(os.pardir)
from utils.HelperFunctions import get_cuda_id, ptoz


class BertNictModel:
    def __init__(self, bert_path=Path("../../data/NICT_BERT-base_JapaneseWikipedia_100K").resolve(),
                 vocab_file_name="vocab.txt",
                 db_file=None,
                 device='cpu',
                 trainable=False):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        for k, v in self.model.named_parameters():
            v.requires_grad = trainable
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        self.device = device
        self.embedding_dim = self.model.embeddings.word_embeddings.embedding_dim
        self.vocab_size = self.model.embeddings.word_embeddings.num_embeddings
        self.max_seq_length = 512

        if self.device != "cpu":
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=get_cuda_id(self.device))

        self.db_file = db_file
        self.conn = None
        self.c = None
        if db_file is not None:
            self.conn = sqlite3.connect(str(self.db_file.resolve()))
            self.c = self.conn.cursor()
        else:
            self.model.to(self.device)

    # def _preprocess_text(self, text):
    #     return text.replace(" ", "")  # for Juman

    # def get_sentence_embedding(self, text):
    #     bert_tokens = self.bert_tokenizer.tokenize(" ".join(text))
    #     token = ["[CLS]"] + bert_tokens[:self.max_seq_length] + ["[SEP]"]
    #     id = self.bert_tokenizer.convert_tokens_to_ids(token) # max_seq_len-2
    #
    #     if len(np.array(token).shape) != 2:
    #         token_tensor = np.array(token).reshape(1, -1)
    #     else:
    #         token_tensor = np.array(token)
    #
    #     if len(np.array(id).shape) != 2:
    #         id_tensor = torch.tensor(id).reshape(1, -1)
    #     else:
    #         id_tensor = torch.tensor(id)
    #
    #     if self.device != "cpu":
    #         if torch.cuda.device_count() > 1:
    #             print("Let's use", torch.cuda.device_count(), "GPUs!")
    #             self.model = nn.DataParallel(self.model, device_ids=get_cuda_id(self.device))
    #     self.model.to(self.device)
    #     all_encoder_layers, _ = self.model(id_tensor)
    #
    #     return {"embedding": all_encoder_layers, "token": token_tensor, "id": id_tensor}

    def get_word_embedding(self, batched_words, tag=None, epoch=None, index=None):
        batched_bert_ids = []
        for words in batched_words:
            words = [self.bert_tokenizer.cls_token] + words[:self.max_seq_length].tolist() + [self.bert_tokenizer.sep_token]
            id = self.bert_tokenizer.convert_tokens_to_ids(words)
            batched_bert_ids.extend([id])

        if self.db_file is not None:
            batched_bert_embs = None
        else:
            batched_bert_embs, _ = self.model(torch.tensor(batched_bert_ids).to(self.device))
        return {"embedding": batched_bert_embs[:, 1:-1, :], "token": batched_words, "id": batched_bert_ids}

    def get_pred_embedding(self, batched_arg_embedding, batched_arg_token, batched_word_pos, pred_pos_index):
        batched_pred_embedding = None
        for word_pos, arg_embedding, arg_token in zip(batched_word_pos, batched_arg_embedding, batched_arg_token):
            index = int((word_pos == pred_pos_index).nonzero().reshape(-1))
            if batched_pred_embedding is None:
                batched_pred_embedding = arg_embedding[index].repeat(len(arg_embedding)).view(-1, len(arg_embedding[index])).unsqueeze(0)
            else:
                batched_pred_embedding = torch.cat((batched_pred_embedding, arg_embedding[index].repeat(len(arg_embedding)).view(-1, len(arg_embedding[index])).unsqueeze(0)), dim=0)
            if "[PAD]" in arg_token:
                pad_from = arg_token.tolist().index("[PAD]")
                for i in range(pad_from, len(batched_pred_embedding[-1])):
                    batched_pred_embedding[-1][i] = arg_embedding[i]
        return {"embedding": batched_pred_embedding, "id": pred_pos_index}

    # def get_embedding(self, text):
    #     items = self.get_sentence_embedding(text)
    #
    #     tokens = items["tokens"]
    #     embeddings = items["embedding"]
    #
    #     pool_tensors = torch.tensor()
    #     for token, embedding in zip(tokens, embeddings):
    #         if "##" in token:
    #             pass
    #         else:
    #             pool_tensors.append(embedding)
    #     return pool_tensors

    def state_dict(self):
        return self.model.state_dict()

    def get_padding_idx(self):
        return self.model.embeddings.word_embeddings.padding_idx


if __name__ == "__main__":
    import argparse
    import sqlite3
    from Batcher import SequenceBatcherBert
    from utils.Datasets import get_datasets_in_sentences
    from utils.Vocab import Vocab
    from utils.Indexer import Indexer

    parser = argparse.ArgumentParser(description='nice bert vector')
    parser.add_argument('--device', default='cpu', type=str)
    arguments = parser.parse_args()

    device = arguments.device
    if device != 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _path = Path("../../data/NICT_BERT-base_JapaneseWikipedia_100K").resolve()
    model = BertNictModel(_path, trainable=False, device=device)
    word_padding_idx = model.get_padding_idx()
    # ret = model.get_word_embedding(np.array([["ど", "う", "いたしまし", "て", "。"]]))
    # print(ret)
    # for k, v in model.named_parameters():
    #     print("{}, {}, {}".format(v.requires_grad, v.size(), k))

    def get_batches(word_padding_idx, with_bccwj=False, with_bert=False):
        TRAIN = "train2"
        DEV = "dev"
        TEST = "test"
        train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(TRAIN, with_bccwj=with_bccwj, with_bert=with_bert)
        dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(DEV, with_bccwj=with_bccwj, with_bert=with_bert)
        test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(TEST, with_bccwj=with_bccwj, with_bert=with_bert)

        vocab = Vocab()
        vocab.fit(train_vocab, -1)
        vocab.fit(dev_vocab, -1)
        vocab.fit(test_vocab, -1)

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

        train_batcher = SequenceBatcherBert(2,
                                            train_args,
                                            train_preds,
                                            train_label,
                                            train_prop,
                                            train_word_pos,
                                            train_ku_pos,
                                            train_modes,
                                            vocab_pad_id=word_padding_idx,
                                            word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                            ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                            mode_pad_id=mode_indexer.get_pad_id(),
                                            shuffle=True)

        dev_batcher = SequenceBatcherBert(4,
                                          dev_args,
                                          dev_preds,
                                          dev_label,
                                          dev_prop,
                                          dev_word_pos,
                                          dev_ku_pos,
                                          dev_modes,
                                          vocab_pad_id=word_padding_idx,
                                          word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                          ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                          mode_pad_id=mode_indexer.get_pad_id(),
                                          shuffle=True)

        test_batcher = SequenceBatcherBert(1,
                                           test_args,
                                           test_preds,
                                           test_label,
                                           test_prop,
                                           test_word_pos,
                                           test_ku_pos,
                                           test_modes,
                                           vocab_pad_id=word_padding_idx,
                                           word_pos_pad_id=word_pos_indexer.get_pad_id(),
                                           ku_pos_pad_id=ku_pos_indexer.get_pad_id(),
                                           mode_pad_id=mode_indexer.get_pad_id(),
                                           shuffle=True)

        return {'train': train_batcher, 'dev': dev_batcher, 'test': test_batcher}

    def save_embs(tag, batcher, corpus):
        epochs = {'train': 20, 'dev': 20, 'test': 1}
        file_db = Path('../../data/NICTBERT/{}-{}-embs.db'.format(corpus, tag))
        if file_db.exists():
            file_db.unlink()
        conn = sqlite3.connect(str(file_db.resolve()))
        c = conn.cursor()
        c.execute('CREATE TABLE dataset (epoch integer, seqid integer, obj blob)')
        with torch.no_grad():
            for e in range(epochs[tag]):
                items = []
                for seqid, t_batch in enumerate(range(len(batcher))):
                    t_args, _, _, _, _, _, _ = batcher.get_batch()
                    ret = model.get_word_embedding(t_args)
                    items.append([e, seqid, ptoz(ret['embedding'])])
                sql = "INSERT INTO dataset (epoch, seqid, obj) VALUES (?, ?, ?)"
                c.executemany(sql, items)
                conn.commit()
                batcher.reshuffle()
            conn.close()


    with_vector = True
    if with_vector:
        batchers = get_batches(word_padding_idx, with_bccwj=False, with_bert=False)
        for tag in ['train', 'dev', 'test']:
            save_embs(tag, batchers[tag], 'ntc')
        batchers = get_batches(word_padding_idx, with_bccwj=True, with_bert=False)
        for tag in ['train', 'dev', 'test']:
            save_embs(tag, batchers[tag], 'bccwj')

