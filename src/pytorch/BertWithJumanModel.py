# -*- coding: utf-8 -*-
from pathlib import Path
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel
from pyknp import Juman
import os
import sys
sys.path.append(os.pardir)
from utils.HelperFunctions import get_cuda_id


class JumanTokenizer():
    def __init__(self):
        if "D:" in os.getcwd():
            self.juman = Juman(command="juman")
        else:
            self.juman = Juman(command="jumanpp")

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class BertWithJumanModel():
    def __init__(self, bert_path=Path("../../data/bert-kyoto/Japanese_L-12_H-768_A-12_E-30_BPE").resolve(),
                 vocab_file_name="vocab.txt",
                 device='cpu',
                 trainable=False):
        super().__init__()
        self.juman_tokenizer = JumanTokenizer()
        self.model = BertModel.from_pretrained(bert_path)
        for k, v in self.model.named_parameters():
            v.requires_grad = trainable
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        self.device = device
        self.embedding_dim = self.model.embeddings.word_embeddings.embedding_dim
        self.vocab_size = self.model.embeddings.word_embeddings.num_embeddings
        self.max_seq_length = 224

        if self.device != "cpu":
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=get_cuda_id(self.device))

        self.model.to(self.device)

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def get_sentence_embedding(self, text):
        token = self.juman_tokenizer.tokenize(text)
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(token))
        token = ["[CLS]"] + bert_tokens[:self.max_seq_length] + ["[SEP]"]
        id = self.bert_tokenizer.convert_tokens_to_ids(token) # max_seq_len-2

        if len(np.array(token).shape) != 2:
            token_tensor = np.array(token).reshape(1, -1)
        else:
            token_tensor = np.array(token)

        if len(np.array(id).shape) != 2:
            id_tensor = torch.tensor(id).reshape(1, -1)
        else:
            id_tensor = torch.tensor(id)

        if self.device != "cpu":
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=get_cuda_id(self.device))
        self.model.to(self.device)
        all_encoder_layers, _ = self.model(id_tensor)

        return {"embedding": all_encoder_layers, "token": token_tensor, "id": id_tensor}

    def get_word_embedding(self, batched_words):
        batched_bert_words = []
        batched_bert_ids = []
        batched_bert_embs = []
        batched_bert_seq_ids = []
        for words in batched_words:
            tokenized_bert_words = []
            seq_ids = []
            seq_id = 0
            for word in words:
                if word == "[PAD]":
                    tokenized_bert_words.append(word)
                    seq_ids.extend([seq_id])
                    seq_id += 1
                else:
                    token = self.juman_tokenizer.tokenize(word)
                    bert_tokens = self.bert_tokenizer.tokenize(" ".join(token))
                    if len(bert_tokens) == 0:
                        bert_tokens = [" "]
                    tokenized_bert_words.extend(bert_tokens)
                    num_id = 1 if len(bert_tokens) == 0 else len(bert_tokens)
                    seq_ids.extend([seq_id] * num_id)
                    seq_id += 1
            token = ["[CLS]"] + tokenized_bert_words[:self.max_seq_length] + ["[SEP]"]
            id = self.bert_tokenizer.convert_tokens_to_ids(token)
            batched_bert_words.append(token)
            batched_bert_ids.append(id)
            batched_bert_seq_ids.append([-1] + seq_ids[:self.max_seq_length] + [seq_ids[-1] + 1])

            embedding, _ = self.model(torch.tensor(id).reshape(1, -1).to(self.device))
            batched_bert_embs.extend(embedding)

        dup_ids = []
        dup_tokens = []
        dup_embs = None
        for seq_ids, ids, tokens, embs in zip(batched_bert_seq_ids, batched_bert_ids, batched_bert_words, batched_bert_embs):
            dup_id = []
            dup_token = []
            dup_emb = None

            def get_duplicated_index(l, x):
                return [i for i, _x in enumerate(l) if _x == x]

            current_pos = 0
            for i in range(min(seq_ids), max(seq_ids) + 1):
                indexes = get_duplicated_index(seq_ids, i)
                if len(indexes) == 1:
                    dup_id.append(int(ids[indexes[0]]))
                    dup_token.append(tokens[indexes[0]])
                    if dup_emb is None:
                        dup_emb = embs[indexes[0]].unsqueeze(0)
                    else:
                        dup_emb = torch.cat((dup_emb, embs[indexes[0]].unsqueeze(0)), dim=0)
                else:
                    if len(indexes) == 0:
                        pass
                    else:
                        dup_id.append([int(ids[ii]) for ii in indexes])
                        dup_token.append([tokens[ii] for ii in indexes])
                        if dup_emb is None:
                            dup_emb = torch.mean(itemgetter(indexes)(embs), dim=0).unsqueeze(0)
                        else:
                            dup_emb = torch.cat((dup_emb, torch.mean(itemgetter(indexes)(embs), dim=0).unsqueeze(0)), dim=0)
                if tokens[current_pos] == "[SEP]":
                    break
                current_pos += len(indexes)

            dup_ids.append(dup_id)
            dup_tokens.append(dup_token)
            if dup_embs is None:
                dup_embs = dup_emb.unsqueeze(0)
            else:
                dup_embs = torch.cat((dup_embs, dup_emb.unsqueeze(0)), dim=0)

        shlinked_id = []
        shlinked_token = []
        shlinked_embedding = None
        for id, token, emb in zip(dup_ids, dup_tokens, dup_embs):
            shlinked_id.append(id[1:-1])
            shlinked_token.append(token[1:-1])
            if shlinked_embedding is None:
                shlinked_embedding = emb[1:-1].unsqueeze(0)
            else:
                shlinked_embedding = torch.cat((shlinked_embedding, emb[1:-1].unsqueeze(0)), dim=0)

        return {"embedding": shlinked_embedding, "token": shlinked_token, "id": shlinked_id}

    def get_pred_embedding(self, batched_arg_embedding, batched_arg_token, batched_word_pos, pred_pos_index):
        batched_pred_embedding = None
        for word_pos, arg_embedding, arg_token in zip(batched_word_pos, batched_arg_embedding, batched_arg_token):
            index = int((word_pos == pred_pos_index).nonzero().reshape(-1))
            if batched_pred_embedding is None:
                batched_pred_embedding = arg_embedding[index].repeat(len(arg_embedding)).view(-1, len(arg_embedding[index])).unsqueeze(0)
            else:
                batched_pred_embedding = torch.cat((batched_pred_embedding, arg_embedding[index].repeat(len(arg_embedding)).view(-1, len(arg_embedding[index])).unsqueeze(0)), dim=0)
            if "[PAD]" in arg_token:
                pad_from = arg_token.index("[PAD]")
                for i in range(pad_from, len(batched_pred_embedding[-1])):
                    batched_pred_embedding[-1][i] = arg_embedding[i]
        return {"embedding": batched_pred_embedding, "id": pred_pos_index}

    def get_embedding(self, text):
        items = self.get_sentence_embedding(text)

        tokens = items["tokens"]
        embeddings = items["embedding"]

        pool_tensors = torch.tensor()
        for token, embedding in zip(tokens, embeddings):
            if "##" in token:
                pass
            else:
                pool_tensors.append(embedding)
        return pool_tensors

    def state_dict(self):
        return self.model.state_dict()

    def get_padding_idx(self):
        return self.model.embeddings.word_embeddings.padding_idx


if __name__ == "__main__":
    _path = Path("../../data/bert-kyoto/Japanese_L-12_H-768_A-12_E-30_BPE").resolve()
    model = BertWithJumanModel(_path, trainable=True)
    ret = model.get_word_embedding(u"どういたしまして．")
    print(ret)
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))


