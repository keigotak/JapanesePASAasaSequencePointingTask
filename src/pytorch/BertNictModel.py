# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel
import os
import sys
sys.path.append(os.pardir)
from utils.HelperFunctions import get_cuda_id


class BertNictModel():
    def __init__(self, bert_path=Path("../../data/NICT_BERT-base_JapaneseWikipedia_100K").resolve(),
                 vocab_file_name="vocab.txt",
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

    def get_word_embedding(self, batched_words):
        batched_bert_ids = []
        for words in batched_words:
            words = [self.bert_tokenizer.cls_token] + words[:self.max_seq_length].tolist() + [self.bert_tokenizer.sep_token]
            id = self.bert_tokenizer.convert_tokens_to_ids(words)
            batched_bert_ids.extend([id])

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
    _path = Path("../../data/NICT_BERT-base_JapaneseWikipedia_100K").resolve()
    model = BertNictModel(_path, trainable=True)
    ret = model.get_word_embedding([["ど", "う", "いたしまし", "て", "。"]])
    print(ret)
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))


