# -*- coding: utf-8 -*-
from pathlib import Path
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from transformers import T5Tokenizer, AutoModelForCausalLM
import os
import sys

sys.path.append(os.pardir)
from utils.HelperFunctions import get_cuda_id
from Model import Model


class GPT2Model(Model):
    def __init__(self,
                 vocab_file_name="vocab.txt",
                 device='cpu',
                 trainable=False):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        for k, v in self.model.named_parameters():
            v.requires_grad = trainable
        self.device = device
        self.embedding_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.max_seq_length = 224

        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.slice = 'None'

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def get_sentence_embedding(self, text):
        token = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.tokenize(" ".join(token))
        token = ["[CLS]"] + tokens[:self.max_seq_length] + ["[SEP]"]
        id = self.tokenizer.encode(token) # max_seq_len-2

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

    def get_word_embedding(self, batched_words, dup_mode='mean'):
        batched_tokens, batched_ids, batched_embs, batched_seq_ids = [], [], [], []
        for words in batched_words:
            tokenized_words, seq_ids = [], []
            seq_id = 0
            for word in words:
                if word == "[PAD]":
                    tokenized_words.append(word)
                    seq_ids.extend([seq_id])
                    seq_id += 1
                else:
                    token = self.tokenizer.tokenize(word)
                    # tokens = self.tokenizer.tokenize(" ".join(token))
                    token = [t for t in token if t != "▁"]
                    if len(token) == 0:
                        token = [" "]
                    tokenized_words.extend(token)
                    num_id = 1 if len(token) == 0 else len(token)
                    seq_ids.extend([seq_id] * num_id)
                    seq_id += 1
            tokens = ["▁"] + tokenized_words[:self.max_seq_length] + ["<\s>"]
            ids = [9] + self.tokenizer.convert_tokens_to_ids(tokenized_words[:self.max_seq_length]) + [2]
            batched_tokens.append(tokens)
            batched_ids.append(ids)
            batched_seq_ids.append([-1] + seq_ids[:self.max_seq_length] + [seq_ids[-1] + 1])

            inputs = self.tokenizer(''.join(words), return_tensors='pt')
            inputs.data['input_ids'] = torch.LongTensor([[9] + ids + [2]]).to(self.device)
            inputs.data['attention_mask'] = torch.LongTensor([[1] * (len(ids) + 2)]).to(self.device)

            embedding = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
            batched_embs.extend(embedding)

        dup_ids, dup_tokens, dup_embs = [], [], []
        for seq_ids, ids, tokens, embs in zip(batched_seq_ids, batched_ids, batched_tokens, batched_embs):
            dup_id, dup_token, dup_emb = [], [], []

            def get_duplicated_index(l, x):
                return [i for i, _x in enumerate(l) if _x == x]

            current_pos = 0
            for i in range(min(seq_ids), max(seq_ids) + 1):
                indexes = get_duplicated_index(seq_ids, i)
                if len(indexes) == 1:
                    dup_id.append(int(ids[indexes[0]]))
                    dup_token.append(tokens[indexes[0]])
                    dup_emb.append(embs[indexes[0]])
                else:
                    if len(indexes) == 0:
                        pass
                    else:
                        dup_id.append([int(ids[ii]) for ii in indexes])
                        dup_token.append([tokens[ii] for ii in indexes])
                        dup_emb.append([embs[ii] for ii in indexes])
                        dup_emb[-1] = torch.mean(torch.stack(dup_emb[-1]), dim=0)
                if self.slice == 'sep':
                    if tokens[current_pos] == "[SEP]":
                        break
                current_pos += len(indexes)

            dup_ids.append(dup_id)
            dup_tokens.append(dup_token)
            dup_embs.append(dup_emb)

            # if dup_emb.shape[0] < max_sentence_length - 1:
            #     dup_emb = torch.cat((dup_emb, torch.zeros(max_sentence_length - dup_emb.shape[0] - 1, dup_emb.shape[1])))
            #
            # if dup_embs is None:
            #     dup_embs = dup_emb.unsqueeze(0)
            # else:
            #     dup_embs = torch.cat((dup_embs, dup_emb.unsqueeze(0)), dim=0)

        max_sentence_length = max([len(dup_token) - 2 for dup_token in dup_tokens])
        shlinked_id = []
        shlinked_token = []
        shlinked_embedding = []
        for id, token, emb in zip(dup_ids, dup_tokens, dup_embs):
            shlinked_id.append(id[1:-1] + [0] * (max_sentence_length - len(emb[1:-1])))
            shlinked_token.append(token[1:-1] + ['[PAD]'] * (max_sentence_length - len(emb[1:-1])))
            shlinked_embedding.append(torch.stack(emb[1:-1] + [torch.zeros((emb[0].shape[0]), device=self.device)] * (max_sentence_length - len(emb[1:-1]))))
        shlinked_embedding = torch.stack(shlinked_embedding)

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
        return self.tokenizer.pad_token_id


if __name__ == "__main__":
    model = GPT2Model(trainable=False)
    ret = model.get_word_embedding([[u"どう", u"いた", u"しま", u"して", u"。"]], dup_mode='lead')
    print(ret)
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))


