from __future__ import print_function, unicode_literals

from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BertTokenizer

import six
import sys
import codecs
import re
import os
import argparse
from collections import defaultdict
import logging

import MeCab


class WSCTask(object):
    def __init__(self, task_id, j_sentence, j_target, j_candidate_string, j_answer):
        self.task_id = task_id

        self.j_sentence = j_sentence
        self.j_target = j_target
        assert (self.j_target in self.j_sentence)

        self.j_candidate_string = j_candidate_string
        self.j_candidates = re.split(r"、　?", self.j_candidate_string)
        for candidate in self.j_candidates:
            assert (candidate in self.j_sentence)
        assert (len(self.j_candidates) == 2)

        self.j_answer = j_answer
        assert ("、" not in self.j_answer)
        assert (self.j_answer in self.j_candidates)

    def __str__(self):
        ret_str = ""
        ret_str = "Task ID: {}\n".format(self.task_id)
        ret_str += "{}\n".format(self.j_sentence)
        ret_str += "target:\t{}\n".format(self.j_target)
        ret_str += "cands:\t{}\n".format(",".join(self.j_candidates))
        ret_str += "answer:\t{}\n".format(self.j_answer)

        return ret_str


class WSCJaReader(object):
    def __init__(self):
        pass

    def read(self, file):
        # The bee landed on the flower because it had pollen.	ハチが花にとまった。それが花粉を持っていたからだ。		ハチは花粉があったので花にとまった。	（φに）花粉があったので、ハチは花にとまった。	ハチが花にとまった。Φが花粉を持っていたからだ。
        # it	それ
        # "The bee,the flower"	ハチ、花
        # the flower	花

        tasks = []

        line_num = 1
        task_id = 1
        e_sentence, j_sentence = None, None
        e_target, j_target = None, None
        e_candidate_string, j_candidate_string = None, None
        e_answer, j_answer = None, None

        for line in codecs.open(file, 'r', 'utf-8'):
            line = line.strip()
            remainder = line_num % 5
            if remainder == 1:
                e_sentence, j_sentence, *comment = line.split("\t")
            elif remainder == 2:
                e_target, j_target, *comment = line.split("\t")
            elif remainder == 3:
                e_candidate_string, j_candidate_string, *comment = line.split("\t")
            elif remainder == 4:
                e_answer, j_answer, *comment = line.split("\t")
            else:
                wsc_task = WSCTask(task_id, j_sentence, j_target, j_candidate_string, j_answer)
                # check whether this task is consistent with its counterpart
                if wsc_task.task_id % 2 == 0:
                    assert (wsc_task.j_candidates == tasks[-1].j_candidates)
                    assert (wsc_task.j_answer != tasks[-1].j_answer)

                tasks.append(wsc_task)
                task_id += 1
            line_num += 1

        return tasks

class GenerateWSCDataset:
    def __init__(self, mode='train'):
        wsc_ja_reader = WSCJaReader()
        data_set = wsc_ja_reader.read(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}.txt')
        mecab = MeCab.Tagger("-Owakati")

        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

        bert_tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        t5_tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")

        write_mecab, write_gpt2, write_bert, write_t5 = False, False, False, True
        f_mecab = Path(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}-mecab.txt').open('w') if write_mecab else None
        f_gpt2 = Path(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}-gpt2.txt').open('w') if write_gpt2 else None
        f_bert = Path(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}-bert.txt').open('w') if write_bert else None
        f_t5 = Path(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}-t5.txt').open('w') if write_t5 else None

        batch_words = []
        for data in data_set:
            words = mecab.parse(data.j_sentence).strip().split(' ')
            labels = ['1' if word == data.j_answer else '0' for word in words]
            pronouns = ['1' if data.j_target in word else '0' for word in words]
            if f_mecab is not None:
                if labels.count('1') == 1 and pronouns.count('1') == 1:
                    f_mecab.write(f">{data.j_sentence}\n")
                    f_mecab.write(f"{' '.join(words)}\n")
                    f_mecab.write(f"{' '.join([data.j_target] * len(words))}\n")
                    f_mecab.write(f"{' '.join(labels)}\n")
                    f_mecab.write(f"{' '.join([data.j_answer] * len(words))}\n")
                    f_mecab.write(f"{' '.join(pronouns)}\n")
                else:
                    f_mecab.write(f">{data.j_sentence}\n")
                    f_mecab.write(f"{' '.join(words)}\n")
                    f_mecab.write(f"{' '.join([data.j_target] * len(words))}\n")
                    if labels.count('1') != 1:
                        f_mecab.write(f"***{' '.join(labels)}\n")
                    else:
                        f_mecab.write(f"{' '.join(labels)}\n")

                    f_mecab.write(f"{' '.join([data.j_answer] * len(words))}\n")
                    if pronouns.count('1') != 1:
                        f_mecab.write(f"***{' '.join(pronouns)}\n")
                    else:
                        f_mecab.write(f"{' '.join(pronouns)}\n")

            # print("--------------------")
            ids_gpt2_words = tokenizer(data.j_sentence.strip())
            gpt2_words = tokenizer.convert_ids_to_tokens(ids_gpt2_words['input_ids'])
            labels = ['1' if word == data.j_answer else '0' for word in gpt2_words]
            pronouns = ['1' if data.j_target in word else '0' for word in gpt2_words]
            if f_gpt2 is not None:
                if labels.count('1') == 1 and pronouns.count('1') == 1:
                    f_gpt2.write(f">{data.j_sentence}\n")
                    f_gpt2.write(f"{' '.join(gpt2_words)}\n")
                    f_gpt2.write(f"{' '.join([data.j_target] * len(gpt2_words))}\n")
                    f_gpt2.write(f"{' '.join(labels)}\n")
                    f_gpt2.write(f"{' '.join([data.j_answer] * len(gpt2_words))}\n")
                    f_gpt2.write(f"{' '.join(map(str, ids_gpt2_words['input_ids']))}\n")
                    f_gpt2.write(f"{' '.join(map(str, ids_gpt2_words['attention_mask']))}\n")
                    f_gpt2.write(f"{' '.join(pronouns)}\n")
                else:
                    f_gpt2.write(f">{data.j_sentence}\n")
                    f_gpt2.write(f"{' '.join(gpt2_words)}\n")
                    f_gpt2.write(f"{' '.join([data.j_target] * len(gpt2_words))}\n")
                    if labels.count('1') != 1:
                        f_gpt2.write(f"***{' '.join(labels)}\n")
                    else:
                        f_gpt2.write(f"{' '.join(labels)}\n")
                    f_gpt2.write(f"{' '.join([data.j_answer] * len(gpt2_words))}\n")
                    f_gpt2.write(f"{' '.join(map(str, ids_gpt2_words['input_ids']))}\n")
                    f_gpt2.write(f"{' '.join(map(str, ids_gpt2_words['attention_mask']))}\n")
                    if pronouns.count('1') != 1:
                        f_gpt2.write(f"***{' '.join(pronouns)}\n")
                    else:
                        f_gpt2.write(f"{' '.join(pronouns)}\n")

            ids_bert_words = bert_tokenizer(data.j_sentence.strip())
            bert_words = bert_tokenizer.convert_ids_to_tokens(ids_bert_words['input_ids'])
            labels = ['1' if word == data.j_answer else '0' for word in bert_words]
            pronouns = ['1' if data.j_target in word else '0' for word in bert_words]
            if f_bert is not None:
                if labels.count('1') == 1 and pronouns.count('1') == 1:
                    f_bert.write(f">{data.j_sentence}\n")
                    f_bert.write(f"{' '.join(bert_words)}\n")
                    f_bert.write(f"{' '.join([data.j_target] * len(bert_words))}\n")
                    f_bert.write(f"{' '.join(labels)}\n")
                    f_bert.write(f"{' '.join([data.j_answer] * len(bert_words))}\n")
                    f_bert.write(f"{' '.join(map(str, ids_bert_words['input_ids']))}\n")
                    f_bert.write(f"{' '.join(map(str, ids_bert_words['attention_mask']))}\n")
                    f_bert.write(f"{' '.join(pronouns)}\n")
                else:
                    f_bert.write(f">{data.j_sentence}\n")
                    f_bert.write(f"{' '.join(bert_words)}\n")
                    f_bert.write(f"{' '.join([data.j_target] * len(bert_words))}\n")
                    if labels.count('1') != 1:
                        f_bert.write(f"***{' '.join(labels)}\n")
                    else:
                        f_bert.write(f"{' '.join(labels)}\n")
                    f_bert.write(f"{' '.join([data.j_answer] * len(bert_words))}\n")
                    f_bert.write(f"{' '.join(map(str, ids_bert_words['input_ids']))}\n")
                    f_bert.write(f"{' '.join(map(str, ids_bert_words['attention_mask']))}\n")
                    if pronouns.count('1') != 1:
                        f_bert.write(f"***{' '.join(pronouns)}\n")
                    else:
                        f_bert.write(f"{' '.join(pronouns)}\n")

            ids_t5_words = t5_tokenizer(data.j_sentence.strip())
            t5_words = t5_tokenizer.convert_ids_to_tokens(ids_t5_words['input_ids'])
            labels = ['1' if word == data.j_answer else '0' for word in t5_words]
            pronouns = ['1' if data.j_target in word else '0' for word in t5_words]
            if f_t5 is not None:
                if labels.count('1') == 1 and pronouns.count('1') == 1:
                    f_t5.write(f">{data.j_sentence}\n")
                    f_t5.write(f"{' '.join(t5_words)}\n")
                    f_t5.write(f"{' '.join([data.j_target] * len(t5_words))}\n")
                    f_t5.write(f"{' '.join(labels)}\n")
                    f_t5.write(f"{' '.join([data.j_answer] * len(t5_words))}\n")
                    f_t5.write(f"{' '.join(map(str, ids_t5_words['input_ids']))}\n")
                    f_t5.write(f"{' '.join(map(str, ids_t5_words['attention_mask']))}\n")
                    f_t5.write(f"{' '.join(pronouns)}\n")
                else:
                    f_t5.write(f">{data.j_sentence}\n")
                    f_t5.write(f"{' '.join(t5_words)}\n")
                    f_t5.write(f"{' '.join([data.j_target] * len(t5_words))}\n")
                    if labels.count('1') != 1:
                        f_t5.write(f"***{' '.join(labels)}\n")
                    else:
                        f_t5.write(f"{' '.join(labels)}\n")
                    f_t5.write(f"{' '.join([data.j_answer] * len(t5_words))}\n")
                    f_t5.write(f"{' '.join(map(str, ids_t5_words['input_ids']))}\n")
                    f_t5.write(f"{' '.join(map(str, ids_t5_words['attention_mask']))}\n")
                    if pronouns.count('1') != 1:
                        f_t5.write(f"***{' '.join(pronouns)}\n")
                    else:
                        f_t5.write(f"{' '.join(pronouns)}\n")

        if f_mecab is not None:
            f_mecab.close()
        if f_gpt2 is not None:
            f_gpt2.close()
        if f_bert is not None:
            f_bert.close()
        if f_t5 is not None:
            f_t5.close()


class WSCGPT2Dataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt').open('r') as f:
                texts = f.readlines()
        else:
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-token.txt').open('r') as f:
                texts = f.readlines()

        self.token_datasets = {}
        for i in range(0, len(texts), 5):
            key = texts[i].strip()[1:]
            record = {}
            record['tokens'] = [t for t in key]
            record['candidates'] = texts[i + 1].strip().split(' ')
            record['pronoun_pos'] = texts[i + 2].strip().split(' ')
            record['answer_pos'] = texts[i + 3].strip().split(' ')
            record['candidate_pos'] = texts[i + 4].strip().split(' ')
            self.token_datasets[key] = record

        if mode == 'train':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-gpt2.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-gpt2.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'test':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-gpt2.txt').open('r') as f:
                texts = f.readlines()

        self.datasets = []
        for i in range(0, len(texts), 8):
            table = {}
            table['tokens'] = texts[i + 1].strip().split(' ')
            table['pronoun'] = texts[i + 2].strip().split(' ')
            table['labels'] = texts[i + 3].strip().split(' ')
            table['target'] = texts[i + 4].strip().split(' ')
            table['token_ids'] = texts[i + 5].strip().split(' ')
            table['attention_heads'] = texts[i + 6].strip().split(' ')
            table['pronoun_labels'] = texts[i + 7].strip().split(' ')

            pronoun_labels, answer_labels, candidate_labels = [], [], []
            current_index, end_index = 0, 0
            sentence = texts[i].strip()[1:] # ''.join(table['tokens'][1:-1])
            for raw_t in table['tokens']:
                t = raw_t.replace('▁', '')
                start_index = current_index
                end_index = start_index + len(t)

                if '1' in self.token_datasets[sentence]['pronoun_pos'][start_index: end_index]:
                    pronoun_labels.append('1')
                else:
                    pronoun_labels.append('0')

                if '1' in self.token_datasets[sentence]['answer_pos'][start_index: end_index]:
                    answer_labels.append('1')
                else:
                    answer_labels.append('0')

                if '1' in self.token_datasets[sentence]['candidate_pos'][start_index: end_index]:
                    candidate_labels.append('1')
                else:
                    candidate_labels.append('0')

                current_index = end_index

            table['pronoun_labels'] = pronoun_labels
            table['answer_labels'] = answer_labels
            table['candidate_labels'] = candidate_labels
            self.datasets.append(table.copy())
        if mode == 'train':
            self.datasets = self.datasets[:1000]
        elif mode == 'dev':
            self.datasets = self.datasets[1000:]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        length = len(self.datasets[idx]['token_ids'])
        # d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode
        return self.datasets[idx]['token_ids'],\
               self.datasets[idx]['pronoun_labels'],\
               self.datasets[idx]['answer_labels'],\
               self.datasets[idx]['candidate_labels']

class WSCBERTDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt').open('r') as f:
                texts = f.readlines()
        else:
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-token.txt').open('r') as f:
                texts = f.readlines()

        self.token_datasets = {}
        for i in range(0, len(texts), 5):
            key = texts[i].strip()[1:]
            record = {}
            record['tokens'] = [t for t in key]
            record['candidates'] = texts[i + 1].strip().split(' ')
            record['pronoun_pos'] = texts[i + 2].strip().split(' ')
            record['answer_pos'] = texts[i + 3].strip().split(' ')
            record['candidate_pos'] = texts[i + 4].strip().split(' ')
            self.token_datasets[key] = record

        if mode == 'train':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-bert.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/old/train-bert.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'test':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-bert.txt').open('r') as f:
                texts = f.readlines()

        self.datasets = []
        for i in range(0, len(texts), 8):
            table = {}
            table['tokens'] = texts[i + 1].strip().split(' ')
            table['pronoun'] = texts[i + 2].strip().split(' ')
            table['labels'] = texts[i + 3].strip().split(' ')
            table['target'] = texts[i + 4].strip().split(' ')
            table['token_ids'] = texts[i + 5].strip().split(' ')
            table['attention_heads'] = texts[i + 6].strip().split(' ')
            table['pronoun_labels'] = texts[i + 7].strip().split(' ')

            pronoun_labels, answer_labels, candidate_labels = [], [], []
            current_index, end_index = 0, 0
            sentence = texts[i].strip()[1:] # ''.join(table['tokens'][1:-1])
            for raw_t in table['tokens']:
                if raw_t == '[CLS]':
                    raw_t = ''
                t = raw_t.replace('##', '')
                start_index = current_index
                end_index = start_index + len(t)

                if '1' in self.token_datasets[sentence]['pronoun_pos'][start_index: end_index]:
                    pronoun_labels.append('1')
                else:
                    pronoun_labels.append('0')

                if '1' in self.token_datasets[sentence]['answer_pos'][start_index: end_index]:
                    answer_labels.append('1')
                else:
                    answer_labels.append('0')

                if '1' in self.token_datasets[sentence]['candidate_pos'][start_index: end_index]:
                    candidate_labels.append('1')
                else:
                    candidate_labels.append('0')

                current_index = end_index

            table['pronoun_labels'] = pronoun_labels
            table['answer_labels'] = answer_labels
            table['candidate_labels'] = candidate_labels
            self.datasets.append(table.copy())
        if mode == 'train':
            self.datasets = self.datasets[:1000]
        elif mode == 'dev':
            self.datasets = self.datasets[1000:]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        length = len(self.datasets[idx]['token_ids'])
        # d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode
        return self.datasets[idx]['token_ids'],\
               self.datasets[idx]['pronoun_labels'],\
               self.datasets[idx]['answer_labels'],\
               self.datasets[idx]['candidate_labels']


class WSCT5Dataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt').open('r') as f:
                texts = f.readlines()
        else:
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-token.txt').open('r') as f:
                texts = f.readlines()

        self.token_datasets = {}
        for i in range(0, len(texts), 5):
            key = texts[i].strip()[1:]
            record = {}
            record['tokens'] = [t for t in key]
            record['candidates'] = texts[i + 1].strip().split(' ')
            record['pronoun_pos'] = texts[i + 2].strip().split(' ')
            record['answer_pos'] = texts[i + 3].strip().split(' ')
            record['candidate_pos'] = texts[i + 4].strip().split(' ')
            self.token_datasets[key] = record

        if mode == 'train':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-t5.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-t5.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'test':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-t5.txt').open('r') as f:
                texts = f.readlines()

        self.datasets = []
        for i in range(0, len(texts), 8):
            table = {}
            table['tokens'] = texts[i + 1].strip().split(' ')
            table['pronoun'] = texts[i + 2].strip().split(' ')
            table['labels'] = texts[i + 3].strip().split(' ')
            table['target'] = texts[i + 4].strip().split(' ')
            table['token_ids'] = texts[i + 5].strip().split(' ')
            table['attention_heads'] = texts[i + 6].strip().split(' ')
            table['pronoun_labels'] = texts[i + 7].strip().split(' ')

            pronoun_labels, answer_labels, candidate_labels = [], [], []
            current_index, end_index = 0, 0
            sentence = texts[i].strip()[1:] # ''.join(table['tokens'][1:-1])
            for raw_t in table['tokens']:
                t = raw_t.replace('▁', '')
                start_index = current_index
                end_index = start_index + len(t)

                if '1' in self.token_datasets[sentence]['pronoun_pos'][start_index: end_index]:
                    pronoun_labels.append('1')
                else:
                    pronoun_labels.append('0')

                if '1' in self.token_datasets[sentence]['answer_pos'][start_index: end_index]:
                    answer_labels.append('1')
                else:
                    answer_labels.append('0')

                if '1' in self.token_datasets[sentence]['candidate_pos'][start_index: end_index]:
                    candidate_labels.append('1')
                else:
                    candidate_labels.append('0')

                current_index = end_index

            table['pronoun_labels'] = pronoun_labels
            table['answer_labels'] = answer_labels
            table['candidate_labels'] = candidate_labels
            self.datasets.append(table.copy())
        if mode == 'train':
            self.datasets = self.datasets[:1000]
        elif mode == 'dev':
            self.datasets = self.datasets[1000:]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        length = len(self.datasets[idx]['token_ids'])
        # d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode
        return self.datasets[idx]['token_ids'],\
               self.datasets[idx]['pronoun_labels'],\
               self.datasets[idx]['answer_labels'],\
               self.datasets[idx]['candidate_labels']


class WSCDatasetForShots(Dataset):
    def __init__(self, mode='train'):
        wsc_ja_reader = WSCJaReader()
        self.datasets = wsc_ja_reader.read(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}.txt')

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return [self.datasets[idx].j_sentence,\
               self.datasets[idx].j_candidates,\
               self.datasets[idx].j_answer,\
               self.datasets[idx].j_target,\
               int(self.datasets[idx].task_id)]


class WSCMecabDataset(Dataset):
    def __init__(self, mode='train'):
        # preparing vocabrary
        with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-mecab.txt').open('r') as f:
            texts = f.readlines()
        self.word_to_id = {'<unk>': 0, '<pad>': 1}
        self.id_to_word = {0: '<unk>', 1: '<pad>'}
        index = 2
        for i in range(0, len(texts), 6):
            words = texts[i + 1].strip().split(' ')
            for word in words:
                if word not in self.word_to_id.keys():
                    self.word_to_id[word] = index
                    self.id_to_word[index] = word
                    index += 1

        # token datasets
        if mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt').open('r') as f:
                texts = f.readlines()
        else:
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-token.txt').open('r') as f:
                texts = f.readlines()

        self.token_datasets = {}
        for i in range(0, len(texts), 5):
            key = texts[i].strip()[1:]
            record = {}
            record['tokens'] = [t for t in key]
            record['candidates'] = texts[i + 1].strip().split(' ')
            record['pronoun_pos'] = texts[i + 2].strip().split(' ')
            record['answer_pos'] = texts[i + 3].strip().split(' ')
            record['candidate_pos'] = texts[i + 4].strip().split(' ')
            self.token_datasets[key] = record

        if mode == 'train':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-mecab.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'dev':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/train-mecab.txt').open('r') as f:
                texts = f.readlines()
        elif mode == 'test':
            with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/fixed/{mode}-mecab.txt').open('r') as f:
                texts = f.readlines()

        self.datasets = []
        for i in range(0, len(texts), 6):
            table = {}
            table['tokens'] = texts[i + 1].strip().split(' ')
            table['token_ids'] = [self.word_to_id[t] if t in self.word_to_id.keys() else self.word_to_id['<unk>'] for t in texts[i + 1].strip().split(' ')]
            table['pronoun'] = texts[i + 2].strip().split(' ')
            table['labels'] = texts[i + 3].strip().split(' ')
            table['target'] = texts[i + 4].strip().split(' ')
            table['pronoun_labels'] = texts[i + 5].strip().split(' ')

            pronoun_labels, answer_labels, candidate_labels = [], [], []
            current_index, end_index = 0, 0
            sentence = texts[i].strip()[1:] # ''.join(table['tokens'][1:-1])
            for raw_t in table['tokens']:
                t = raw_t.replace('▁', '')
                start_index = current_index
                end_index = start_index + len(t)

                if '1' in self.token_datasets[sentence]['pronoun_pos'][start_index: end_index]:
                    pronoun_labels.append('1')
                else:
                    pronoun_labels.append('0')

                if '1' in self.token_datasets[sentence]['answer_pos'][start_index: end_index]:
                    answer_labels.append('1')
                else:
                    answer_labels.append('0')

                if '1' in self.token_datasets[sentence]['candidate_pos'][start_index: end_index]:
                    candidate_labels.append('1')
                else:
                    candidate_labels.append('0')

                current_index = end_index

            table['pronoun_labels'] = pronoun_labels
            table['answer_labels'] = answer_labels
            table['candidate_labels'] = candidate_labels
            self.datasets.append(table.copy())
        if mode == 'train':
            self.datasets = self.datasets[:1000]
        elif mode == 'dev':
            self.datasets = self.datasets[1000:]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        length = len(self.datasets[idx]['token_ids'])
        # d_args, d_preds, d_labels, d_props, d_word_pos, d_ku_pos, d_mode
        return self.datasets[idx]['token_ids'],\
               self.datasets[idx]['pronoun_labels'],\
               self.datasets[idx]['answer_labels'],\
               self.datasets[idx]['candidate_labels'],\
               self.datasets[idx]['tokens']

if __name__ == '__main__':
    print('[train]')
    dataset = GenerateWSCDataset(mode='train')
    print('[test]')
    dataset = GenerateWSCDataset(mode='test')

    # WSCDataset(mode='train')
    # WSCGPT2Dataset(mode='train')
    # WSCMecabDataset(mode='test')
    # WSCBERTDataset(mode='test')

