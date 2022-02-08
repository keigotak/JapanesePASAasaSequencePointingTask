import random
from enum import Enum

import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

random.seed(0)

from pathlib import Path
import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(0)

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers import InputExample
from sentence_transformers.losses import CosineSimilarityLoss, SoftmaxLoss
from sentence_transformers.evaluation import TripletEvaluator, LabelAccuracyEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.cross_encoder import CrossEncoder

from transformers import GPT2Tokenizer, GPT2Model
from transformers import T5Tokenizer, AutoModelForCausalLM, T5Model
from transformers import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

def get_properties(mode):
    if mode == 'rinna-search':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.search.triplet', 100
    elif mode == 'rinna-best':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.best.triplet', 100
    elif mode == 'tohoku-bert-search':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.search.triplet', 100
    elif mode == 'tohoku-bert-best':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.best.triplet', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.search.triplet', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.triplet', 100

def get_datasets(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    texts = [text.strip().split('\t') for text in texts]
    s0, s1, s2, l = [], [], [], []
    for text in texts:
        t1 = random.choice([text[1], text[2]])
        if t1 == text[1]:
            s0.append(text[0])
            s1.append(text[1])
            s2.append(text[2])
            l.append(0)
        else:
            s0.append(text[0])
            s1.append(text[2])
            s2.append(text[1])
            l.append(1)
    return s0, s1, s2, l

def allocate_data_to_device(data, device='cpu'):
    if device != 'cpu':
        return data.to('cuda:0')
    else:
        return data

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

def loss_func(rep_anchor, rep_pos, rep_neg, distance_metric=TripletDistanceMetric.EUCLIDEAN):
    triplet_margin = float(1.0)
    distance_pos = distance_metric(rep_anchor, rep_pos)
    distance_neg = distance_metric(rep_anchor, rep_neg)

    losses = F.relu(distance_pos - distance_neg + triplet_margin)
    return losses.mean()

def train_model():
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0' # 'cuda:0'
    with_activation_function = False
    with_print_logits = False

    run_mode = 'rinna-search'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.211213'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    # elif 'mbart' in model_name:
    #     model = MBartForConditionalGeneration.from_pretrained(model_name)
    #     tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = allocate_data_to_device(model, DEVICE)

    train_sentence0, train_sentence1, train_sentence2, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    dev_sentence0, dev_sentence1, dev_sentence2, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence0, test_sentence1, test_sentence2, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=2e-5, weight_decay=0.01)
    distance_metric = TripletDistanceMetric.EUCLIDEAN

    result_lines = []
    for e in range(30):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        optimizer.zero_grad()
        for s0, s1, s2, l in zip(train_sentence0, train_sentence1, train_sentence2, train_labels):
            inputs = tokenizer(s0, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h0 = outputs.last_hidden_state
            h0 = torch.mean(h0, dim=1)
            # o0 = output_layer(h0)
            # o0 = activation_function(o0) if with_activation_function else o0

            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h1 = outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h2 = outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            # o2 = output_layer(h2)
            # o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(h0, h1, h2, distance_metric=distance_metric)
            # loss = loss_func(torch.stack([o1, o2], dim=1).squeeze(2), allocate_data_to_device(torch.LongTensor([l]), DEVICE))
            train_total_loss.append(loss.item())
            running_loss.append(loss)
            if len(running_loss) >= BATCH_SIZE:
                running_loss = torch.mean(torch.stack(running_loss), dim=0)
                running_loss.backward()
                optimizer.step()
                running_loss = []
                optimizer.zero_grad()
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if len(running_loss) > 0:
            running_loss = torch.mean(torch.stack(running_loss), dim=0)
            running_loss.backward()
            optimizer.step()
            running_loss = []
            optimizer.zero_grad()

        # DEV
        model.eval()
        output_layer.eval()
        dev_total_loss = []
        dev_tt, dev_ff = 0, 0
        for s0, s1, s2, l in zip(dev_sentence0, dev_sentence1, dev_sentence2, dev_labels):
            inputs = tokenizer(s0, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h0 = outputs.last_hidden_state
            h0 = torch.mean(h0, dim=1)
            # o0 = output_layer(h0)
            # o0 = activation_function(o0) if with_activation_function else o0

            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h1 = outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h2 = outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            # o2 = output_layer(h2)
            # o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(h0, h1, h2, distance_metric=distance_metric)
            dev_total_loss.append(loss.item())

            d1, d2 = distance_metric(h0, h1), distance_metric(h0, h2)
            if d1 < d2:
                predict_label = 0
            else:
                predict_label = 1

            if predict_label == l:
                dev_tt += 1
            else:
                dev_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {o2.item()}, {l}, {predict_label}')

        dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
        dev_acc = dev_tt / (dev_tt + dev_ff)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s0, s1, s2, l in zip(test_sentence0, test_sentence1, test_sentence2, test_labels):
            inputs = tokenizer(s0, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h0 = outputs.last_hidden_state
            h0 = torch.mean(h0, dim=1)
            # o0 = output_layer(h0)
            # o0 = activation_function(o0) if with_activation_function else o0

            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h1 = outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True)
            h2 = outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            # o2 = output_layer(h2)
            # o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(h0, h1, h2, distance_metric=distance_metric)
            test_total_loss.append(loss.item())

            d1, d2 = distance_metric(h0, h1), distance_metric(h0, h2)
            if d1 < d2:
                predict_label = 0
            else:
                predict_label = 1

            if predict_label == l:
                test_tt += 1
            else:
                test_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {o2.item()}, {l}, {predict_label}')

        test_total_loss = sum(test_total_loss) / len(test_total_loss)
        test_acc = test_tt / (test_tt + test_ff)

        print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, dev_acc: {dev_acc}, test_loss: {test_total_loss}, test_acc: {test_acc}')
        result_lines.append([e, train_total_loss, dev_total_loss, dev_acc, test_total_loss, test_acc])

    with Path(f'./result.triplet.{model_name}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')

if __name__ == '__main__':
    train_model()