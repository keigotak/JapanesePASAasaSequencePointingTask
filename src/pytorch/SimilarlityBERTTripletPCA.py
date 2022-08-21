import os
import random
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)

import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(0)

from transformers import T5Tokenizer, T5Model
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertModel
from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast
from ValueWatcher import ValueWatcher


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
  # Reference : https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/optimization.py#L33
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.triplet', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.triplet', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.search.triplet', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.triplet', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.triplet', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.triplet', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.triplet', 100

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

# def loss_func(rep_anchor, rep_pos, rep_neg, distance_metric=TripletDistanceMetric.EUCLIDEAN):
#     triplet_margin = float(1.0)
#     distance_pos = distance_metric(rep_anchor, rep_pos)
#     distance_neg = distance_metric(rep_anchor, rep_neg)
#
#     losses = F.relu(distance_pos - distance_neg + triplet_margin)
#     return losses.mean()

def train_model(run_mode='rinna-japanese-gpt-1b'):
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0' # 'cuda:0'
    with_activation_function = False
    with_dropout = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.220330'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        embedding_dim = model.config.hidden_size
        tokenizer.do_lower_case = True
    elif 'mbart' in model_name:
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
        embedding_dim = model.config.hidden_size
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        embedding_dim = model.config.d_model
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
        embedding_dim = model.config.hidden_size
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in set(['rinna/japanese-gpt-1b']):
            embedding_dim = model.embed_dim
        elif model_name in set(['rinna/japanese-roberta-base', 'nlp-waseda/roberta-base-japanese']):
            embedding_dim = model.config.hidden_size
        else:
            embedding_dim = model.config.d_model
    model = allocate_data_to_device(model, DEVICE)

    train_sentence0, train_sentence1, train_sentence2, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    dev_sentence0, dev_sentence1, dev_sentence2, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence0, test_sentence1, test_sentence2, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    output_layer = allocate_data_to_device(torch.nn.Linear(embedding_dim, 1), DEVICE)
    activation_function = torch.nn.SELU()
    dropout_layer = torch.nn.Dropout(0.2)

    alpha = 1.0
    parameters = list(model.parameters()) + list(output_layer.parameters())

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(params=optimizer_grouped_parameters, lr=2e-5, weight_decay=0.01)
    # optimizer = AdamW(params=model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer = AdamW(params=parameters, lr=2e-5, weight_decay=0.01)
    loss_func1 = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    loss_func2 = torch.nn.CrossEntropyLoss()
    loss_func3 = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    # loss_func3 = losses.TripletMarginLoss(distance = CosineSimilarity(),
	# 			     reducer = ThresholdReducer(high=0.3),
	# 		 	     embedding_regularizer = LpRegularizer(p=2))

    # loss_func = torch.nn.CosineSimilarity()
    distance_metric = TripletDistanceMetric.COSINE
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=WARMUP_STEPS)
    vw = ValueWatcher()

    result_lines = []

    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for s0, s1, s2, l in zip(train_sentence0, train_sentence1, train_sentence2, train_labels):
            inputs = tokenizer(s0, return_tensors='pt') # anchor
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h0 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h0 = torch.mean(h0, dim=1)
            o0 = output_layer(dropout_layer(h0)) if with_dropout else output_layer(h0)
            o0 = activation_function(o0) if with_activation_function else o0

            inputs = tokenizer(s1, return_tensors='pt') # negative
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(dropout_layer(h1)) if with_dropout else output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt') # positive
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            o2 = output_layer(dropout_layer(h2)) if with_dropout else output_layer(h2)
            o2 = activation_function(o2) if with_activation_function else o2

            loss1 = loss_func1(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func1(anchor=h0, positive=h2, negative=h1)
            loss2 = loss_func2(torch.cat([o1, o2], dim=1), torch.as_tensor([l], dtype=torch.long, device=DEVICE))
            loss3 = loss_func3(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func3(anchor=h0, positive=h2, negative=h1)
            # loss = alpha.weight * loss1 + beta.weight * loss2
            loss = loss3
            train_total_loss.append(loss.item())
            running_loss.append(loss)
            if len(running_loss) >= BATCH_SIZE:
                running_loss = torch.mean(torch.stack(running_loss), dim=0)
                optimizer.zero_grad(set_to_none=True)
                running_loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss = []
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if len(running_loss) > 0:
            running_loss = torch.mean(torch.stack(running_loss), dim=0)
            optimizer.zero_grad(set_to_none=True)
            running_loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss = []

        # DEV
        with torch.inference_mode():
            model.eval()
            output_layer.eval()
            dev_total_loss = []
            dev_tt, dev_ff = 0, 0
            for s0, s1, s2, l in zip(dev_sentence0, dev_sentence1, dev_sentence2, dev_labels):
                inputs = tokenizer(s0, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h0 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h0 = torch.mean(h0, dim=1)
                o0 = output_layer(h0)
                o0 = activation_function(o0) if with_activation_function else o0

                inputs = tokenizer(s1, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h1 = torch.mean(h1, dim=1)
                o1 = output_layer(h1)
                o1 = activation_function(o1) if with_activation_function else o1

                inputs = tokenizer(s2, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h2 = torch.mean(h2, dim=1)
                o2 = output_layer(h2)
                o2 = activation_function(o2) if with_activation_function else o2

                loss1 = loss_func1(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func1(anchor=h0, positive=h2, negative=h1)
                loss2 = loss_func2(torch.cat([o1, o2], dim=1), torch.as_tensor([l], dtype=torch.long, device=DEVICE))
                loss3 = loss_func3(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func3(anchor=h0, positive=h2, negative=h1)
                # loss = alpha.weight * loss1 + beta.weight * loss2
                loss = loss3
                dev_total_loss.append(loss.item())

                d1, d2 = torch.dist(h0, h1), torch.dist(h0, h2)
                predict_label = 0 if d1 < d2 else 1
                # predict_label = 0 if o2 < o1 else 1
                if predict_label == l:
                    dev_tt += 1
                else:
                    dev_ff += 1

                if with_print_logits:
                    print(f'{o1.item()}, {o2.item()}, {l}, {predict_label}')

            dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
            dev_acc = dev_tt / (dev_tt + dev_ff)

            vw.update(dev_acc)
            if vw.is_updated():
                with Path(f'{OUTPUT_PATH}/{model_name.replace("/", ".")}.{e}.pt').open('wb') as f:
                    torch.save({'model': model.state_dict(), 'output_layer': output_layer.state_dict()}, f)

            # TEST
            test_total_loss = []
            test_tt, test_ff = 0, 0
            for s0, s1, s2, l in zip(test_sentence0, test_sentence1, test_sentence2, test_labels):
                inputs = tokenizer(s0, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h0 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h0 = torch.mean(h0, dim=1)
                o0 = output_layer(h0)
                o0 = activation_function(o0) if with_activation_function else o0

                inputs = tokenizer(s1, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h1 = torch.mean(h1, dim=1)
                o1 = output_layer(h1)
                o1 = activation_function(o1) if with_activation_function else o1

                inputs = tokenizer(s2, return_tensors='pt')
                inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
                outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
                h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
                h2 = torch.mean(h2, dim=1)
                o2 = output_layer(h2)
                o2 = activation_function(o2) if with_activation_function else o2

                loss1 = loss_func1(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func1(anchor=h0, positive=h2, negative=h1)
                loss2 = loss_func2(torch.cat([o1, o2], dim=1), torch.as_tensor([l], dtype=torch.long, device=DEVICE))
                loss3 = loss_func3(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func3(anchor=h0, positive=h2, negative=h1)
                # loss = alpha.weight * loss1 + beta.weight * loss2
                loss = loss3
                test_total_loss.append(loss.item())

                d1, d2 = torch.dist(h0, h1), torch.dist(h0, h2)
                predict_label = 0 if d1 < d2 else 1
                # predict_label = 0 if o2 < o1 else 1

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

    with Path(f'{OUTPUT_PATH}/result.triplet.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')
        f.write(str(alpha) + '\n')

if __name__ == '__main__':
    is_single = True
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'rinna-japanese-gpt-1b']
    if is_single:
        train_model(run_modes[-1])
    else:
        for run_mode in run_modes:
            train_model(run_mode)