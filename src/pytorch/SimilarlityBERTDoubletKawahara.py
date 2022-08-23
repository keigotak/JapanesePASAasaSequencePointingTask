import os
import random

import torch.nn

random.seed(0)
torch.manual_seed(0)

from pathlib import Path
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
import itertools

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.doublet', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.doublet', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.doublet', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.doublet', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.doublet', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.doublet', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-large-japanese.doublet', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.doublet', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', '../../results/xlm-roberta-large.doublet', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', '../../results/xlm-roberta-base.doublet', 100


def get_datasets(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    texts = [text.strip().split('\t') for text in texts]
    s1, s2, s3, l = [], [], [], []
    for text in texts:
        t1 = random.choice([text[1], text[2]])
        if t1 == text[1]:
            s1.append(text[1])
            s2.append(text[2])
            l.append(0)
        else:
            s1.append(text[2])
            s2.append(text[1])
            l.append(1)
        s3.append(text[0])
    return s1, s2, s3, l

def get_jfckb_datasets_for_filtering(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    texts = [text.strip().split('\t') for text in texts]
    lines = [int(text[0][8:])-1 for text in texts]
    sentences = [text[1][6:] + text[2][6:] for text in texts]
    return lines,sentences

def split_by_filter(sentences1, sentences2, sentences3, labels, filter):
    filter = set(filter)
    in_filter_s1, in_filter_s2, in_filter_s3, in_filter_l = [], [], [], []
    not_in_filter_s1, not_in_filter_s2, not_in_filter_s3, not_in_filter_l = [], [], [], []
    for s1, s2, s3, l in zip(sentences1, sentences2, sentences3, labels):
        if s3 in filter:
            in_filter_s1.append(s1)
            in_filter_s2.append(s2)
            in_filter_s3.append(s3)
            in_filter_l.append(l)
        else:
            not_in_filter_s1.append(s1)
            not_in_filter_s2.append(s2)
            not_in_filter_s3.append(s3)
            not_in_filter_l.append(l)
    return {'in_filter': [in_filter_s1, in_filter_s2, in_filter_s3, in_filter_l],
            'not_in_filter': [not_in_filter_s1, not_in_filter_s2, not_in_filter_s3, not_in_filter_l]}


def allocate_data_to_device(data, device='cpu'):
    if device != 'cpu':
        return data.to('cuda:0')
    else:
        return data

def train_model(run_mode='rinna-gpt2'):
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0' # 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.kawahara.220823.1'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    elif 'mbart' in model_name:
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = allocate_data_to_device(model, DEVICE)

    train_sentence1, train_sentence2, train_sentence3, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    dev_sentence1, dev_sentence2, dev_sentence3, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence1, test_sentence2, test_sentence3, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    sentences1 = train_sentence1 + dev_sentence1 + test_sentence1
    sentences2 = train_sentence2 + dev_sentence2 + test_sentence2
    sentences3 = train_sentence3 + dev_sentence3 + test_sentence3
    labels = train_labels + dev_labels + test_labels

    test_ids, test_sentences = get_jfckb_datasets_for_filtering('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/WSC-kawahara/JWSC-LREC2018-JFCKB.tsv')

    test_sentence1, test_sentence2, test_sentence3, test_labels = [], [], [], []
    for test_sentence in test_sentences:
        if test_sentence in sentences3:
            test_id = sentences3.index(test_sentence)
            test_sentence1.append(sentences1[test_id])
            test_sentence2.append(sentences2[test_id])
            test_sentence3.append(sentences3[test_id])
            test_labels.append(labels[test_id])
        else:
            print(test_sentence)

    sentences1 = [sentence1 for i, sentence1 in enumerate(sentences1) if sentences3[i] not in set(test_sentences)]
    sentences2 = [sentence2 for i, sentence2 in enumerate(sentences2) if sentences3[i] not in set(test_sentences)]
    labels = [label for i, label in enumerate(labels) if sentences3[i] not in set(test_sentences)]
    sentences3 = [sentence3 for i, sentence3 in enumerate(sentences3) if sentences3[i] not in set(test_sentences)]

    train_sentence1, train_sentence2, train_sentence3, train_labels = sentences1[:1000], sentences2[:1000], sentences3[:1000], labels[:1000]
    dev_sentence1, dev_sentence2, dev_sentence3, dev_labels = sentences1[1000:], sentences2[1000:], sentences3[1000:], labels[1000:]

    output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=2e-6 if 'xlm' in model_name else 2e-5, weight_decay=0.01)
    # optimizer = AdamW(params=list(output_layer.parameters()), lr=2e-5, weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss()
    vw = ValueWatcher()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for s1, s2, l in zip(train_sentence1, train_sentence2, train_labels):
            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            o2 = output_layer(h2)
            o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(torch.cat([o1, o2], dim=1), torch.as_tensor([l], dtype=torch.long, device=DEVICE))
            train_total_loss.append(loss.item())
            running_loss.append(loss)
            if len(running_loss) >= BATCH_SIZE:
                running_loss = torch.mean(torch.stack(running_loss), dim=0)
                optimizer.zero_grad(set_to_none=True)
                running_loss.backward()
                optimizer.step()
                running_loss = []
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if len(running_loss) > 0:
            running_loss = torch.mean(torch.stack(running_loss), dim=0)
            optimizer.zero_grad(set_to_none=True)
            running_loss.backward()
            optimizer.step()
            running_loss = []

        # DEV
        model.eval()
        output_layer.eval()
        dev_total_loss = []
        dev_tt, dev_ff = 0, 0
        for s1, s2, l in zip(dev_sentence1, dev_sentence2, dev_labels):
            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            o2 = output_layer(h2)
            o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(torch.cat([o1, o2], dim=1), allocate_data_to_device(torch.LongTensor([l]), DEVICE))
            dev_total_loss.append(loss.item())

            if o1 > o2:
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

        vw.update(dev_acc)
        if vw.is_updated():
            with Path(f'{OUTPUT_PATH}/{model_name.replace("/", ".")}.{e}.pt').open('wb') as f:
                torch.save({'model': model.state_dict(), 'output_layer': output_layer.state_dict()}, f)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s1, s2, l in zip(test_sentence1, test_sentence2, test_labels):
            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            o2 = output_layer(h2)
            o2 = activation_function(o2) if with_activation_function else o2

            loss = loss_func(torch.cat([o1, o2], dim=1), allocate_data_to_device(torch.LongTensor([l]), DEVICE))
            test_total_loss.append(loss.item())

            if o1 > o2:
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

    with Path(f'{OUTPUT_PATH}/result.doublet.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')


if __name__ == '__main__':
    is_single = True
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    if is_single:
        train_model(run_modes[-2])
    else:
        for run_mode in run_modes:
            train_model(run_mode)


'''
'''
