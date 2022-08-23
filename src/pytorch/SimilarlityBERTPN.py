import os
import random

import numpy as np
import torch
import torch.nn as nn

random.seed(0)
torch.manual_seed(0)

from pathlib import Path
import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(0)

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model
from transformers import T5Tokenizer, T5TokenizerFast, AutoModelForCausalLM, T5Model
from transformers import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertJapaneseTokenizer, BertModel
from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast

class PointerNetworks(nn.Module):
    def __init__(self, hyper_parameters={}):
        super(PointerNetworks, self).__init__()
        self.embedding_dim = hyper_parameters['embedding_dim']
        self.hidden_size = self.embedding_dim
        self.num_layers = hyper_parameters['num_layers']
        self.with_use_rnn_repeatedly = hyper_parameters['with_use_rnn_repeatedly']
        self.vocab_size = hyper_parameters['vocab_size']

        if self.with_use_rnn_repeatedly:
            self.encoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.decoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.encoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
            self.decoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size)] * self.num_layers)
        else:
            self.encoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.decoder_f = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.encoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])
            self.decoder_b = nn.ModuleList([nn.GRU(self.embedding_dim, self.hidden_size) for _ in range(self.num_layers)])

        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=hyper_parameters['num_heads'])
        self.linear_logits = nn.Linear(self.hidden_size, 1)
        self.linear_decoder = nn.Linear(self.hidden_size, self.vocab_size)

        self.projection_matrix_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.projection_matrix_decoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.projection_vector = nn.Linear(self.hidden_size, 1, bias=False)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(self.embedding_dim))
        self.decoder_start_input.data.uniform_(
            -(1. / np.sqrt(self.embedding_dim)), 1. / np.sqrt(self.embedding_dim)
        )

    def forward(self, embeddings):
        x = embeddings.transpose(0, 1)

        for i in range(self.num_layers):
            for enc in [self.encoder_f[i], self.encoder_b[i]]:
                encoder_outputs, hidden = enc(x)
                x = encoder_outputs + x
                x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
        encoder_outputs = x.clone()

        x, weights = self.attention(x, x, x)

        for i in range(self.num_layers):
            for dec in [self.decoder_f[i], self.decoder_b[i]]:
                decoder_outputs, hidden = dec(x, hidden)
                x = decoder_outputs + x
                x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
        decoder_outputs = x.clone()
        x = x.transpose(0, 1)

        logits = self.linear_logits(x).squeeze(2)

        x_projected_encoder = self.projection_matrix_encoder(encoder_outputs)
        x_projected_decoder = self.projection_matrix_decoder(decoder_outputs)
        pointer_outputs = self.projection_vector(torch.selu(x_projected_encoder + x_projected_decoder))

        decoder_outputs = self.linear_decoder(x).transpose(1, 2)

        return {'logits': logits, 'decoder_outputs': decoder_outputs, 'pointer_outputs': pointer_outputs}

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.pn', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.pn', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.pn', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.pn', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.pn', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.pn', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-large-japanese.pn', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.pn', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', '../../results/xlm-roberta-large.pn', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', '../../results/xlm-roberta-base.pn', 100

def get_datasets(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    sentences, candidates, pronouns, positive_labels, negative_labels = [], [], [], [], []
    for i in range(0, len(texts), 5):
        sentences.append(texts[i + 0].strip()[1:])
        candidates.append(texts[i + 1].strip().split(' '))
        pronouns.append(texts[i + 2].strip().split(' '))
        positive_labels.append(list(map(int, texts[i + 3].strip().split(' '))))
        negative_labels.append(list(map(int, texts[i + 4].strip().split(' '))))

    return {'sentences': sentences, 'candidates': candidates, 'pronouns': pronouns, 'positive_labels': positive_labels, 'negative_labels': negative_labels}

def get_label(offsets, labels):
    index = labels.index(1)
    for i, offset in enumerate(offsets[1:]):
        if offset[0] <= index < offset[1]:
            return i
    return 0

def get_aggregated_label(offsets, tokens, labels):
    aggregated_labels = []
    for offset in offsets:
        aggregated_labels.append(0)
        items = set(labels[offset[0]: offset[1]])
        if 1 in items:
            aggregated_labels[-1] = 1
    return aggregated_labels

def allocate_data_to_device(data, device='cpu'):
    if device != 'cpu':
        return data.to('cuda:0')
    else:
        return data

def train_model(run_mode='rinna-gpt2'):
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.220823.1'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        tokenizer.do_lower_case = True
        embedding_dim = model.config.hidden_size
    elif 'mbart' in model_name:
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
        embedding_dim = model.config.hidden_size
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        embedding_dim = model.config.d_model
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
        embedding_dim = model.config.hidden_size
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizerFas.from_pretrained(model_name)
        if model_name in set(['rinna-japanese-gpt-1b']):
            embedding_dim = model.embed_dim
        elif model_name in set(['rinna/japanese-roberta-base', 'nlp-waseda/roberta-base-japanese', 'nlp-waseda/roberta-large-japanese', 'xlm-roberta-large', 'xlm-roberta-base']):
            embedding_dim = model.config.hidden_size
        else:
            embedding_dim = model.config.d_model
    model = allocate_data_to_device(model, DEVICE)
    hyper_parameters = {
        'embedding_dim': embedding_dim,
        'vocab_size': tokenizer.vocab_size,
        'with_use_rnn_repeatedly': False,
        'num_layers': 3,
        'num_heads': 1
    }
    pointer_networks = allocate_data_to_device(PointerNetworks(hyper_parameters=hyper_parameters), DEVICE)

    train_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt')
    test_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-token.txt')
    train_sentences, train_labels = train_datasets['sentences'], train_datasets['positive_labels']
    train_sentences, train_labels, dev_sentences, dev_labels = train_sentences[:1000], train_labels[:1000], train_sentences[1000:], train_labels[1000:]
    test_sentences, test_labels = test_datasets['sentences'], test_datasets['positive_labels']
    dev_negative_labels = train_datasets['negative_labels'][1000:]
    test_negative_labels = test_datasets['negative_labels']

    # output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(pointer_networks.parameters()), lr=2e-6 if 'xlm' in model_name else 2e-5, weight_decay=0.01)
    # loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.MultiLabelMarginLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        pointer_networks.train()
        for s, l in zip(train_sentences, train_labels):
            inputs = tokenizer(s, return_tensors='pt')
            # encodings = inputs.encodings[0]
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            pos_index = get_aggregated_label(inputs.data['offset_mapping'], tokenizer.convert_ids_to_tokens(inputs.data['input_ids'][0]), l) if 'tohoku' in model_name else get_aggregated_label(encodings.offsets, encodings.tokens, l)

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            if 'mbart' in model_name:
                h1 = outputs.encoder_last_hidden_state
            elif 't5' in model_name:
                h1 = outputs.last_hidden_state
            else:
                h1 = outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = pointer_networks(h1)['logits']
            o1 = activation_function(o1) if with_activation_function else o1

            loss = loss_func(o1.squeeze(0), torch.as_tensor(pos_index, dtype=torch.float, device=DEVICE))
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
        pointer_networks.eval()
        dev_total_loss = []
        dev_tt, dev_ff = 0, 0
        for s, l, neg_l in zip(dev_sentences, dev_labels, dev_negative_labels):
            inputs = tokenizer(s, return_tensors='pt')
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            # neg_index = get_label(inputs.data['offset_mapping'], neg_l) if 'tohoku' in model_name else get_label(encodings.offsets, neg_l)
            pos_index = get_aggregated_label(inputs.data['offset_mapping'], tokenizer.convert_ids_to_tokens(inputs.data['input_ids'][0]), l) if 'tohoku' in model_name else get_aggregated_label(encodings.offsets, encodings.tokens, l)
            neg_index = get_aggregated_label(inputs.data['offset_mapping'], tokenizer.convert_ids_to_tokens(inputs.data['input_ids'][0]), neg_l) if 'tohoku' in model_name else get_aggregated_label(encodings.offsets, encodings.tokens, neg_l)

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            if 'mbart' in model_name:
                h1 = outputs.encoder_last_hidden_state
            elif 't5' in model_name:
                h1 = outputs.last_hidden_state
            else:
                h1 = outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = pointer_networks(h1)['logits']
            o1 = activation_function(o1) if with_activation_function else o1

            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            loss = loss_func(o1.squeeze(0), torch.as_tensor(pos_index, dtype=torch.float, device=DEVICE))
            dev_total_loss.append(loss.item())

            if torch.max(o1[0][torch.as_tensor(pos_index, dtype=torch.int) == 1]).item() > torch.max(o1[0][torch.as_tensor(neg_index, dtype=torch.int) == 1]).item():
                dev_tt += 1
            else:
                dev_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {l}, {torch.argmax(o1)}, {pos_index}')

        dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
        dev_acc = dev_tt / (dev_tt + dev_ff)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s, l, neg_l in zip(test_sentences, test_labels, test_negative_labels):
            inputs = tokenizer(s, return_tensors='pt')
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            # neg_index = get_label(inputs.data['offset_mapping'], neg_l) if 'tohoku' in model_name else get_label(encodings.offsets, neg_l)
            pos_index = get_aggregated_label(inputs.data['offset_mapping'], tokenizer.convert_ids_to_tokens(inputs.data['input_ids'][0]), l) if 'tohoku' in model_name else get_aggregated_label(encodings.offsets, encodings.tokens, l)
            neg_index = get_aggregated_label(inputs.data['offset_mapping'], tokenizer.convert_ids_to_tokens(inputs.data['input_ids'][0]), neg_l) if 'tohoku' in model_name else get_aggregated_label(encodings.offsets, encodings.tokens, neg_l)

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            if 'mbart' in model_name:
                h1 = outputs.encoder_last_hidden_state
            elif 't5' in model_name:
                h1 = outputs.last_hidden_state
            else:
                h1 = outputs.last_hidden_state
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = pointer_networks(h1)['logits']
            o1 = activation_function(o1) if with_activation_function else o1

            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            loss = loss_func(o1.squeeze(0), torch.as_tensor(pos_index, dtype=torch.float, device=DEVICE))
            test_total_loss.append(loss.item())

            if torch.max(o1[0][torch.as_tensor(pos_index, dtype=torch.int) == 1]).item() > torch.max(o1[0][torch.as_tensor(neg_index, dtype=torch.int) == 1]).item():
                test_tt += 1
            else:
                test_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {l}, {torch.argmax(o1)}, {pos_index}')

        test_total_loss = sum(test_total_loss) / len(test_total_loss)
        test_acc = test_tt / (test_tt + test_ff)

        print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, dev_acc: {dev_acc}, test_loss: {test_total_loss}, test_acc: {test_acc}')
        result_lines.append([e, train_total_loss, dev_total_loss, dev_acc, test_total_loss, test_acc])

    with Path(f'{OUTPUT_PATH}/result.pn.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')


if __name__ == '__main__':
    is_single = False
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    if is_single:
        train_model(run_modes[-2])
    else:
        for run_mode in run_modes:
            train_model(run_mode)


'''
'''
