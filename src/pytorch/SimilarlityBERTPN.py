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
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.search.pn', 100
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

def train_model():
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    run_mode = 'xlm-roberta-base'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.220625'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
        embedding_dim = model.config.hidden_size
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
    optimizer = AdamW(params=list(model.parameters()) + list(pointer_networks.parameters()), lr=2e-5, weight_decay=0.01)
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
    train_model()


'''
rinna-japanese-gpt-1b
../../results/wsc_sbert.rinna-japanese-gpt-1b.pn.220222
Some weights of the model checkpoint at rinna/japanese-gpt-1b were not used when initializing GPT2Model: ['lm_head.weight']
- This IS expected if you are initializing GPT2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing GPT2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
/home/keigo/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use thePyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  FutureWarning,
e: 0, train_loss: 0.49320010157860816, dev_loss: 0.2638821696735317, dev_acc: 0.5, test_loss: 0.2625935527608327, test_acc: 0.5035460992907801
e: 1, train_loss: 0.25393537170439956, dev_loss: 0.253750424901521, dev_acc: 0.5, test_loss: 0.2531131903330485, test_acc: 0.5035460992907801
e: 2, train_loss: 0.24910919696837663, dev_loss: 0.2512320958957169, dev_acc: 0.4968944099378882, test_loss: 0.24779023321226556, test_acc: 0.4929078014184397
e: 3, train_loss: 0.2464297277852893, dev_loss: 0.25680976004704187, dev_acc: 0.5, test_loss: 0.24936795122393057, test_acc: 0.49822695035460995
e: 4, train_loss: 0.24475897684693337, dev_loss: 0.2556564912477635, dev_acc: 0.5, test_loss: 0.24727711258522161, test_acc: 0.5
e: 5, train_loss: 0.24344524328410624, dev_loss: 0.24830259331820173, dev_acc: 0.5, test_loss: 0.24355007110969396, test_acc: 0.5
e: 6, train_loss: 0.24077301913499832, dev_loss: 0.26375621376755815, dev_acc: 0.5, test_loss: 0.25344751883588784, test_acc: 0.49822695035460995
e: 7, train_loss: 0.23903957033902407, dev_loss: 0.2562278732888817, dev_acc: 0.5031055900621118, test_loss: 0.24740752606844227, test_acc: 0.5
e: 8, train_loss: 0.23942259442806244, dev_loss: 0.26161089280377264, dev_acc: 0.5, test_loss: 0.25052141448390397, test_acc: 0.5
e: 9, train_loss: 0.2419098305180669, dev_loss: 0.260164495835208, dev_acc: 0.4968944099378882, test_loss: 0.25660026142809617, test_acc: 0.5
e: 10, train_loss: 0.24213852094113827, dev_loss: 0.24450041312053336, dev_acc: 0.4968944099378882, test_loss: 0.24034939624421986, test_acc: 0.49822695035460995
e: 11, train_loss: 0.22561184633523226, dev_loss: 0.23501742730692307, dev_acc: 0.48757763975155277, test_loss: 0.22562893725773123, test_acc: 0.49645390070921985
e: 12, train_loss: 0.21284956306219102, dev_loss: 0.22321110521877033, dev_acc: 0.5, test_loss: 0.21686694800113956, test_acc: 0.5035460992907801
e: 13, train_loss: 0.20454916135594248, dev_loss: 0.22176535846376272, dev_acc: 0.5124223602484472, test_loss: 0.2175228595799694, test_acc: 0.5
e: 14, train_loss: 0.19984867795929312, dev_loss: 0.22516836792878483, dev_acc: 0.5031055900621118, test_loss: 0.20949787615487972, test_acc: 0.49645390070921985
e: 15, train_loss: 0.19358561837673188, dev_loss: 0.22115084711716781, dev_acc: 0.5062111801242236, test_loss: 0.21093518417724905, test_acc: 0.5
e: 16, train_loss: 0.19294837664812803, dev_loss: 0.2145182831409555, dev_acc: 0.4937888198757764, test_loss: 0.20624980494274314, test_acc: 0.5
e: 17, train_loss: 0.18699566317349672, dev_loss: 0.209839444954573, dev_acc: 0.5, test_loss: 0.20152069735241698, test_acc: 0.49822695035460995
e: 18, train_loss: 0.18424201670475304, dev_loss: 0.21826199715181907, dev_acc: 0.5031055900621118, test_loss: 0.20571685664302913, test_acc: 0.5
e: 19, train_loss: 0.18534493640251457, dev_loss: 0.23126122082427422, dev_acc: 0.4937888198757764, test_loss: 0.2173555254117183, test_acc: 0.5070921985815603
e: 20, train_loss: 0.18304250153712928, dev_loss: 0.22183491278046408, dev_acc: 0.5031055900621118, test_loss: 0.20632043488799257, test_acc: 0.5035460992907801
e: 21, train_loss: 0.1802263075299561, dev_loss: 0.22495511868355436, dev_acc: 0.5062111801242236, test_loss: 0.2107991883333059, test_acc: 0.5106382978723404
e: 22, train_loss: 0.1828378857523203, dev_loss: 0.23553415951290116, dev_acc: 0.5217391304347826, test_loss: 0.22836971621141366, test_acc: 0.5301418439716312
e: 23, train_loss: 0.18168769160285592, dev_loss: 0.22082529270223208, dev_acc: 0.5403726708074534, test_loss: 0.20968690788016675, test_acc: 0.5390070921985816
e: 24, train_loss: 0.17636349546164273, dev_loss: 0.22637364308795202, dev_acc: 0.5341614906832298, test_loss: 0.21071934922604907, test_acc: 0.5354609929078015
e: 25, train_loss: 0.17173552691191435, dev_loss: 0.22172150060950968, dev_acc: 0.562111801242236, test_loss: 0.210233308071046, test_acc: 0.5762411347517731
e: 26, train_loss: 0.16036520379781724, dev_loss: 0.2249327686424396, dev_acc: 0.5590062111801242, test_loss: 0.22014968376606703, test_acc: 0.5904255319148937
e: 27, train_loss: 0.14953845535591245, dev_loss: 0.2613615583171432, dev_acc: 0.5900621118012422, test_loss: 0.23490556177240632, test_acc: 0.650709219858156
e: 28, train_loss: 0.14306637167930603, dev_loss: 0.23273483279988355, dev_acc: 0.6304347826086957, test_loss: 0.21077270539979456, test_acc: 0.6524822695035462
e: 29, train_loss: 0.1125410674600862, dev_loss: 0.328571827951327, dev_acc: 0.5900621118012422, test_loss: 0.27559292715179884, test_acc: 0.6578014184397163
e: 30, train_loss: 0.10681850423221477, dev_loss: 0.3028262540776144, dev_acc: 0.5962732919254659, test_loss: 0.25021417922313915, test_acc: 0.6719858156028369
e: 31, train_loss: 0.11054590334626846, dev_loss: 0.252867237927959, dev_acc: 0.6304347826086957, test_loss: 0.23946964368884657, test_acc: 0.648936170212766
e: 32, train_loss: 0.08892031960829627, dev_loss: 0.29925664372869437, dev_acc: 0.6521739130434783, test_loss: 0.2895990449597511, test_acc: 0.6773049645390071
e: 33, train_loss: 0.06949685538909398, dev_loss: 0.3592695395127069, dev_acc: 0.6180124223602484, test_loss: 0.31509349082249005, test_acc: 0.6843971631205674
e: 34, train_loss: 0.05425452147296164, dev_loss: 0.44414328501759726, dev_acc: 0.5993788819875776, test_loss: 0.3874437218326691, test_acc: 0.6719858156028369
e: 35, train_loss: 0.05348577486150316, dev_loss: 0.3626743588044875, dev_acc: 0.6366459627329193, test_loss: 0.3422626014373675, test_acc: 0.6879432624113475
e: 36, train_loss: 0.03863009900200268, dev_loss: 0.45333604302877517, dev_acc: 0.639751552795031, test_loss: 0.43051605247801616, test_acc: 0.6666666666666666
e: 37, train_loss: 0.027221699051675388, dev_loss: 0.49353985025270686, dev_acc: 0.6211180124223602, test_loss: 0.43117315198973855, test_acc: 0.6826241134751773
e: 38, train_loss: 0.028068545884143533, dev_loss: 0.4010733427002949, dev_acc: 0.6614906832298136, test_loss: 0.3923024876204175, test_acc: 0.6826241134751773
e: 39, train_loss: 0.03148205360756401, dev_loss: 0.48950811287804696, dev_acc: 0.6583850931677019, test_loss: 0.4892647209794535, test_acc: 0.6648936170212766
e: 40, train_loss: 0.033178778049077665, dev_loss: 0.46793152551068734, dev_acc: 0.6180124223602484, test_loss: 0.4272178606728024, test_acc: 0.6932624113475178
e: 41, train_loss: 0.020890956278953127, dev_loss: 0.5059217601316637, dev_acc: 0.6428571428571429, test_loss: 0.5062899369377966, test_acc: 0.6578014184397163
e: 42, train_loss: 0.023722222017695457, dev_loss: 0.5397993214913056, dev_acc: 0.6335403726708074, test_loss: 0.5085991418346624, test_acc: 0.6524822695035462
e: 43, train_loss: 0.024195992723953168, dev_loss: 0.5163615156065959, dev_acc: 0.6583850931677019, test_loss: 0.48955762311122936, test_acc: 0.6861702127659575
e: 44, train_loss: 0.020810427537691793, dev_loss: 0.5384818761874118, dev_acc: 0.6211180124223602, test_loss: 0.46796669256982953, test_acc: 0.6914893617021277
e: 45, train_loss: 0.016506444557853682, dev_loss: 0.6041141917724626, dev_acc: 0.6304347826086957, test_loss: 0.5471890171966377, test_acc: 0.6684397163120568
e: 46, train_loss: 0.021641011154835722, dev_loss: 0.5926208757304046, dev_acc: 0.6304347826086957, test_loss: 0.6096293934562623, test_acc: 0.650709219858156
e: 47, train_loss: 0.01512800203167717, dev_loss: 0.6223050302248413, dev_acc: 0.6428571428571429, test_loss: 0.6038178175230546, test_acc: 0.6790780141843972
e: 48, train_loss: 0.030207861629536183, dev_loss: 0.6473018041357441, dev_acc: 0.6086956521739131, test_loss: 0.6006049144879614, test_acc: 0.6578014184397163
e: 49, train_loss: 0.017044451806736107, dev_loss: 0.5429579400257771, dev_acc: 0.6614906832298136, test_loss: 0.49823011982536936, test_acc: 0.6968085106382979
e: 50, train_loss: 0.016662945735390167, dev_loss: 0.5549065909945509, dev_acc: 0.6490683229813664, test_loss: 0.5736373966931215, test_acc: 0.6578014184397163
e: 51, train_loss: 0.0157839146935612, dev_loss: 0.6183769812894764, dev_acc: 0.6180124223602484, test_loss: 0.5507526130574382, test_acc: 0.6914893617021277
e: 52, train_loss: 0.02144226588933225, dev_loss: 0.4952592177801053, dev_acc: 0.6366459627329193, test_loss: 0.4556949900605182, test_acc: 0.675531914893617
e: 53, train_loss: 0.018533883418516327, dev_loss: 0.531443055163515, dev_acc: 0.6304347826086957, test_loss: 0.5079536081906302, test_acc: 0.650709219858156
e: 54, train_loss: 0.0353637785368328, dev_loss: 0.4136665382803975, dev_acc: 0.6242236024844721, test_loss: 0.40550085548870196, test_acc: 0.6560283687943262
e: 55, train_loss: 0.021074705282226203, dev_loss: 0.6095098832917999, dev_acc: 0.6118012422360248, test_loss: 0.5637653104541022, test_acc: 0.6773049645390071
e: 56, train_loss: 0.016011904758269337, dev_loss: 0.6013486539949898, dev_acc: 0.6211180124223602, test_loss: 0.5369924996864258, test_acc: 0.6666666666666666
e: 57, train_loss: 0.011346234242215359, dev_loss: 0.6479821471845205, dev_acc: 0.5900621118012422, test_loss: 0.5597745597453294, test_acc: 0.675531914893617
e: 58, train_loss: 0.00868771672062121, dev_loss: 0.7091603820820366, dev_acc: 0.5993788819875776, test_loss: 0.5862949638615369, test_acc: 0.700354609929078
e: 59, train_loss: 0.014473035617841332, dev_loss: 0.6922548072567006, dev_acc: 0.6055900621118012, test_loss: 0.585191269524089, test_acc: 0.6648936170212766
e: 60, train_loss: 0.028498264530143386, dev_loss: 0.5187524827719052, dev_acc: 0.6180124223602484, test_loss: 0.48181745904317813, test_acc: 0.6684397163120568
e: 61, train_loss: 0.021582139263802674, dev_loss: 0.46820343948739884, dev_acc: 0.6366459627329193, test_loss: 0.40001836115218836, test_acc: 0.6897163120567376
e: 62, train_loss: 0.01129686711335944, dev_loss: 0.5340334203871879, dev_acc: 0.6428571428571429, test_loss: 0.4612793954282658, test_acc: 0.6879432624113475
e: 63, train_loss: 0.006544655083930592, dev_loss: 0.667686084375364, dev_acc: 0.6366459627329193, test_loss: 0.5584745748630654, test_acc: 0.6897163120567376
e: 64, train_loss: 0.005526855417358092, dev_loss: 0.6616455147509498, dev_acc: 0.6304347826086957, test_loss: 0.5516808285666207, test_acc: 0.6808510638297872
e: 65, train_loss: 0.014227737802326033, dev_loss: 0.5948191799243121, dev_acc: 0.6366459627329193, test_loss: 0.5366669672517506, test_acc: 0.6932624113475178
e: 66, train_loss: 0.011485958283687068, dev_loss: 0.6259072305256296, dev_acc: 0.6242236024844721, test_loss: 0.5342639998355505, test_acc: 0.6879432624113475
e: 67, train_loss: 0.011153878336547678, dev_loss: 0.6132536260039663, dev_acc: 0.6366459627329193, test_loss: 0.5321970533744961, test_acc: 0.6968085106382979
e: 68, train_loss: 0.008989765713407791, dev_loss: 0.6573889996831841, dev_acc: 0.639751552795031, test_loss: 0.5630586833771939, test_acc: 0.6879432624113475
e: 69, train_loss: 0.013240781115480672, dev_loss: 0.5432002843243207, dev_acc: 0.6583850931677019, test_loss: 0.4979835606462628, test_acc: 0.7021276595744681
e: 70, train_loss: 0.010571178503771024, dev_loss: 0.6253985624854366, dev_acc: 0.6335403726708074, test_loss: 0.55940922128128, test_acc: 0.6950354609929078
e: 71, train_loss: 0.009458410911803014, dev_loss: 0.6046842695844202, dev_acc: 0.6521739130434783, test_loss: 0.5592916119160317, test_acc: 0.6861702127659575
e: 72, train_loss: 0.005847095123679566, dev_loss: 0.6476318692153903, dev_acc: 0.6614906832298136, test_loss: 0.5937618392667127, test_acc: 0.6932624113475178
e: 73, train_loss: 0.004082400175175614, dev_loss: 0.6935971840933183, dev_acc: 0.639751552795031, test_loss: 0.6178215080232875, test_acc: 0.6879432624113475
e: 74, train_loss: 0.006359031604341226, dev_loss: 0.6632899078278092, dev_acc: 0.6521739130434783, test_loss: 0.6053662149258984, test_acc: 0.6985815602836879
e: 75, train_loss: 0.006742675515364368, dev_loss: 0.7089376106136078, dev_acc: 0.6677018633540373, test_loss: 0.6281662828622453, test_acc: 0.700354609929078
e: 76, train_loss: 0.006788562178241719, dev_loss: 0.7391238587405426, dev_acc: 0.6645962732919255, test_loss: 0.6477796160298709, test_acc: 0.6897163120567376
e: 77, train_loss: 0.004203547445258771, dev_loss: 0.7513456190923068, dev_acc: 0.6614906832298136, test_loss: 0.6560004689659842, test_acc: 0.7092198581560284
e: 78, train_loss: 0.004915509521832923, dev_loss: 0.7518552554921436, dev_acc: 0.6770186335403726, test_loss: 0.6667185185228383, test_acc: 0.7039007092198581
e: 79, train_loss: 0.005986200365117355, dev_loss: 0.7406577762980598, dev_acc: 0.6677018633540373, test_loss: 0.659327452201855, test_acc: 0.7056737588652482
e: 80, train_loss: 0.006750404694209197, dev_loss: 0.7409512022035525, dev_acc: 0.65527950310559, test_loss: 0.6688112916137634, test_acc: 0.6950354609929078
e: 81, train_loss: 0.008062874438248897, dev_loss: 0.7533964048529952, dev_acc: 0.6521739130434783, test_loss: 0.6731351724714597, test_acc: 0.6985815602836879
e: 82, train_loss: 0.006984639225394104, dev_loss: 0.7179139795403628, dev_acc: 0.6677018633540373, test_loss: 0.6476426055860635, test_acc: 0.700354609929078
e: 83, train_loss: 0.004865461906032962, dev_loss: 0.7811566623756759, dev_acc: 0.6583850931677019, test_loss: 0.6987433992738681, test_acc: 0.7021276595744681
e: 84, train_loss: 0.005273526602353559, dev_loss: 0.7817522050285083, dev_acc: 0.65527950310559, test_loss: 0.6960985551505109, test_acc: 0.6950354609929078
e: 85, train_loss: 0.00373209106636979, dev_loss: 0.7305836508516486, dev_acc: 0.6459627329192547, test_loss: 0.6479005762736016, test_acc: 0.7074468085106383
e: 86, train_loss: 0.0026197095417002884, dev_loss: 0.7989540332534877, dev_acc: 0.6708074534161491, test_loss: 0.7191133262937599, test_acc: 0.6985815602836879
e: 87, train_loss: 0.002367281935556889, dev_loss: 0.8358273264640804, dev_acc: 0.6677018633540373, test_loss: 0.7454809799241641, test_acc: 0.7039007092198581
e: 88, train_loss: 0.00143036883138004, dev_loss: 0.8800636591997886, dev_acc: 0.6459627329192547, test_loss: 0.7797977989957448, test_acc: 0.7021276595744681
e: 89, train_loss: 0.01662938035083576, dev_loss: 0.628023126173391, dev_acc: 0.6211180124223602, test_loss: 0.5605151152143504, test_acc: 0.6666666666666666
e: 90, train_loss: 0.009812716756386763, dev_loss: 0.6872851865733574, dev_acc: 0.65527950310559, test_loss: 0.6649270563609746, test_acc: 0.6808510638297872
e: 91, train_loss: 0.007439611293031206, dev_loss: 0.691200266658484, dev_acc: 0.6428571428571429, test_loss: 0.6583608184730305, test_acc: 0.7021276595744681
e: 92, train_loss: 0.007541906690615178, dev_loss: 0.7546845809590486, dev_acc: 0.6211180124223602, test_loss: 0.6986661347993969, test_acc: 0.6808510638297872
e: 93, train_loss: 0.005743989860492647, dev_loss: 0.7635009804635331, dev_acc: 0.6428571428571429, test_loss: 0.7500537906956917, test_acc: 0.6684397163120568
e: 94, train_loss: 0.01221057550375422, dev_loss: 0.5886023317083773, dev_acc: 0.6583850931677019, test_loss: 0.578491877737895, test_acc: 0.6719858156028369
e: 95, train_loss: 0.0189648571237675, dev_loss: 0.6300296078649882, dev_acc: 0.6242236024844721, test_loss: 0.5450646091143174, test_acc: 0.650709219858156
e: 96, train_loss: 0.017519363938082278, dev_loss: 0.5556564393022629, dev_acc: 0.639751552795031, test_loss: 0.5332096422881043, test_acc: 0.6666666666666666
e: 97, train_loss: 0.005327246807096799, dev_loss: 0.6388703392414093, dev_acc: 0.6583850931677019, test_loss: 0.6410000900578676, test_acc: 0.6826241134751773
e: 98, train_loss: 0.006988809470277488, dev_loss: 0.6579774333631915, dev_acc: 0.6366459627329193, test_loss: 0.6454289838933492, test_acc: 0.6684397163120568
e: 99, train_loss: 0.013806018753452236, dev_loss: 0.6449625218818082, dev_acc: 0.6645962732919255, test_loss: 0.5947318749596945, test_acc: 0.6879432624113475



tohoku-bert-search
../../results/wsc_sbert.bert-base-japanese-whole-word-masking.search.doublet.pn.220204
e: 0, train_loss: 1.5727808130085468, dev_loss: 1.3476937851920632, dev_acc: 0.5031055900621118, test_loss: 1.1218847557902336, test_acc: 0.7358156028368794
e: 1, train_loss: 0.9197148198932409, dev_loss: 1.0593069929930363, dev_acc: 0.5186335403726708, test_loss: 0.9234643528322154, test_acc: 0.6436170212765957
e: 2, train_loss: 0.8068151720575988, dev_loss: 1.0726420405488577, dev_acc: 0.5434782608695652, test_loss: 0.9364775118495648, test_acc: 0.6117021276595744
e: 3, train_loss: 0.764993749845773, dev_loss: 1.0314561583749626, dev_acc: 0.5341614906832298, test_loss: 0.889352810961906, test_acc: 0.6524822695035462
e: 4, train_loss: 0.7021933975592256, dev_loss: 1.0796271493160947, dev_acc: 0.5527950310559007, test_loss: 0.9080010190406312, test_acc: 0.74822695035461
e: 5, train_loss: 0.6558718701992184, dev_loss: 1.0526522849715367, dev_acc: 0.5434782608695652, test_loss: 0.8977502057572797, test_acc: 0.7216312056737588
e: 6, train_loss: 0.6118467406504787, dev_loss: 1.0648451022780645, dev_acc: 0.5745341614906833, test_loss: 0.8857094294356276, test_acc: 0.7056737588652482
e: 7, train_loss: 0.5376653109535109, dev_loss: 1.2079578366373545, dev_acc: 0.5527950310559007, test_loss: 0.9985289609006076, test_acc: 0.7109929078014184
e: 8, train_loss: 0.468746782829694, dev_loss: 1.2837367718641628, dev_acc: 0.5496894409937888, test_loss: 0.9941526380587042, test_acc: 0.7358156028368794
e: 9, train_loss: 0.4516771516756271, dev_loss: 1.329616304770436, dev_acc: 0.531055900621118, test_loss: 1.0382937585455816, test_acc: 0.6932624113475178
e: 10, train_loss: 0.4086228241773788, dev_loss: 1.4811604524881619, dev_acc: 0.5714285714285714, test_loss: 1.4361961094212006, test_acc: 0.6524822695035462
e: 11, train_loss: 0.37339396359884997, dev_loss: 1.4378202623972938, dev_acc: 0.5683229813664596, test_loss: 1.1692014111807607, test_acc: 0.7358156028368794
e: 12, train_loss: 0.2764649082075339, dev_loss: 1.6985748759955115, dev_acc: 0.5652173913043478, test_loss: 1.3290080748303699, test_acc: 0.7322695035460993
e: 13, train_loss: 0.2425517653148272, dev_loss: 2.0786028794328497, dev_acc: 0.5031055900621118, test_loss: 1.5562513608864017, test_acc: 0.7446808510638298
e: 14, train_loss: 0.30817384865610803, dev_loss: 1.4578542657798845, dev_acc: 0.5683229813664596, test_loss: 1.1653554142884857, test_acc: 0.700354609929078
e: 15, train_loss: 0.16298222896212247, dev_loss: 1.7783161573013853, dev_acc: 0.5745341614906833, test_loss: 1.5917946368115878, test_acc: 0.6808510638297872
e: 16, train_loss: 0.1443115597633805, dev_loss: 1.975255959121981, dev_acc: 0.5807453416149069, test_loss: 1.5699543813871624, test_acc: 0.7109929078014184
e: 17, train_loss: 0.11376598815381295, dev_loss: 1.9912561514546898, dev_acc: 0.5527950310559007, test_loss: 1.6147176057525505, test_acc: 0.6861702127659575
e: 18, train_loss: 0.062026433423954586, dev_loss: 2.287093545964828, dev_acc: 0.5559006211180124, test_loss: 1.7661430963942897, test_acc: 0.7163120567375887
e: 19, train_loss: 0.08435895346752659, dev_loss: 2.0851565492850725, dev_acc: 0.5714285714285714, test_loss: 1.604992812140777, test_acc: 0.7127659574468085
e: 20, train_loss: 0.05236291715324751, dev_loss: 2.461467767297819, dev_acc: 0.5527950310559007, test_loss: 1.907380812749209, test_acc: 0.725177304964539
e: 21, train_loss: 0.0635819245532166, dev_loss: 2.455906037323872, dev_acc: 0.5559006211180124, test_loss: 1.9497042679793763, test_acc: 0.7127659574468085
e: 22, train_loss: 0.03928333761133399, dev_loss: 3.181208508377656, dev_acc: 0.546583850931677, test_loss: 2.6472495193708045, test_acc: 0.7446808510638298
e: 23, train_loss: 0.06917656449003061, dev_loss: 2.6695649979968663, dev_acc: 0.5652173913043478, test_loss: 2.18649829269616, test_acc: 0.7358156028368794
e: 24, train_loss: 0.058570775395262446, dev_loss: 2.932172227035393, dev_acc: 0.5496894409937888, test_loss: 2.400867427963385, test_acc: 0.75
e: 25, train_loss: 0.05474953854719433, dev_loss: 2.7719251432526524, dev_acc: 0.5403726708074534, test_loss: 2.1311193434673266, test_acc: 0.7375886524822695
e: 26, train_loss: 0.050757890782818324, dev_loss: 2.6432985098953656, dev_acc: 0.5590062111801242, test_loss: 1.9311369860069343, test_acc: 0.7216312056737588
e: 27, train_loss: 0.03231224990996543, dev_loss: 2.7685454004071888, dev_acc: 0.593167701863354, test_loss: 2.1442861178250663, test_acc: 0.6879432624113475
e: 28, train_loss: 0.03351499537672908, dev_loss: 2.8921499004540148, dev_acc: 0.5807453416149069, test_loss: 2.0821251971951162, test_acc: 0.7074468085106383
e: 29, train_loss: 0.022186928056549617, dev_loss: 3.2987714646792594, dev_acc: 0.5652173913043478, test_loss: 2.4947189087680766, test_acc: 0.725177304964539
e: 30, train_loss: 0.021588669964980226, dev_loss: 3.460951506701688, dev_acc: 0.5652173913043478, test_loss: 2.6448616635963145, test_acc: 0.725177304964539
e: 31, train_loss: 0.011624920680297805, dev_loss: 3.3988897851464213, dev_acc: 0.5714285714285714, test_loss: 2.6254223080659673, test_acc: 0.700354609929078
e: 32, train_loss: 0.016779935528497162, dev_loss: 3.4029270899059023, dev_acc: 0.562111801242236, test_loss: 2.6853581786411596, test_acc: 0.6985815602836879
e: 33, train_loss: 0.017604514465039075, dev_loss: 3.6737348211816796, dev_acc: 0.562111801242236, test_loss: 2.8347023602326864, test_acc: 0.725177304964539
e: 34, train_loss: 0.017315828781051663, dev_loss: 3.2774476568635285, dev_acc: 0.577639751552795, test_loss: 2.7048252441697467, test_acc: 0.6773049645390071
e: 35, train_loss: 0.046036193400887215, dev_loss: 3.02612839558015, dev_acc: 0.5559006211180124, test_loss: 2.4453991241255824, test_acc: 0.6861702127659575
e: 36, train_loss: 0.021466816113683763, dev_loss: 2.7809751340826967, dev_acc: 0.562111801242236, test_loss: 2.235908745151914, test_acc: 0.6790780141843972
e: 37, train_loss: 0.01863886308331166, dev_loss: 3.0619935747872717, dev_acc: 0.5559006211180124, test_loss: 2.439986514019586, test_acc: 0.7021276595744681
e: 38, train_loss: 0.01942345473497835, dev_loss: 3.350781173289311, dev_acc: 0.5652173913043478, test_loss: 2.7690556835139044, test_acc: 0.7198581560283688
e: 39, train_loss: 0.017720320977896224, dev_loss: 3.516227504358633, dev_acc: 0.5559006211180124, test_loss: 2.9133592132404313, test_acc: 0.7216312056737588
e: 40, train_loss: 0.02837516342055096, dev_loss: 3.153574966282484, dev_acc: 0.5714285714285714, test_loss: 2.478090883095217, test_acc: 0.6950354609929078
e: 41, train_loss: 0.012127468565356252, dev_loss: 3.2545101389535582, dev_acc: 0.5652173913043478, test_loss: 2.577751647614329, test_acc: 0.6843971631205674
e: 42, train_loss: 0.00620211158490008, dev_loss: 3.5326990391488287, dev_acc: 0.5590062111801242, test_loss: 2.807480661802471, test_acc: 0.7163120567375887
e: 43, train_loss: 0.015165503961132004, dev_loss: 3.4296441060785186, dev_acc: 0.5652173913043478, test_loss: 2.745019690516903, test_acc: 0.7092198581560284
e: 44, train_loss: 0.006760001322469065, dev_loss: 3.4051855467037706, dev_acc: 0.5745341614906833, test_loss: 2.6506257042250363, test_acc: 0.6897163120567376
e: 45, train_loss: 0.01232774748566203, dev_loss: 3.3216937661841426, dev_acc: 0.562111801242236, test_loss: 2.5593507656280634, test_acc: 0.7056737588652482
e: 46, train_loss: 0.01438438820654136, dev_loss: 3.6000381417245335, dev_acc: 0.5745341614906833, test_loss: 2.7037596850918444, test_acc: 0.723404255319149
e: 47, train_loss: 0.004348332041585536, dev_loss: 3.5951692973829665, dev_acc: 0.546583850931677, test_loss: 2.670168713592257, test_acc: 0.7021276595744681
e: 48, train_loss: 0.008680667903769574, dev_loss: 3.597593340520244, dev_acc: 0.562111801242236, test_loss: 2.730617004889655, test_acc: 0.6985815602836879
e: 49, train_loss: 0.0034073764059024824, dev_loss: 3.9762254837634083, dev_acc: 0.5527950310559007, test_loss: 3.020693934035953, test_acc: 0.7163120567375887
e: 50, train_loss: 0.008445798569162711, dev_loss: 3.7751372455896184, dev_acc: 0.5714285714285714, test_loss: 2.9044570114222186, test_acc: 0.7056737588652482
e: 51, train_loss: 0.01418184392846895, dev_loss: 3.71268111860493, dev_acc: 0.5559006211180124, test_loss: 2.9231806083559686, test_acc: 0.725177304964539
e: 52, train_loss: 0.001443008700806672, dev_loss: 3.656303634326428, dev_acc: 0.5559006211180124, test_loss: 2.7694267790082994, test_acc: 0.7127659574468085
e: 53, train_loss: 0.001525435555958765, dev_loss: 3.877604385647231, dev_acc: 0.5652173913043478, test_loss: 2.973082781063126, test_acc: 0.7163120567375887
e: 54, train_loss: 0.008362638499641804, dev_loss: 3.619688545807072, dev_acc: 0.577639751552795, test_loss: 2.8131173143041717, test_acc: 0.7092198581560284
e: 55, train_loss: 0.019634032815182934, dev_loss: 3.4332341446867636, dev_acc: 0.5714285714285714, test_loss: 2.5869998065098176, test_acc: 0.7092198581560284
e: 56, train_loss: 0.039893797084214075, dev_loss: 3.04791810146857, dev_acc: 0.5590062111801242, test_loss: 2.380549098353911, test_acc: 0.723404255319149
e: 57, train_loss: 0.016545294696547898, dev_loss: 2.920568899989596, dev_acc: 0.562111801242236, test_loss: 2.4384811589049398, test_acc: 0.7304964539007093
e: 58, train_loss: 0.0207376274512593, dev_loss: 2.8771356861914064, dev_acc: 0.5745341614906833, test_loss: 2.213805918444233, test_acc: 0.7039007092198581
e: 59, train_loss: 0.013952193573900559, dev_loss: 3.2332977477345124, dev_acc: 0.5496894409937888, test_loss: 2.518615508961349, test_acc: 0.7092198581560284
e: 60, train_loss: 0.010148104074583785, dev_loss: 3.329562820123652, dev_acc: 0.5652173913043478, test_loss: 2.5542300891439615, test_acc: 0.6950354609929078
e: 61, train_loss: 0.00652313078693254, dev_loss: 3.453515967054177, dev_acc: 0.562111801242236, test_loss: 2.7523622685268307, test_acc: 0.6826241134751773
e: 62, train_loss: 0.01773575766335864, dev_loss: 3.2973280863475383, dev_acc: 0.5527950310559007, test_loss: 2.7106062634278363, test_acc: 0.7287234042553191
e: 63, train_loss: 0.008720573735570782, dev_loss: 3.846510973187634, dev_acc: 0.5434782608695652, test_loss: 3.3422223084049008, test_acc: 0.7464539007092199
e: 64, train_loss: 0.02524255139128252, dev_loss: 2.9074343477147506, dev_acc: 0.546583850931677, test_loss: 2.4961856767323853, test_acc: 0.6879432624113475
e: 65, train_loss: 0.006211025200356687, dev_loss: 3.1977760353296256, dev_acc: 0.5559006211180124, test_loss: 2.6679604299983133, test_acc: 0.7127659574468085
e: 66, train_loss: 0.00781700577445514, dev_loss: 3.5270263091113487, dev_acc: 0.5745341614906833, test_loss: 2.9507162624765164, test_acc: 0.7216312056737588
e: 67, train_loss: 0.001614351751947055, dev_loss: 3.5757406099600946, dev_acc: 0.5683229813664596, test_loss: 3.0366347455364657, test_acc: 0.6985815602836879
e: 68, train_loss: 0.002904372495548465, dev_loss: 3.6624620492404674, dev_acc: 0.593167701863354, test_loss: 3.1641388515626985, test_acc: 0.7163120567375887
e: 69, train_loss: 0.012439937907420066, dev_loss: 3.562373573935672, dev_acc: 0.5714285714285714, test_loss: 3.0203449149880623, test_acc: 0.7074468085106383
e: 70, train_loss: 0.0034115551202860317, dev_loss: 3.429403414240351, dev_acc: 0.562111801242236, test_loss: 2.9373170849244383, test_acc: 0.7145390070921985
e: 71, train_loss: 0.014816288230752804, dev_loss: 3.3225230890492474, dev_acc: 0.5683229813664596, test_loss: 2.9109919768846737, test_acc: 0.723404255319149
e: 72, train_loss: 0.007575188386291131, dev_loss: 3.169871635164204, dev_acc: 0.562111801242236, test_loss: 2.843518274932141, test_acc: 0.725177304964539
e: 73, train_loss: 0.020395916887998737, dev_loss: 3.230998561910866, dev_acc: 0.5590062111801242, test_loss: 2.808605313611508, test_acc: 0.7269503546099291
e: 74, train_loss: 0.010901808420708448, dev_loss: 3.0774982932472885, dev_acc: 0.562111801242236, test_loss: 2.740251171283592, test_acc: 0.6932624113475178
e: 75, train_loss: 0.010110855527298497, dev_loss: 3.842173288673543, dev_acc: 0.562111801242236, test_loss: 3.439514010375777, test_acc: 0.7322695035460993
e: 76, train_loss: 0.005171853393400454, dev_loss: 3.425495425009457, dev_acc: 0.5683229813664596, test_loss: 3.0966277587399396, test_acc: 0.7198581560283688
e: 77, train_loss: 0.004589857470085178, dev_loss: 3.3450836251221086, dev_acc: 0.5838509316770186, test_loss: 2.9461251399919983, test_acc: 0.7039007092198581
e: 78, train_loss: 0.021600761139619173, dev_loss: 2.9765716157994517, dev_acc: 0.5652173913043478, test_loss: 2.6591611113572884, test_acc: 0.700354609929078
e: 79, train_loss: 0.0032642258413912942, dev_loss: 3.2037032086287294, dev_acc: 0.5683229813664596, test_loss: 2.7640029513309328, test_acc: 0.7163120567375887
e: 80, train_loss: 0.014908622409515203, dev_loss: 3.3080188637415042, dev_acc: 0.5590062111801242, test_loss: 2.8631824330324807, test_acc: 0.723404255319149
e: 81, train_loss: 0.042133659251535734, dev_loss: 2.4328544887679504, dev_acc: 0.593167701863354, test_loss: 2.1385962176697366, test_acc: 0.700354609929078
e: 82, train_loss: 0.012147073408835923, dev_loss: 2.733932190876869, dev_acc: 0.5745341614906833, test_loss: 2.33912342178496, test_acc: 0.7163120567375887
e: 83, train_loss: 0.008636428014857131, dev_loss: 2.831240461383308, dev_acc: 0.5807453416149069, test_loss: 2.4641164673712797, test_acc: 0.7163120567375887
e: 84, train_loss: 0.015147013341584057, dev_loss: 2.8467921516652703, dev_acc: 0.5838509316770186, test_loss: 2.469144384766695, test_acc: 0.7145390070921985
e: 85, train_loss: 0.0035549718168148272, dev_loss: 2.9502605862266003, dev_acc: 0.577639751552795, test_loss: 2.576187716865218, test_acc: 0.7021276595744681
e: 86, train_loss: 0.0027027506110048876, dev_loss: 3.471239358728819, dev_acc: 0.562111801242236, test_loss: 2.990035864367443, test_acc: 0.7198581560283688
e: 87, train_loss: 0.0015177250263978977, dev_loss: 3.3273790664005287, dev_acc: 0.5869565217391305, test_loss: 2.955217156675743, test_acc: 0.6985815602836879
e: 88, train_loss: 0.004773861193118044, dev_loss: 3.951225476708097, dev_acc: 0.5527950310559007, test_loss: 3.4620587853955347, test_acc: 0.7358156028368794
e: 89, train_loss: 0.016463411932953798, dev_loss: 3.4946195038020527, dev_acc: 0.5993788819875776, test_loss: 2.9626570683356466, test_acc: 0.7039007092198581
e: 90, train_loss: 0.010570770376358524, dev_loss: 3.7451181962088183, dev_acc: 0.5652173913043478, test_loss: 3.252953269170888, test_acc: 0.725177304964539
e: 91, train_loss: 0.014721987523462871, dev_loss: 3.300163548963367, dev_acc: 0.5683229813664596, test_loss: 2.8350218478674374, test_acc: 0.7216312056737588
e: 92, train_loss: 0.01672022528449014, dev_loss: 2.8624709406989175, dev_acc: 0.6086956521739131, test_loss: 2.558260663810185, test_acc: 0.6950354609929078
e: 93, train_loss: 0.011573124021923831, dev_loss: 3.0009587223512577, dev_acc: 0.5869565217391305, test_loss: 2.6695271022043987, test_acc: 0.7127659574468085
e: 94, train_loss: 0.02561113324265534, dev_loss: 2.778672507773058, dev_acc: 0.5714285714285714, test_loss: 2.428365234513598, test_acc: 0.723404255319149
e: 95, train_loss: 0.022555310220629506, dev_loss: 2.5778334764775606, dev_acc: 0.5807453416149069, test_loss: 2.2131704021890606, test_acc: 0.7180851063829787
e: 96, train_loss: 0.008595853956670908, dev_loss: 2.821354571695113, dev_acc: 0.593167701863354, test_loss: 2.368199115938281, test_acc: 0.7039007092198581
e: 97, train_loss: 0.005721358396924415, dev_loss: 3.0192801244091756, dev_acc: 0.6024844720496895, test_loss: 2.527446052496826, test_acc: 0.7092198581560284
e: 98, train_loss: 0.008730500822702141, dev_loss: 3.2145096282610215, dev_acc: 0.5838509316770186, test_loss: 2.721470992154775, test_acc: 0.723404255319149
e: 99, train_loss: 0.00891362932010719, dev_loss: 3.154910337483147, dev_acc: 0.5807453416149069, test_loss: 2.7135178380847376, test_acc: 0.7304964539007093

t5-base

'''