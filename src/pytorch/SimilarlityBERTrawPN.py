import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW

random.seed(0)
torch.manual_seed(0)

from pathlib import Path
import ipadic
import MeCab

class PointerNetworks(nn.Module):
    def __init__(self, hyper_parameters={}):
        super(PointerNetworks, self).__init__()
        self.embedding_dim = hyper_parameters['embedding_dim']
        self.hidden_size = self.embedding_dim
        self.num_layers = hyper_parameters['num_layers']
        self.with_use_rnn_repeatedly = hyper_parameters['with_use_rnn_repeatedly']
        self.vocab_size = hyper_parameters['vocab_size']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

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

    def forward(self, batch_ids):
        x = self.embedding(batch_ids)
        x = x.transpose(0, 1)

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
    if mode == 'raw':
        return '', '../../results/wsc_sbert.raw.search', 100

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

def get_label(words, labels):
    current = 0
    index = labels.index(1)
    for i, w in enumerate(words):
        if current <= index < current + len(w):
            return i
        current += len(w)
    return 0

def get_aggregated_label(tokens, labels):
    aggregated_labels = []
    offsets, current = [], 0
    for t in tokens:
        offsets.append((current, current + len(t)))
        current += len(t)

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

class Indexer:
    def __init__(self):
        self.unknown_token = '<unk>'
        self.padding_token = '<pad>'
        self.special_tokens = [self.unknown_token, self.padding_token]
        self.i2w = {i: t for i, t in enumerate(self.special_tokens)}
        self.w2i = {t: i for i, t in enumerate(self.special_tokens)}
        self.current = len(self.special_tokens)

    def __len__(self):
        return self.current

    def fit(self, words):
        for word in words:
            if word not in self.w2i.keys():
                self.w2i[word] = self.current
                self.i2w[self.current] = word
                self.current += 1

    def encode(self, words):
        return [self.w2i[word] if word in self.w2i.keys() else self.w2i['<unk>'] for word in words]

    def batch_encode(self, batch_words, with_padding=True):
        max_length = max(map(len, batch_words))
        if with_padding:
            return [self.encode(words) + [self.w2i[self.padding_token]] * (max_length - len(self.encode(words))) for words in batch_words]
        else:
            return [self.encode(words) for words in batch_words]

    def __call__(self, words):
        return self.encode(words)

    def convert_id_to_tokens(self, indexes):
        return [self.i2w[index] for index in indexes]


class RawEmbeddingGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=3):
        super(RawEmbeddingGRU, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.f_grus = nn.ModuleList([nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim) for _ in range(num_layers)])
        self.b_grus = nn.ModuleList([nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, batch_ids):
        inputs = self.embedding(batch_ids)

        for i in range(self.num_layers):
            f_hidden, _ = self.f_grus[i](inputs)
            f_hidden = inputs + f_hidden
            f_hidden = self.dropout(torch.flip(f_hidden, [0, 1]))

            b_hiddens, _ = self.b_grus[i](f_hidden)
            inputs = b_hiddens + f_hidden
            inputs = self.dropout(torch.flip(inputs, [0, 1]))

        inputs = self.linear(inputs).squeeze(2)
        return inputs


def train_model():
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0' # 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    run_mode = 'raw'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.rawpn.220823.1'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    train_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt')
    test_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-token.txt')
    train_sentences, train_labels = train_datasets['sentences'], train_datasets['positive_labels']
    train_sentences, train_labels, dev_sentences, dev_labels = train_sentences[:1000], train_labels[:1000], train_sentences[1000:], train_labels[1000:]
    test_sentences, test_labels = test_datasets['sentences'], test_datasets['positive_labels']
    dev_negative_labels = train_datasets['negative_labels'][1000:]
    test_negative_labels = test_datasets['negative_labels']

    vocab = set([])
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
    for sentence in train_datasets['sentences']:
        words = tagger.parse(sentence)
        vocab = vocab | set([w.split('\t')[0] for w in words.split('\n')][:-2])

    tokenizer = Indexer()
    tokenizer.fit(vocab)
    # model = RawEmbeddingGRU(vocab_size=len(tokenizer), embedding_dim=300)
    # model.to(DEVICE)
    hyper_parameters = {
        'embedding_dim': 768,
        'vocab_size': len(tokenizer),
        'with_use_rnn_repeatedly': False,
        'num_layers': 3,
        'num_heads': 1
    }
    model = allocate_data_to_device(PointerNetworks(hyper_parameters=hyper_parameters), DEVICE)

    optimizer = AdamW(params=model.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        for s, l in zip(train_sentences, train_labels):
            words = [w.split('\t')[0] for w in tagger.parse(s).split('\n')][:-2]
            inputs = tokenizer(words)
            # index = get_label(words, l)
            index = get_aggregated_label(words, l)

            o1 = model(torch.as_tensor([inputs], dtype=torch.long, device=DEVICE))['logits']

            loss = loss_func(o1.squeeze(0), torch.as_tensor(index, dtype=torch.float, device=DEVICE))
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
        dev_total_loss = []
        dev_tt, dev_ff = 0, 0
        for s, l, neg_l in zip(dev_sentences, dev_labels, dev_negative_labels):
            words = [w.split('\t')[0] for w in tagger.parse(s).split('\n')][:-2]
            inputs = tokenizer(words)
            # index = get_label(words, l)
            # neg_index = get_label(words, neg_l)
            pos_index = get_aggregated_label(words, l)
            neg_index = get_aggregated_label(words, neg_l)

            o1 = model(torch.as_tensor([inputs], dtype=torch.long, device=DEVICE))['logits']

            loss = loss_func(o1.squeeze(0), torch.as_tensor(pos_index, dtype=torch.float, device=DEVICE))

            dev_total_loss.append(loss.item())

            if torch.max(o1[0][torch.as_tensor(pos_index, dtype=torch.int) == 1]).item() > torch.max(o1[0][torch.as_tensor(neg_index, dtype=torch.int) == 1]).item():
                dev_tt += 1
            else:
                dev_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {l}, {torch.argmax(o1[0])}, {pos_index}')

        dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
        dev_acc = dev_tt / (dev_tt + dev_ff)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s, l, neg_l in zip(test_sentences, test_labels, test_negative_labels):
            words = [w.split('\t')[0] for w in tagger.parse(s).split('\n')][:-2]
            inputs = tokenizer(words)
            # index = get_label(words, l)
            # neg_index = get_label(words, neg_l)
            pos_index = get_aggregated_label(words, l)
            neg_index = get_aggregated_label(words, neg_l)

            o1 = model(torch.as_tensor([inputs], dtype=torch.long, device=DEVICE))['logits']

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


'''