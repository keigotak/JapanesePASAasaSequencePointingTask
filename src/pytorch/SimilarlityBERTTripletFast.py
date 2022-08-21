import os
import random
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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

class JWSCDataset(torch.utils.data.Dataset):
    def __init__(self, s_a, s_p, s_n, y):
        super().__init__()
        self.sentence_anchor = s_a
        self.sentence_positive = s_p
        self.sentence_negative = s_n
        self.label = y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.sentence_anchor[index], self.sentence_positive[index], self.sentence_negative[index], self.label[index]



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

def pooling_mean(x, mask):
    # mask のエッジのインデックス取得
    mask = torch.argmin(mask, dim=1, keepdim=True)
    # mask をソート
    mask, idx = torch.sort(mask, dim=0)
    # mask で並べ直した順に x をソート
    x = torch.gather(x, dim=0, index=idx.unsqueeze(2).expand(-1, x.shape[1], x.shape[2]))
    # mask の長さでまとまったので，同じ長さの mask をまとめて処理
    unique_items, counts = torch.unique_consecutive(mask, return_counts=True)
    unique_items = unique_items.tolist()
    counts = [0] + torch.cumsum(counts, -1).tolist()
    x = torch.cat([torch.mean(x[counts[i]: counts[i+1], :ui, :], dim=1) if ui != 0 else torch.mean(x[counts[i]: counts[i+1], :, :], dim=1) for i, ui in enumerate(unique_items)])
    # 元に戻す
    idx = torch.argsort(idx, dim=0)
    x = torch.gather(x, dim=0, index=idx.expand(-1, x.shape[1]))
    return x

def pooling_max(x, mask):
    # mask のエッジのインデックス取得
    mask = torch.argmin(mask, dim=1, keepdim=True)
    # mask をソート
    mask, idx = torch.sort(mask, dim=0)
    # mask で並べ直した順に x をソート
    x = torch.gather(x, dim=0, index=idx.unsqueeze(2).expand(-1, x.shape[1], x.shape[2]))
    # mask の長さでまとまったので，同じ長さの mask をまとめて処理
    unique_items, counts = torch.unique_consecutive(mask, return_counts=True)
    unique_items = unique_items.tolist()
    counts = [0] + torch.cumsum(counts, -1).tolist()
    x = torch.cat([torch.max(x[counts[i]: counts[i+1], :ui, :], dim=1)[0] if ui != 0 else torch.max(x[counts[i]: counts[i+1], :, :], dim=1)[0] for i, ui in enumerate(unique_items)])
    # 元に戻す
    idx = torch.argsort(idx, dim=0)
    x = torch.gather(x, dim=0, index=idx.expand(-1, x.shape[1]))
    return x


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
    output_layer = allocate_data_to_device(torch.nn.Linear(embedding_dim, 1), DEVICE)
    activation_function = torch.nn.SELU()
    dropout_layer = torch.nn.Dropout(0.2)

    alpha = 1.0
    parameters = list(model.parameters()) + list(output_layer.parameters())
    optimizer = AdamW(params=parameters, lr=2e-5, weight_decay=0.01)
    loss_func1 = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    loss_func2 = torch.nn.CrossEntropyLoss()
    loss_func3 = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    # loss_func3 = losses.TripletMarginLoss(distance = CosineSimilarity(),
    # 			     reducer = ThresholdReducer(high=0.3),
    # 		 	     embedding_regularizer = LpRegularizer(p=2))

    # loss_func = torch.nn.CosineSimilarity()
    distance_metric = TripletDistanceMetric.EUCLIDEAN
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=WARMUP_STEPS)
    vw = ValueWatcher()

    train_sentence0, train_sentence1, train_sentence2, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    dev_sentence0, dev_sentence1, dev_sentence2, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence0, test_sentence1, test_sentence2, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    train_tokens0, train_tokens1, train_tokens2 = [tokenizer(s, return_tensors='pt') for s in train_sentence0], [tokenizer(s, return_tensors='pt') for s in train_sentence1], [tokenizer(s, return_tensors='pt') for s in train_sentence2]
    dev_tokens0, dev_tokens1, dev_tokens2 = [tokenizer(s, return_tensors='pt') for s in dev_sentence0], [tokenizer(s, return_tensors='pt') for s in dev_sentence1], [tokenizer(s, return_tensors='pt') for s in dev_sentence2]
    test_tokens0, test_tokens1, test_tokens2 = [tokenizer(s, return_tensors='pt') for s in test_sentence0], [tokenizer(s, return_tensors='pt') for s in test_sentence1], [tokenizer(s, return_tensors='pt') for s in test_sentence2]

    train_dataset = JWSCDataset(train_tokens0, train_tokens1, train_tokens2, train_labels)
    dev_dataset = JWSCDataset(dev_tokens0, dev_tokens1, dev_tokens2, dev_labels)
    test_dataset = JWSCDataset(test_tokens0, test_tokens1, test_tokens2, test_labels)

    def collate_fn(examples):
        max_sentence_length = 512
        padding_id = 0
        s0 = [torch.as_tensor(example[0].input_ids, dtype=torch.long).squeeze(0) for example in examples]
        s0 = torch.nn.utils.rnn.pad_sequence(s0, batch_first=True, padding_value=padding_id)
        attention_mask0 = [torch.as_tensor(example[0].attention_mask, dtype=torch.long).squeeze(0) for example in examples]
        attention_mask0 = torch.nn.utils.rnn.pad_sequence(attention_mask0, batch_first=True, padding_value=0)

        s1 = [torch.as_tensor(example[1].input_ids, dtype=torch.long).squeeze(0) for example in examples]
        s1 = torch.nn.utils.rnn.pad_sequence(s1, batch_first=True, padding_value=padding_id)
        attention_mask1 = [torch.as_tensor(example[1].attention_mask, dtype=torch.long).squeeze(0) for example in examples]
        attention_mask1 = torch.nn.utils.rnn.pad_sequence(attention_mask1, batch_first=True, padding_value=0)

        s2 = [torch.as_tensor(example[2].input_ids, dtype=torch.long).squeeze(0) for example in examples]
        s2 = torch.nn.utils.rnn.pad_sequence(s2, batch_first=True, padding_value=padding_id)
        attention_mask2 = [torch.as_tensor(example[2].attention_mask, dtype=torch.long).squeeze(0) for example in examples]
        attention_mask2 = torch.nn.utils.rnn.pad_sequence(attention_mask2, batch_first=True, padding_value=0)

        y = torch.as_tensor([example[3] for example in examples], dtype=torch.long)

        if max_sentence_length != -1:
            if s0.size(1) > max_sentence_length:
                s0 = s0[:, :max_sentence_length]
                attention_mask0 = attention_mask0[:, :max_sentence_length]
            if s1.size(1) > max_sentence_length:
                s1 = s1[:, :max_sentence_length]
                attention_mask1 = attention_mask1[:, :max_sentence_length]
            if s2.size(1) > max_sentence_length:
                s2 = s2[:, :max_sentence_length]
                attention_mask2 = attention_mask2[:, :max_sentence_length]

        x = {'sentence0': s0, 'sentence1': s1, 'sentence2': s2}
        attention_masks = {'sentence0': attention_mask0, 'sentence1': attention_mask1, 'sentence2': attention_mask2}

        return x, y, attention_masks, examples

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False
    )

    result_lines = []

    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for x, y, attention_masks, allx in train_dataloader:
            inputs = {'input_ids': x['sentence0'], 'attention_mask': attention_masks['sentence0']} # anchor
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h0 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h0 = pooling_mean(h0, attention_masks['sentence0'])
            o0 = output_layer(dropout_layer(h0)) if with_dropout else output_layer(h0)
            o0 = activation_function(o0) if with_activation_function else o0

            inputs = {'input_ids': x['sentence2'], 'attention_mask': attention_masks['sentence2']} # negative
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = pooling_mean(h1, attention_masks['sentence2'])
            o1 = output_layer(dropout_layer(h1)) if with_dropout else output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = {'input_ids': x['sentence1'], 'attention_mask': attention_masks['sentence1']} # positive
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = pooling_mean(h2, attention_masks['sentence1'])
            o2 = output_layer(dropout_layer(h2)) if with_dropout else output_layer(h2)
            o2 = activation_function(o2) if with_activation_function else o2

            loss1 = loss_func1(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func1(anchor=h0, positive=h2, negative=h1)
            # loss2 = loss_func2(torch.cat([o1, o2], dim=1), torch.as_tensor([l], dtype=torch.long, device=DEVICE))
            # loss3 = loss_func3(anchor=h0, positive=h1, negative=h2) if l == 0 else loss_func3(anchor=h0, positive=h2, negative=h1)
            # loss = alpha.weight * loss1 + beta.weight * loss2
            loss = loss1
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
        train_model(run_modes[1])
    else:
        for run_mode in run_modes:
            train_model(run_mode)