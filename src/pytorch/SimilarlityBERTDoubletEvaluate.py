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

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.doublet.220823.1', 18
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.doublet.220823.1', 22
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.doublet.220823.1', 100
    elif mode == 't5-base-encoder':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.doublet.220823.1.encoder', 98 # 98: encoder, 89: decoder
    elif mode == 't5-base-decoder':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.doublet.220823.1.decoder', 89  # 98: encoder, 89: decoder
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.doublet.220823.1', 65
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.doublet.220823.1', 41
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-large-japanese.doublet.220823.1', 91
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.doublet.220823.1', 56
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', '../../results/xlm-roberta-large.doublet.220823.1', 89
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', '../../results/xlm-roberta-base.doublet.220823.1', 86


def get_datasets(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    texts = [text.strip().split('\t') for text in texts]
    s1, s2, l = [], [], []
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
    return s1, s2, l

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    weight_path = Path(OUTPUT_PATH + f'/{model_name.replace("/", ".")}.{NUM_EPOCHS}.pt')
    with weight_path.open('rb') as f:
        weights = torch.load(f)
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
    model.load_state_dict(weights['model'], strict=True)
    model = allocate_data_to_device(model, DEVICE)
    output_layer = torch.nn.Linear(model.config.hidden_size, 1)
    output_layer.load_state_dict(weights['output_layer'], strict=True)
    output_layer = allocate_data_to_device(output_layer, DEVICE)

    # train_sentence1, train_sentence2, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    # dev_sentence1, dev_sentence2, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence1, test_sentence2, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    # output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    # optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=2e-5, weight_decay=0.01)
    # loss_func = torch.nn.CrossEntropyLoss()
    # vw = ValueWatcher()

    model.eval()
    output_layer.eval()
    # TEST
    lines = []
    test_tt, test_ff = 0, 0
    for i, (s1, s2, l) in enumerate(zip(test_sentence1, test_sentence2, test_labels)):
        inputs = tokenizer(s1, return_tensors='pt')
        inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
        outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
        # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
        h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
        h1 = torch.mean(h1, dim=1)
        o1 = output_layer(h1)
        o1 = activation_function(o1) if with_activation_function else o1

        inputs = tokenizer(s2, return_tensors='pt')
        inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
        outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
        # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
        h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
        h2 = torch.mean(h2, dim=1)
        o2 = output_layer(h2)
        o2 = activation_function(o2) if with_activation_function else o2

        # loss = loss_func(torch.cat([o1, o2], dim=1), allocate_data_to_device(torch.LongTensor([l]), DEVICE))
        # test_total_loss.append(loss.item())

        if o1 > o2:
            predict_label = 0
        else:
            predict_label = 1

        if predict_label == l:
            test_tt += 1
        else:
            test_ff += 1

        lines.append([i, s1, s2, l, predict_label, h1[0].tolist(), h2[0].tolist(), float(o1), float(o2)])

        # if with_print_logits:
        #     print(f'{o1.item()}, {o2.item()}, {l}, {predict_label}')

    # test_total_loss = sum(test_total_loss) / len(test_total_loss)
    test_acc = test_tt / (test_tt + test_ff)

    # print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, dev_acc: {dev_acc}, test_loss: {test_total_loss}, test_acc: {test_acc}')
    info = [test_acc, model_name, str(weight_path)]

    with Path(f'{OUTPUT_PATH}/details.results.doublet.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['index', 'sentence1', 'sentence2', 'label', 'predicted_label', 'score1', 'score2']))
        f.write('\n')
        for line in lines:
            f.write(','.join(map(str, [item for item in line if type(item) != list])))
            f.write('\n')

    with Path(f'{OUTPUT_PATH}/details.results.doublet.{model_name.replace("/", ".")}.raw.pt').open('wb') as f:
        torch.save({'results': lines, 'info': info}, f)


if __name__ == '__main__':
    is_single = False
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    if is_single:
        train_model(run_modes[1])
    else:
        for run_mode in run_modes:
            train_model(run_mode)


'''
'''
