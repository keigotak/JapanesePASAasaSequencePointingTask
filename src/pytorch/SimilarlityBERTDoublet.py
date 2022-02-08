import os
import random

import torch.nn

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
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.cross_encoder import CrossEncoder

from transformers import GPT2Tokenizer, GPT2Model
from transformers import T5Tokenizer, AutoModelForCausalLM, T5Model
from transformers import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertJapaneseTokenizer, BertModel

def get_properties(mode):
    if mode == 'rinna-search':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.search.doublet', 100
    elif mode == 'rinna-best':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.best.doublet', 100
    elif mode == 'tohoku-bert-search':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.search.doublet', 100
    elif mode == 'tohoku-bert-best':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.best.doublet', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.search.doublet', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web', 100

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

def train_model():
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0' # 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    run_mode = 't5-base'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.220204'
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
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = allocate_data_to_device(model, DEVICE)

    train_sentence1, train_sentence2, train_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-triplet.txt')
    dev_sentence1, dev_sentence2, dev_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/dev-triplet.txt')
    test_sentence1, test_sentence2, test_labels = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')

    output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=7e-5, weight_decay=0.01)
    # optimizer = AdamW(params=list(output_layer.parameters()), lr=1e-4, weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for s1, s2, l in zip(train_sentence1, train_sentence2, train_labels):
            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
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
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
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

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s1, s2, l in zip(test_sentence1, test_sentence2, test_labels):
            inputs = tokenizer(s1, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            o1 = output_layer(h1)
            o1 = activation_function(o1) if with_activation_function else o1

            inputs = tokenizer(s2, return_tensors='pt')
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys()}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
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

    with Path(f'./result.doublet.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')

if __name__ == '__main__':
    train_model()


'''
rinna-search
e: 0, train_loss: 0.6922497947216034, dev_loss: 0.6925485961555694, dev_acc: 0.5217391304347826, test_loss: 0.6922162011371437, test_acc: 0.5230496453900709
e: 1, train_loss: 0.693102871477604, dev_loss: 0.6920896607526341, dev_acc: 0.5248447204968945, test_loss: 0.6916689348559008, test_acc: 0.526595744680851
e: 2, train_loss: 0.6921369276642799, dev_loss: 0.6915139326397677, dev_acc: 0.5341614906832298, test_loss: 0.6910844484330915, test_acc: 0.5354609929078015
e: 3, train_loss: 0.691099071085453, dev_loss: 0.69085136851909, dev_acc: 0.5496894409937888, test_loss: 0.6904926864390678, test_acc: 0.5531914893617021
e: 4, train_loss: 0.6909227387905121, dev_loss: 0.6901404535548287, dev_acc: 0.562111801242236, test_loss: 0.6897573855751795, test_acc: 0.549645390070922
e: 5, train_loss: 0.6901818223595619, dev_loss: 0.6894204581376189, dev_acc: 0.562111801242236, test_loss: 0.688970186714585, test_acc: 0.5425531914893617
e: 6, train_loss: 0.6898062525391578, dev_loss: 0.6888962273271928, dev_acc: 0.5559006211180124, test_loss: 0.688391517451469, test_acc: 0.5460992907801419
e: 7, train_loss: 0.68795107036829, dev_loss: 0.688446912891376, dev_acc: 0.5496894409937888, test_loss: 0.6879610984672045, test_acc: 0.5514184397163121
e: 8, train_loss: 0.68667988550663, dev_loss: 0.6878355887735853, dev_acc: 0.5652173913043478, test_loss: 0.6874390552018551, test_acc: 0.5602836879432624
e: 9, train_loss: 0.6874637801647187, dev_loss: 0.687139915568488, dev_acc: 0.5652173913043478, test_loss: 0.6867773388083099, test_acc: 0.5638297872340425
e: 10, train_loss: 0.6861494804024696, dev_loss: 0.6867490892084489, dev_acc: 0.5683229813664596, test_loss: 0.6862029349761651, test_acc: 0.5638297872340425
e: 11, train_loss: 0.6851973412036896, dev_loss: 0.6863473579750298, dev_acc: 0.5683229813664596, test_loss: 0.6856856164357341, test_acc: 0.5780141843971631
e: 12, train_loss: 0.6835910316705703, dev_loss: 0.6858147175415702, dev_acc: 0.5745341614906833, test_loss: 0.6851613048120593, test_acc: 0.5797872340425532
e: 13, train_loss: 0.6850525026917458, dev_loss: 0.6852792009063389, dev_acc: 0.5745341614906833, test_loss: 0.6846388094391384, test_acc: 0.5797872340425532
e: 14, train_loss: 0.6840133333802223, dev_loss: 0.6849473820339819, dev_acc: 0.5683229813664596, test_loss: 0.6842162705270957, test_acc: 0.5762411347517731
e: 15, train_loss: 0.6841371202468872, dev_loss: 0.6845975081372705, dev_acc: 0.5807453416149069, test_loss: 0.6837815907090268, test_acc: 0.5673758865248227
e: 16, train_loss: 0.6832972838878631, dev_loss: 0.684111596819777, dev_acc: 0.5900621118012422, test_loss: 0.6833367673217827, test_acc: 0.574468085106383
e: 17, train_loss: 0.6835240550041198, dev_loss: 0.683579161855745, dev_acc: 0.593167701863354, test_loss: 0.6828786470260181, test_acc: 0.5726950354609929
e: 18, train_loss: 0.6842514065504074, dev_loss: 0.6830882019137744, dev_acc: 0.5838509316770186, test_loss: 0.6824076398678706, test_acc: 0.5780141843971631
e: 19, train_loss: 0.681564647257328, dev_loss: 0.6824865757678606, dev_acc: 0.5838509316770186, test_loss: 0.6819104823871707, test_acc: 0.5815602836879432
e: 20, train_loss: 0.6824130092561245, dev_loss: 0.6819066995789546, dev_acc: 0.5869565217391305, test_loss: 0.6814612851920703, test_acc: 0.5762411347517731
e: 21, train_loss: 0.6821332125663757, dev_loss: 0.6814168673124372, dev_acc: 0.6055900621118012, test_loss: 0.6810687724897202, test_acc: 0.5762411347517731
e: 22, train_loss: 0.6806392118930816, dev_loss: 0.6810913217363891, dev_acc: 0.5962732919254659, test_loss: 0.6806363289554914, test_acc: 0.5726950354609929
e: 23, train_loss: 0.6803536997139454, dev_loss: 0.6806165909174806, dev_acc: 0.593167701863354, test_loss: 0.6801339782087515, test_acc: 0.5780141843971631
e: 24, train_loss: 0.6811757555902004, dev_loss: 0.6801684026762566, dev_acc: 0.593167701863354, test_loss: 0.6796094335562793, test_acc: 0.5797872340425532
e: 25, train_loss: 0.6795935217142105, dev_loss: 0.6797136708816386, dev_acc: 0.5962732919254659, test_loss: 0.6791276482614219, test_acc: 0.5780141843971631
e: 26, train_loss: 0.6779721370339393, dev_loss: 0.6792182899224832, dev_acc: 0.6055900621118012, test_loss: 0.6786476670638889, test_acc: 0.5851063829787234
e: 27, train_loss: 0.6785684305131435, dev_loss: 0.6788368947942804, dev_acc: 0.6118012422360248, test_loss: 0.6782152299762617, test_acc: 0.5939716312056738
e: 28, train_loss: 0.6829168789386749, dev_loss: 0.6785188850408755, dev_acc: 0.6086956521739131, test_loss: 0.6779302974753346, test_acc: 0.5939716312056738
e: 29, train_loss: 0.6772406424582005, dev_loss: 0.6780635236953356, dev_acc: 0.6055900621118012, test_loss: 0.6776055158875512, test_acc: 0.5886524822695035
e: 30, train_loss: 0.677919608771801, dev_loss: 0.6776011351102628, dev_acc: 0.6055900621118012, test_loss: 0.6771790018529756, test_acc: 0.5975177304964538
e: 31, train_loss: 0.6780877492129803, dev_loss: 0.6773832359854479, dev_acc: 0.6024844720496895, test_loss: 0.6768923957191461, test_acc: 0.599290780141844
e: 32, train_loss: 0.6751723112165928, dev_loss: 0.6770009799218326, dev_acc: 0.6024844720496895, test_loss: 0.6764650907605252, test_acc: 0.601063829787234
e: 33, train_loss: 0.6755139940977096, dev_loss: 0.6766858936651893, dev_acc: 0.6024844720496895, test_loss: 0.6761241684145961, test_acc: 0.6046099290780141
e: 34, train_loss: 0.6750402366220951, dev_loss: 0.6763146509480032, dev_acc: 0.6024844720496895, test_loss: 0.6757231376906658, test_acc: 0.599290780141844
e: 35, train_loss: 0.6754631456136704, dev_loss: 0.6758199291569846, dev_acc: 0.6180124223602484, test_loss: 0.6752933517415473, test_acc: 0.601063829787234
e: 36, train_loss: 0.6743210592865944, dev_loss: 0.6754260858208496, dev_acc: 0.6273291925465838, test_loss: 0.6749055489686364, test_acc: 0.6028368794326241
e: 37, train_loss: 0.6750210883617401, dev_loss: 0.6749855620520455, dev_acc: 0.6242236024844721, test_loss: 0.6744781414456401, test_acc: 0.6046099290780141
e: 38, train_loss: 0.6731438107788563, dev_loss: 0.6746245344973499, dev_acc: 0.6273291925465838, test_loss: 0.6741574933976991, test_acc: 0.6046099290780141
e: 39, train_loss: 0.6726097064316273, dev_loss: 0.6740809756776561, dev_acc: 0.6304347826086957, test_loss: 0.673777033515433, test_acc: 0.6046099290780141
e: 40, train_loss: 0.672502072095871, dev_loss: 0.6737122140500856, dev_acc: 0.6273291925465838, test_loss: 0.6733509112440103, test_acc: 0.6046099290780141
e: 41, train_loss: 0.6710624910891057, dev_loss: 0.6733976597926632, dev_acc: 0.6211180124223602, test_loss: 0.6730098761882343, test_acc: 0.6152482269503546
e: 42, train_loss: 0.6733034859597683, dev_loss: 0.6730617633702592, dev_acc: 0.6180124223602484, test_loss: 0.6726152654869336, test_acc: 0.6134751773049646
e: 43, train_loss: 0.6763373256921769, dev_loss: 0.6727814096841753, dev_acc: 0.6086956521739131, test_loss: 0.672373077865188, test_acc: 0.6117021276595744
e: 44, train_loss: 0.6725185040533542, dev_loss: 0.6724983605538836, dev_acc: 0.6118012422360248, test_loss: 0.6720461559263949, test_acc: 0.6099290780141844
e: 45, train_loss: 0.6733724861443042, dev_loss: 0.6720842345160727, dev_acc: 0.6180124223602484, test_loss: 0.6716631013926462, test_acc: 0.6134751773049646
e: 46, train_loss: 0.6741190395951271, dev_loss: 0.6717163843582876, dev_acc: 0.6086956521739131, test_loss: 0.6713914155748719, test_acc: 0.6134751773049646
e: 47, train_loss: 0.6746571590751409, dev_loss: 0.6714568610146919, dev_acc: 0.6086956521739131, test_loss: 0.6711889236689882, test_acc: 0.6046099290780141
e: 48, train_loss: 0.6710311845541, dev_loss: 0.67114192497286, dev_acc: 0.6086956521739131, test_loss: 0.6709374548167202, test_acc: 0.6063829787234043
e: 49, train_loss: 0.6734554137587547, dev_loss: 0.6709233696416298, dev_acc: 0.6086956521739131, test_loss: 0.6706807312443324, test_acc: 0.6081560283687943
e: 50, train_loss: 0.6669345887899398, dev_loss: 0.6704441452433604, dev_acc: 0.6180124223602484, test_loss: 0.6702523335867318, test_acc: 0.6063829787234043
e: 51, train_loss: 0.6717435279637575, dev_loss: 0.6700027411028465, dev_acc: 0.6055900621118012, test_loss: 0.6699029403798124, test_acc: 0.6046099290780141
e: 52, train_loss: 0.6652062000632286, dev_loss: 0.6696135148128367, dev_acc: 0.5993788819875776, test_loss: 0.6695723570354865, test_acc: 0.6063829787234043
e: 53, train_loss: 0.6659172269105911, dev_loss: 0.6691716292074749, dev_acc: 0.6024844720496895, test_loss: 0.6692364544786037, test_acc: 0.6099290780141844
e: 54, train_loss: 0.6662008186876773, dev_loss: 0.6688200647231215, dev_acc: 0.6024844720496895, test_loss: 0.6689170024602126, test_acc: 0.6117021276595744
e: 55, train_loss: 0.66929245467484, dev_loss: 0.6685514480615995, dev_acc: 0.5993788819875776, test_loss: 0.6686122920496244, test_acc: 0.6170212765957447
e: 56, train_loss: 0.6735128472447396, dev_loss: 0.6683323278745509, dev_acc: 0.5993788819875776, test_loss: 0.6683076928957556, test_acc: 0.6117021276595744
e: 57, train_loss: 0.6707975660860539, dev_loss: 0.6680995366588143, dev_acc: 0.6024844720496895, test_loss: 0.6680957221868613, test_acc: 0.6152482269503546
e: 58, train_loss: 0.6676925639212131, dev_loss: 0.6678190938434245, dev_acc: 0.6024844720496895, test_loss: 0.6677168217216823, test_acc: 0.6134751773049646
e: 59, train_loss: 0.6766608771830798, dev_loss: 0.6676378698082444, dev_acc: 0.6086956521739131, test_loss: 0.6674555255800274, test_acc: 0.6099290780141844
e: 60, train_loss: 0.6670354898124933, dev_loss: 0.6673414753460736, dev_acc: 0.6055900621118012, test_loss: 0.6671890290916389, test_acc: 0.6117021276595744
e: 61, train_loss: 0.6675041280686855, dev_loss: 0.6669594071666647, dev_acc: 0.6118012422360248, test_loss: 0.6669009554312162, test_acc: 0.6081560283687943
e: 62, train_loss: 0.6639994479715824, dev_loss: 0.6666180312818621, dev_acc: 0.6149068322981367, test_loss: 0.6666000450545169, test_acc: 0.6063829787234043
e: 63, train_loss: 0.6682025741040707, dev_loss: 0.6663992969515902, dev_acc: 0.6086956521739131, test_loss: 0.6663386687530694, test_acc: 0.6081560283687943
e: 64, train_loss: 0.6709533857256174, dev_loss: 0.6663886074508939, dev_acc: 0.6118012422360248, test_loss: 0.6662694707988425, test_acc: 0.6063829787234043
e: 65, train_loss: 0.6658176682889462, dev_loss: 0.666331371813087, dev_acc: 0.6118012422360248, test_loss: 0.6661390781349746, test_acc: 0.6081560283687943
e: 66, train_loss: 0.6658051608204841, dev_loss: 0.6661709781203952, dev_acc: 0.6149068322981367, test_loss: 0.6659608489655434, test_acc: 0.6028368794326241
e: 67, train_loss: 0.6633849840015172, dev_loss: 0.6659810808689698, dev_acc: 0.6118012422360248, test_loss: 0.6657726331382778, test_acc: 0.601063829787234
e: 68, train_loss: 0.6662776139378548, dev_loss: 0.6657676009102638, dev_acc: 0.6024844720496895, test_loss: 0.6655260345086138, test_acc: 0.6028368794326241
e: 69, train_loss: 0.6695386825054884, dev_loss: 0.6656733902159685, dev_acc: 0.6211180124223602, test_loss: 0.6653099328490859, test_acc: 0.6063829787234043
e: 70, train_loss: 0.6679110628664493, dev_loss: 0.6655466925838719, dev_acc: 0.6149068322981367, test_loss: 0.665201668938001, test_acc: 0.6099290780141844
e: 71, train_loss: 0.6649759068191051, dev_loss: 0.6654222430834859, dev_acc: 0.6180124223602484, test_loss: 0.665052103684515, test_acc: 0.6028368794326241
e: 72, train_loss: 0.6624040635079146, dev_loss: 0.6652634208617003, dev_acc: 0.6180124223602484, test_loss: 0.664792306536267, test_acc: 0.6063829787234043
e: 73, train_loss: 0.671829874098301, dev_loss: 0.6651101949977578, dev_acc: 0.6242236024844721, test_loss: 0.6645731153428978, test_acc: 0.6134751773049646
e: 74, train_loss: 0.6724540723711252, dev_loss: 0.6652125615325774, dev_acc: 0.6180124223602484, test_loss: 0.6645081538891962, test_acc: 0.6099290780141844
e: 75, train_loss: 0.6691075375825166, dev_loss: 0.6652335254301937, dev_acc: 0.6149068322981367, test_loss: 0.6643945243447384, test_acc: 0.6081560283687943
e: 76, train_loss: 0.6656008603274822, dev_loss: 0.6651150311187187, dev_acc: 0.6055900621118012, test_loss: 0.6642206529195004, test_acc: 0.6152482269503546
e: 77, train_loss: 0.6607236443459987, dev_loss: 0.6648400247282122, dev_acc: 0.5993788819875776, test_loss: 0.6640622437740049, test_acc: 0.6152482269503546
e: 78, train_loss: 0.6648114628195763, dev_loss: 0.6646090471966667, dev_acc: 0.6024844720496895, test_loss: 0.6639009897220642, test_acc: 0.6117021276595744
e: 79, train_loss: 0.6685339709371328, dev_loss: 0.6644494086503983, dev_acc: 0.6149068322981367, test_loss: 0.6636380292739429, test_acc: 0.6117021276595744
e: 80, train_loss: 0.6606237058490515, dev_loss: 0.6642579862975185, dev_acc: 0.6118012422360248, test_loss: 0.6633339583186816, test_acc: 0.6152482269503546
e: 81, train_loss: 0.6637530003041029, dev_loss: 0.664137956611118, dev_acc: 0.6118012422360248, test_loss: 0.663162803211322, test_acc: 0.6134751773049646
e: 82, train_loss: 0.6743425985127688, dev_loss: 0.66414208451043, dev_acc: 0.6055900621118012, test_loss: 0.6630540999537664, test_acc: 0.6152482269503546
e: 83, train_loss: 0.6605799116641283, dev_loss: 0.6640685795812133, dev_acc: 0.6055900621118012, test_loss: 0.6628612090985403, test_acc: 0.6117021276595744
e: 84, train_loss: 0.6609429051578045, dev_loss: 0.663888596970102, dev_acc: 0.6086956521739131, test_loss: 0.6625783896034069, test_acc: 0.6117021276595744
e: 85, train_loss: 0.6600728568583727, dev_loss: 0.6636544846414779, dev_acc: 0.6086956521739131, test_loss: 0.6623590266503764, test_acc: 0.6117021276595744
e: 86, train_loss: 0.6620417958050966, dev_loss: 0.6633695908214735, dev_acc: 0.6118012422360248, test_loss: 0.6621265412697978, test_acc: 0.6152482269503546
e: 87, train_loss: 0.6639743136316538, dev_loss: 0.6631610141777844, dev_acc: 0.6086956521739131, test_loss: 0.6619284141359599, test_acc: 0.6152482269503546
e: 88, train_loss: 0.6602436469495296, dev_loss: 0.6630480941037954, dev_acc: 0.6024844720496895, test_loss: 0.6618613380648143, test_acc: 0.6134751773049646
e: 89, train_loss: 0.6555339834243059, dev_loss: 0.6628141952967792, dev_acc: 0.6086956521739131, test_loss: 0.6616001610531874, test_acc: 0.6117021276595744
e: 90, train_loss: 0.6651822135895491, dev_loss: 0.6625903440373284, dev_acc: 0.6055900621118012, test_loss: 0.6613152311150486, test_acc: 0.6117021276595744
e: 91, train_loss: 0.663080814242363, dev_loss: 0.6624820433233095, dev_acc: 0.6055900621118012, test_loss: 0.6610967316367524, test_acc: 0.6117021276595744
e: 92, train_loss: 0.6628251769542695, dev_loss: 0.6623739271615603, dev_acc: 0.6086956521739131, test_loss: 0.660955677905404, test_acc: 0.6099290780141844
e: 93, train_loss: 0.6660457501113415, dev_loss: 0.6621328219308616, dev_acc: 0.6055900621118012, test_loss: 0.6608214081126325, test_acc: 0.6117021276595744
e: 94, train_loss: 0.6590537998527288, dev_loss: 0.6618709228238704, dev_acc: 0.6086956521739131, test_loss: 0.6606896009216917, test_acc: 0.6099290780141844
e: 95, train_loss: 0.6568614123016596, dev_loss: 0.6615017717299254, dev_acc: 0.6086956521739131, test_loss: 0.6604592754432919, test_acc: 0.6081560283687943
e: 96, train_loss: 0.6631518274843693, dev_loss: 0.661361163169701, dev_acc: 0.6024844720496895, test_loss: 0.6602709819926015, test_acc: 0.6046099290780141
e: 97, train_loss: 0.6581334061026574, dev_loss: 0.6613750945522178, dev_acc: 0.6055900621118012, test_loss: 0.6600801802326178, test_acc: 0.6081560283687943
e: 98, train_loss: 0.654211922109127, dev_loss: 0.6612646839818599, dev_acc: 0.6086956521739131, test_loss: 0.6598678313671275, test_acc: 0.6117021276595744
e: 99, train_loss: 0.6572815782874822, dev_loss: 0.66092808257719, dev_acc: 0.6180124223602484, test_loss: 0.659613244631823, test_acc: 0.6134751773049646


t5-base
e: 0, train_loss: 0.6920077018737792, dev_loss: 0.6934184489413078, dev_acc: 0.4782608695652174, test_loss: 0.6931878971926709, test_acc: 0.4875886524822695
e: 1, train_loss: 0.6929621396660804, dev_loss: 0.693230116219254, dev_acc: 0.4813664596273292, test_loss: 0.6930828728574387, test_acc: 0.5301418439716312
e: 2, train_loss: 0.692686106979847, dev_loss: 0.6927471099803166, dev_acc: 0.515527950310559, test_loss: 0.692330262022661, test_acc: 0.5407801418439716
e: 3, train_loss: 0.6923840358257294, dev_loss: 0.6926766841307931, dev_acc: 0.5186335403726708, test_loss: 0.6920486627318335, test_acc: 0.5106382978723404
e: 4, train_loss: 0.6930855429768562, dev_loss: 0.6932179270693974, dev_acc: 0.4937888198757764, test_loss: 0.6914012148870644, test_acc: 0.5354609929078015
e: 5, train_loss: 0.6932471084594727, dev_loss: 0.6922687425376466, dev_acc: 0.546583850931677, test_loss: 0.6917228022365706, test_acc: 0.5336879432624113
e: 6, train_loss: 0.6906131347417831, dev_loss: 0.6908290724946845, dev_acc: 0.531055900621118, test_loss: 0.690138874430183, test_acc: 0.5390070921985816
e: 7, train_loss: 0.6910726065635682, dev_loss: 0.6916875970659789, dev_acc: 0.577639751552795, test_loss: 0.6898060114459789, test_acc: 0.5460992907801419
e: 8, train_loss: 0.6901000267267228, dev_loss: 0.68837277359844, dev_acc: 0.5341614906832298, test_loss: 0.6851473384081049, test_acc: 0.5868794326241135
e: 9, train_loss: 0.6842799907624721, dev_loss: 0.6900012787454617, dev_acc: 0.5372670807453416, test_loss: 0.694330282086599, test_acc: 0.5212765957446809
e: 10, train_loss: 0.677575493067503, dev_loss: 0.6921832052447041, dev_acc: 0.531055900621118, test_loss: 0.6945979729664664, test_acc: 0.5460992907801419
e: 11, train_loss: 0.6591529229283333, dev_loss: 0.724484258904035, dev_acc: 0.5372670807453416, test_loss: 0.7514332193115079, test_acc: 0.5141843971631206
e: 12, train_loss: 0.6702673283293843, dev_loss: 0.6862515794657031, dev_acc: 0.5527950310559007, test_loss: 0.6846977730433569, test_acc: 0.5460992907801419
e: 13, train_loss: 0.6428852747082711, dev_loss: 0.6823081855910905, dev_acc: 0.5590062111801242, test_loss: 0.6978572881549385, test_acc: 0.5407801418439716
e: 14, train_loss: 0.5894537035003304, dev_loss: 0.6912713322950446, dev_acc: 0.5248447204968945, test_loss: 0.6838260186003878, test_acc: 0.5443262411347518
e: 15, train_loss: 0.6116248355917633, dev_loss: 0.6864476063236686, dev_acc: 0.531055900621118, test_loss: 0.6815350660723998, test_acc: 0.574468085106383
e: 16, train_loss: 0.5979983848631382, dev_loss: 0.7400019166771299, dev_acc: 0.5217391304347826, test_loss: 0.7367693634327264, test_acc: 0.5620567375886525
e: 17, train_loss: 0.5068554627848789, dev_loss: 0.9439547739026205, dev_acc: 0.5590062111801242, test_loss: 1.0155628458153372, test_acc: 0.5833333333333334
e: 18, train_loss: 0.41804872675972, dev_loss: 0.719381377795098, dev_acc: 0.5248447204968945, test_loss: 0.7203585397293593, test_acc: 0.5762411347517731
e: 19, train_loss: 0.4724812369593419, dev_loss: 0.7088322736646818, dev_acc: 0.5559006211180124, test_loss: 0.7376461426615186, test_acc: 0.5620567375886525
e: 20, train_loss: 0.4207717349273153, dev_loss: 0.965438318325305, dev_acc: 0.5590062111801242, test_loss: 1.0813349191004868, test_acc: 0.5531914893617021
e: 21, train_loss: 0.25667442162396037, dev_loss: 1.1427670373133585, dev_acc: 0.577639751552795, test_loss: 1.273138037055915, test_acc: 0.574468085106383
e: 22, train_loss: 0.25264601099593664, dev_loss: 1.0365060822913852, dev_acc: 0.5900621118012422, test_loss: 1.1589371306499292, test_acc: 0.601063829787234
e: 23, train_loss: 0.21398683922640702, dev_loss: 1.1129140006349822, dev_acc: 0.5590062111801242, test_loss: 1.144012275871513, test_acc: 0.6081560283687943
e: 24, train_loss: 0.20677343815100174, dev_loss: 1.4979585055365567, dev_acc: 0.5590062111801242, test_loss: 1.6029029104968657, test_acc: 0.5957446808510638
e: 25, train_loss: 0.262114652087681, dev_loss: 0.943516607220597, dev_acc: 0.577639751552795, test_loss: 1.000827979154668, test_acc: 0.601063829787234
e: 26, train_loss: 0.2802346746174371, dev_loss: 0.9373843569821181, dev_acc: 0.5652173913043478, test_loss: 0.947063971594255, test_acc: 0.5921985815602837
e: 27, train_loss: 0.15383646133193907, dev_loss: 1.181757344893842, dev_acc: 0.5807453416149069, test_loss: 1.234540315535098, test_acc: 0.5762411347517731
e: 28, train_loss: 0.08244865371487094, dev_loss: 1.4813183555109586, dev_acc: 0.5745341614906833, test_loss: 1.5396902225078888, test_acc: 0.5957446808510638
e: 29, train_loss: 0.06175899365502198, dev_loss: 1.6050808561679286, dev_acc: 0.593167701863354, test_loss: 1.7388385465105567, test_acc: 0.5886524822695035


tohoku-bert-search
e: 0, train_loss: 0.6936138625144959, dev_loss: 0.6935397025591098, dev_acc: 0.4906832298136646, test_loss: 0.6934074076777654, test_acc: 0.5
e: 1, train_loss: 0.6933441357016563, dev_loss: 0.6932489198187123, dev_acc: 0.4968944099378882, test_loss: 0.6929907372868653, test_acc: 0.5088652482269503
e: 2, train_loss: 0.6931307252049446, dev_loss: 0.6929158820128589, dev_acc: 0.5062111801242236, test_loss: 0.692562720454331, test_acc: 0.5159574468085106
e: 3, train_loss: 0.6922374631166458, dev_loss: 0.6926148737439458, dev_acc: 0.5124223602484472, test_loss: 0.6921607479981496, test_acc: 0.5283687943262412
e: 4, train_loss: 0.6918849789500237, dev_loss: 0.6923740187787121, dev_acc: 0.5186335403726708, test_loss: 0.6918318941660807, test_acc: 0.5372340425531915
e: 5, train_loss: 0.6928182462453842, dev_loss: 0.6921127888356676, dev_acc: 0.5434782608695652, test_loss: 0.6914661484407195, test_acc: 0.5319148936170213
e: 6, train_loss: 0.6922780898809433, dev_loss: 0.69178560783404, dev_acc: 0.5403726708074534, test_loss: 0.6910425807144625, test_acc: 0.5443262411347518
e: 7, train_loss: 0.6913572249412536, dev_loss: 0.6915132221968278, dev_acc: 0.5372670807453416, test_loss: 0.6906156854849335, test_acc: 0.5460992907801419
e: 8, train_loss: 0.690950011074543, dev_loss: 0.6913195654101993, dev_acc: 0.5341614906832298, test_loss: 0.6902666444896807, test_acc: 0.5638297872340425
e: 9, train_loss: 0.6898891728520393, dev_loss: 0.6911051501028286, dev_acc: 0.5403726708074534, test_loss: 0.6899243683045637, test_acc: 0.5549645390070922
e: 10, train_loss: 0.692132198214531, dev_loss: 0.6910376785704808, dev_acc: 0.531055900621118, test_loss: 0.6897220698231501, test_acc: 0.5585106382978723
e: 11, train_loss: 0.6899878889918327, dev_loss: 0.6907874025543284, dev_acc: 0.5403726708074534, test_loss: 0.6893981635993254, test_acc: 0.5726950354609929
e: 12, train_loss: 0.6902424638271332, dev_loss: 0.6905239444711934, dev_acc: 0.546583850931677, test_loss: 0.6890806255611122, test_acc: 0.574468085106383
e: 13, train_loss: 0.6890035672187805, dev_loss: 0.6902399411112625, dev_acc: 0.5403726708074534, test_loss: 0.6886818724955227, test_acc: 0.5815602836879432
e: 14, train_loss: 0.6890629606246949, dev_loss: 0.6898887205568159, dev_acc: 0.5496894409937888, test_loss: 0.6881532226259827, test_acc: 0.5921985815602837
e: 15, train_loss: 0.6873184638619423, dev_loss: 0.6896145284546088, dev_acc: 0.5559006211180124, test_loss: 0.6876399961980522, test_acc: 0.599290780141844
e: 16, train_loss: 0.6902954177260399, dev_loss: 0.6894665763985296, dev_acc: 0.546583850931677, test_loss: 0.6872884406054274, test_acc: 0.5921985815602837
e: 17, train_loss: 0.6885690290331841, dev_loss: 0.6893242817857991, dev_acc: 0.546583850931677, test_loss: 0.6870429617293338, test_acc: 0.5815602836879432
e: 18, train_loss: 0.6873765417933464, dev_loss: 0.6890894492960865, dev_acc: 0.546583850931677, test_loss: 0.6866993483499433, test_acc: 0.5797872340425532
e: 19, train_loss: 0.6879263503551483, dev_loss: 0.6889458739239237, dev_acc: 0.5496894409937888, test_loss: 0.6864577959826652, test_acc: 0.5904255319148937
e: 20, train_loss: 0.691013379573822, dev_loss: 0.6888631635941334, dev_acc: 0.5590062111801242, test_loss: 0.6863387949923252, test_acc: 0.5815602836879432
e: 21, train_loss: 0.6900650672912597, dev_loss: 0.6887890807590129, dev_acc: 0.5527950310559007, test_loss: 0.6862016181785164, test_acc: 0.5833333333333334
e: 22, train_loss: 0.6871554501652718, dev_loss: 0.6886011406501628, dev_acc: 0.5527950310559007, test_loss: 0.685913253337779, test_acc: 0.5851063829787234
e: 23, train_loss: 0.6874753128886223, dev_loss: 0.6883182703338054, dev_acc: 0.5559006211180124, test_loss: 0.6855432285484693, test_acc: 0.5957446808510638
e: 24, train_loss: 0.6894434897005558, dev_loss: 0.6880982199811047, dev_acc: 0.5590062111801242, test_loss: 0.6852457497982268, test_acc: 0.5957446808510638
e: 25, train_loss: 0.6868698861300945, dev_loss: 0.687895601580602, dev_acc: 0.5714285714285714, test_loss: 0.6849497540834102, test_acc: 0.5975177304964538
e: 26, train_loss: 0.6862268909215927, dev_loss: 0.6878221538496314, dev_acc: 0.5714285714285714, test_loss: 0.68468990647201, test_acc: 0.5939716312056738
e: 27, train_loss: 0.6881680883765221, dev_loss: 0.6877545527419688, dev_acc: 0.5807453416149069, test_loss: 0.6845097520672683, test_acc: 0.599290780141844
e: 28, train_loss: 0.6880674895644188, dev_loss: 0.6876586929241323, dev_acc: 0.5745341614906833, test_loss: 0.6843199503759966, test_acc: 0.5975177304964538
e: 29, train_loss: 0.6866715426445007, dev_loss: 0.6875122618971404, dev_acc: 0.5807453416149069, test_loss: 0.6841049405699926, test_acc: 0.5975177304964538
e: 30, train_loss: 0.6893379039168358, dev_loss: 0.6874303377192953, dev_acc: 0.5838509316770186, test_loss: 0.6839285511919793, test_acc: 0.6081560283687943
e: 31, train_loss: 0.6890790514349937, dev_loss: 0.6873688503451969, dev_acc: 0.5838509316770186, test_loss: 0.6838230777082713, test_acc: 0.6028368794326241
e: 32, train_loss: 0.6858600866794586, dev_loss: 0.6872872351119237, dev_acc: 0.577639751552795, test_loss: 0.6836643740006373, test_acc: 0.6046099290780141
e: 33, train_loss: 0.6868155688047409, dev_loss: 0.6871259477197754, dev_acc: 0.5807453416149069, test_loss: 0.6834236324256193, test_acc: 0.6063829787234043
e: 34, train_loss: 0.686900076508522, dev_loss: 0.6870445660922838, dev_acc: 0.5807453416149069, test_loss: 0.6832707492595024, test_acc: 0.599290780141844
e: 35, train_loss: 0.6866524793505668, dev_loss: 0.6869683010237557, dev_acc: 0.5869565217391305, test_loss: 0.6831071227788925, test_acc: 0.6046099290780141
e: 36, train_loss: 0.6879284374117851, dev_loss: 0.6868891886302403, dev_acc: 0.5807453416149069, test_loss: 0.682994322573885, test_acc: 0.6117021276595744
e: 37, train_loss: 0.6853404222428798, dev_loss: 0.6868407011772535, dev_acc: 0.577639751552795, test_loss: 0.6828914797263788, test_acc: 0.6152482269503546
e: 38, train_loss: 0.6859791131019592, dev_loss: 0.6867561269991146, dev_acc: 0.593167701863354, test_loss: 0.6827474231204242, test_acc: 0.6081560283687943
e: 39, train_loss: 0.6875123087763786, dev_loss: 0.6867142203061477, dev_acc: 0.5993788819875776, test_loss: 0.6826512530340371, test_acc: 0.6134751773049646
e: 40, train_loss: 0.684601177662611, dev_loss: 0.6865202845623775, dev_acc: 0.6024844720496895, test_loss: 0.6823952758143134, test_acc: 0.6152482269503546
e: 41, train_loss: 0.6871387366056442, dev_loss: 0.6863219832411463, dev_acc: 0.593167701863354, test_loss: 0.6821530468709079, test_acc: 0.6152482269503546
e: 42, train_loss: 0.6874164627790451, dev_loss: 0.686119531437477, dev_acc: 0.593167701863354, test_loss: 0.681911475269507, test_acc: 0.6117021276595744
e: 43, train_loss: 0.6835538631975651, dev_loss: 0.686002135461902, dev_acc: 0.593167701863354, test_loss: 0.6817336923158761, test_acc: 0.6081560283687943
e: 44, train_loss: 0.6896548708677291, dev_loss: 0.6860061584052092, dev_acc: 0.5962732919254659, test_loss: 0.6816237828304582, test_acc: 0.6117021276595744
e: 45, train_loss: 0.688523102670908, dev_loss: 0.6860340150616924, dev_acc: 0.5993788819875776, test_loss: 0.6815840142732816, test_acc: 0.6134751773049646
e: 46, train_loss: 0.6852524676024914, dev_loss: 0.6858877432272301, dev_acc: 0.6086956521739131, test_loss: 0.6814305076362394, test_acc: 0.6134751773049646
e: 47, train_loss: 0.6861896669268608, dev_loss: 0.6858221949627681, dev_acc: 0.5962732919254659, test_loss: 0.6813345149899206, test_acc: 0.6063829787234043
e: 48, train_loss: 0.6821684205532074, dev_loss: 0.6857099066609922, dev_acc: 0.6024844720496895, test_loss: 0.6811680315022773, test_acc: 0.6081560283687943
e: 49, train_loss: 0.6886721945703029, dev_loss: 0.6856849541575272, dev_acc: 0.6024844720496895, test_loss: 0.6810502787219718, test_acc: 0.6046099290780141
e: 50, train_loss: 0.6805607107579708, dev_loss: 0.685622616399149, dev_acc: 0.5993788819875776, test_loss: 0.6809336315866903, test_acc: 0.6063829787234043
e: 51, train_loss: 0.6848560373485089, dev_loss: 0.6856012174061367, dev_acc: 0.5869565217391305, test_loss: 0.6808199003233132, test_acc: 0.6099290780141844
e: 52, train_loss: 0.6871132576465606, dev_loss: 0.6855364789873917, dev_acc: 0.5993788819875776, test_loss: 0.6807360789665939, test_acc: 0.6117021276595744
e: 53, train_loss: 0.6842141044437885, dev_loss: 0.6853577555706782, dev_acc: 0.6055900621118012, test_loss: 0.6805603917925915, test_acc: 0.6170212765957447
e: 54, train_loss: 0.6849203606843949, dev_loss: 0.6852650174072811, dev_acc: 0.6024844720496895, test_loss: 0.6804145686592616, test_acc: 0.6099290780141844
e: 55, train_loss: 0.6872816348969937, dev_loss: 0.685184173332238, dev_acc: 0.5962732919254659, test_loss: 0.6803416092979148, test_acc: 0.601063829787234
e: 56, train_loss: 0.68060829693079, dev_loss: 0.6850997183633887, dev_acc: 0.5900621118012422, test_loss: 0.6801848632435427, test_acc: 0.6099290780141844
e: 57, train_loss: 0.6876678286194802, dev_loss: 0.6850314906665257, dev_acc: 0.5900621118012422, test_loss: 0.6800412763940528, test_acc: 0.6099290780141844
e: 58, train_loss: 0.6869501080811024, dev_loss: 0.6849571351679216, dev_acc: 0.5869565217391305, test_loss: 0.6800369031674472, test_acc: 0.6081560283687943
e: 59, train_loss: 0.6841952636539936, dev_loss: 0.6849208559308734, dev_acc: 0.5900621118012422, test_loss: 0.6800253670266334, test_acc: 0.6063829787234043
e: 60, train_loss: 0.6856814146339894, dev_loss: 0.684855487220776, dev_acc: 0.5962732919254659, test_loss: 0.6799820966965763, test_acc: 0.6117021276595744
e: 61, train_loss: 0.68607632163167, dev_loss: 0.6848109014656233, dev_acc: 0.5962732919254659, test_loss: 0.6799272635938428, test_acc: 0.6046099290780141
e: 62, train_loss: 0.6820347284078598, dev_loss: 0.6846490493102103, dev_acc: 0.5993788819875776, test_loss: 0.6797177661606606, test_acc: 0.6081560283687943
e: 63, train_loss: 0.6826306659281254, dev_loss: 0.6844406570336834, dev_acc: 0.593167701863354, test_loss: 0.679453377922376, test_acc: 0.6063829787234043
e: 64, train_loss: 0.6820174936950206, dev_loss: 0.6842839864099989, dev_acc: 0.5962732919254659, test_loss: 0.6792592054351847, test_acc: 0.6028368794326241
e: 65, train_loss: 0.6849304976165295, dev_loss: 0.6842274630662077, dev_acc: 0.5900621118012422, test_loss: 0.6791828299033726, test_acc: 0.6081560283687943
e: 66, train_loss: 0.6813553712964058, dev_loss: 0.6841362498191573, dev_acc: 0.5993788819875776, test_loss: 0.679022972254043, test_acc: 0.6081560283687943
e: 67, train_loss: 0.6850333994030953, dev_loss: 0.6840264863849427, dev_acc: 0.6024844720496895, test_loss: 0.6788956083304493, test_acc: 0.6134751773049646
e: 68, train_loss: 0.6848481628894806, dev_loss: 0.6839863354374903, dev_acc: 0.5993788819875776, test_loss: 0.678891215341311, test_acc: 0.6117021276595744
e: 69, train_loss: 0.6814068730473518, dev_loss: 0.6838775490011487, dev_acc: 0.5993788819875776, test_loss: 0.6787092642898255, test_acc: 0.6117021276595744
e: 70, train_loss: 0.6859124186635017, dev_loss: 0.6838379489338916, dev_acc: 0.5962732919254659, test_loss: 0.6785919105118894, test_acc: 0.6063829787234043
e: 71, train_loss: 0.6841476064324379, dev_loss: 0.68369024233048, dev_acc: 0.5962732919254659, test_loss: 0.6783967430287219, test_acc: 0.6117021276595744
e: 72, train_loss: 0.6844039385318756, dev_loss: 0.6836043227903591, dev_acc: 0.593167701863354, test_loss: 0.6782473753741447, test_acc: 0.6152482269503546
e: 73, train_loss: 0.6837659240365028, dev_loss: 0.6836086067353716, dev_acc: 0.593167701863354, test_loss: 0.6782300937788707, test_acc: 0.6117021276595744
e: 74, train_loss: 0.6869245877563953, dev_loss: 0.6836026710753115, dev_acc: 0.5962732919254659, test_loss: 0.678151454591582, test_acc: 0.6063829787234043
e: 75, train_loss: 0.6831673287451268, dev_loss: 0.6836299054000688, dev_acc: 0.593167701863354, test_loss: 0.6781373041424346, test_acc: 0.6028368794326241
e: 76, train_loss: 0.6854040107429028, dev_loss: 0.6835356593502234, dev_acc: 0.5900621118012422, test_loss: 0.6780689340745304, test_acc: 0.6046099290780141
e: 77, train_loss: 0.678353501111269, dev_loss: 0.6833793234010661, dev_acc: 0.5869565217391305, test_loss: 0.6778591707982915, test_acc: 0.6081560283687943
e: 78, train_loss: 0.6820757780969143, dev_loss: 0.6832726803255378, dev_acc: 0.5900621118012422, test_loss: 0.6776737894149537, test_acc: 0.6046099290780141
e: 79, train_loss: 0.6816446945071221, dev_loss: 0.6832282095222, dev_acc: 0.5900621118012422, test_loss: 0.677490826722578, test_acc: 0.6099290780141844
e: 80, train_loss: 0.6805655043423176, dev_loss: 0.6831243963715452, dev_acc: 0.5869565217391305, test_loss: 0.6772743887829442, test_acc: 0.6046099290780141
e: 81, train_loss: 0.6808863610327244, dev_loss: 0.6829980580332857, dev_acc: 0.593167701863354, test_loss: 0.6771882324882433, test_acc: 0.6046099290780141
e: 82, train_loss: 0.6817124342322349, dev_loss: 0.6829205120202178, dev_acc: 0.593167701863354, test_loss: 0.6770912390862797, test_acc: 0.6046099290780141
e: 83, train_loss: 0.6770754920840263, dev_loss: 0.6827895165600392, dev_acc: 0.5962732919254659, test_loss: 0.6769027541926567, test_acc: 0.6046099290780141
e: 84, train_loss: 0.6811616016626358, dev_loss: 0.6827134689929323, dev_acc: 0.5993788819875776, test_loss: 0.6767506735227632, test_acc: 0.6134751773049646
e: 85, train_loss: 0.6829823806285859, dev_loss: 0.6826868003569775, dev_acc: 0.5993788819875776, test_loss: 0.6766683086752892, test_acc: 0.6134751773049646
e: 86, train_loss: 0.685588268995285, dev_loss: 0.6826466383030696, dev_acc: 0.6024844720496895, test_loss: 0.6766108952938242, test_acc: 0.6099290780141844
e: 87, train_loss: 0.6813640987277031, dev_loss: 0.6825910242077726, dev_acc: 0.6055900621118012, test_loss: 0.6766090954766206, test_acc: 0.6117021276595744
e: 88, train_loss: 0.6801931271851063, dev_loss: 0.6824542507992027, dev_acc: 0.6055900621118012, test_loss: 0.6764965604805777, test_acc: 0.6028368794326241
e: 89, train_loss: 0.6806743997335434, dev_loss: 0.682409110461703, dev_acc: 0.6024844720496895, test_loss: 0.6764220861678428, test_acc: 0.6099290780141844
e: 90, train_loss: 0.6812490362226963, dev_loss: 0.682326216505181, dev_acc: 0.6055900621118012, test_loss: 0.6763207576904736, test_acc: 0.6028368794326241
e: 91, train_loss: 0.678188225388527, dev_loss: 0.6822745155843889, dev_acc: 0.5993788819875776, test_loss: 0.6762449940574085, test_acc: 0.6046099290780141
e: 92, train_loss: 0.6813188326954842, dev_loss: 0.6822089708369711, dev_acc: 0.5993788819875776, test_loss: 0.6761950351139332, test_acc: 0.6081560283687943
e: 93, train_loss: 0.685962826937437, dev_loss: 0.6822176094380965, dev_acc: 0.5962732919254659, test_loss: 0.6761730105635968, test_acc: 0.6081560283687943
e: 94, train_loss: 0.6834763711392879, dev_loss: 0.6822564461216423, dev_acc: 0.593167701863354, test_loss: 0.6761680430342966, test_acc: 0.6063829787234043
e: 95, train_loss: 0.6816840192973613, dev_loss: 0.6822188592845608, dev_acc: 0.5993788819875776, test_loss: 0.6760720977863521, test_acc: 0.6028368794326241
e: 96, train_loss: 0.6824144937694073, dev_loss: 0.6821640554051962, dev_acc: 0.5993788819875776, test_loss: 0.6759644340433127, test_acc: 0.6028368794326241
e: 97, train_loss: 0.6847426337003708, dev_loss: 0.6821729544527042, dev_acc: 0.6024844720496895, test_loss: 0.6759708720318814, test_acc: 0.6063829787234043
e: 98, train_loss: 0.6825190255045891, dev_loss: 0.6821264578318744, dev_acc: 0.6055900621118012, test_loss: 0.6759312541243878, test_acc: 0.6063829787234043
e: 99, train_loss: 0.6815605650544166, dev_loss: 0.6820769284082495, dev_acc: 0.6055900621118012, test_loss: 0.675779531346568, test_acc: 0.6028368794326241

t5-base
e: 0, train_loss: 0.69295100492239, dev_loss: 0.6932233055556043, dev_acc: 0.5186335403726708, test_loss: 0.693152867310436, test_acc: 0.5035460992907801
e: 1, train_loss: 0.6931651350259781, dev_loss: 0.6932027060422838, dev_acc: 0.5248447204968945, test_loss: 0.6931405648694816, test_acc: 0.5053191489361702
e: 2, train_loss: 0.692668488562107, dev_loss: 0.6931469832888301, dev_acc: 0.5062111801242236, test_loss: 0.6931184764872206, test_acc: 0.5141843971631206
e: 3, train_loss: 0.6934316868782043, dev_loss: 0.6931100672816638, dev_acc: 0.5062111801242236, test_loss: 0.6930834933164272, test_acc: 0.5088652482269503
e: 4, train_loss: 0.6930154339075089, dev_loss: 0.6931131943042234, dev_acc: 0.4813664596273292, test_loss: 0.6930414792282361, test_acc: 0.5159574468085106
e: 5, train_loss: 0.6937456955313682, dev_loss: 0.6930726705500798, dev_acc: 0.4782608695652174, test_loss: 0.6929953798968741, test_acc: 0.5230496453900709
e: 6, train_loss: 0.6934891489744186, dev_loss: 0.6930270868798961, dev_acc: 0.4782608695652174, test_loss: 0.692963127548813, test_acc: 0.5141843971631206
e: 7, train_loss: 0.6932195603251458, dev_loss: 0.6929768992136724, dev_acc: 0.4813664596273292, test_loss: 0.692923016581975, test_acc: 0.5141843971631206
e: 8, train_loss: 0.6929701972007751, dev_loss: 0.6929773275526414, dev_acc: 0.4782608695652174, test_loss: 0.692927562387277, test_acc: 0.5106382978723404
e: 9, train_loss: 0.6935134708285332, dev_loss: 0.6929775187687844, dev_acc: 0.4906832298136646, test_loss: 0.6929111184803307, test_acc: 0.5141843971631206
e: 10, train_loss: 0.6937148065567017, dev_loss: 0.6929934245088826, dev_acc: 0.4906832298136646, test_loss: 0.6928939480096736, test_acc: 0.5070921985815603
e: 11, train_loss: 0.6924834634065629, dev_loss: 0.6929394913756329, dev_acc: 0.5031055900621118, test_loss: 0.6928368086087788, test_acc: 0.5088652482269503
e: 12, train_loss: 0.6926027191281319, dev_loss: 0.6928494137636623, dev_acc: 0.5093167701863354, test_loss: 0.6927503482669803, test_acc: 0.5212765957446809
e: 13, train_loss: 0.691960627734661, dev_loss: 0.6928322387408026, dev_acc: 0.5341614906832298, test_loss: 0.6926868055940519, test_acc: 0.5336879432624113
e: 14, train_loss: 0.6923525869846344, dev_loss: 0.692818254047299, dev_acc: 0.5186335403726708, test_loss: 0.6926493111020284, test_acc: 0.526595744680851
e: 15, train_loss: 0.6926623747944832, dev_loss: 0.6928470299480864, dev_acc: 0.5248447204968945, test_loss: 0.6926551005730393, test_acc: 0.524822695035461
e: 16, train_loss: 0.6938695982098579, dev_loss: 0.6927938352090232, dev_acc: 0.4968944099378882, test_loss: 0.6926244425435438, test_acc: 0.5195035460992907
e: 17, train_loss: 0.6921008920073509, dev_loss: 0.6927407818921605, dev_acc: 0.5124223602484472, test_loss: 0.6925803103768233, test_acc: 0.5283687943262412
e: 18, train_loss: 0.6926827163696289, dev_loss: 0.6927432623338996, dev_acc: 0.515527950310559, test_loss: 0.6925485192249853, test_acc: 0.5319148936170213
e: 19, train_loss: 0.6928669701218605, dev_loss: 0.69273481550424, dev_acc: 0.531055900621118, test_loss: 0.6925108144257931, test_acc: 0.5283687943262412
e: 20, train_loss: 0.6936916890144348, dev_loss: 0.6926464464353479, dev_acc: 0.5248447204968945, test_loss: 0.6924857835397653, test_acc: 0.526595744680851
e: 21, train_loss: 0.693276721060276, dev_loss: 0.6925756897985565, dev_acc: 0.531055900621118, test_loss: 0.6924742103045713, test_acc: 0.5124113475177305
e: 22, train_loss: 0.693346793293953, dev_loss: 0.6925388219193642, dev_acc: 0.5341614906832298, test_loss: 0.6924780878826235, test_acc: 0.50177304964539
e: 23, train_loss: 0.6926479839086532, dev_loss: 0.6925248627707085, dev_acc: 0.531055900621118, test_loss: 0.6924591971204636, test_acc: 0.5177304964539007
e: 24, train_loss: 0.6928511652350425, dev_loss: 0.6925199407598247, dev_acc: 0.5434782608695652, test_loss: 0.6924453976940601, test_acc: 0.526595744680851
e: 25, train_loss: 0.691502125620842, dev_loss: 0.6924343729241295, dev_acc: 0.5279503105590062, test_loss: 0.6924106528361639, test_acc: 0.5230496453900709
e: 26, train_loss: 0.6922089520096779, dev_loss: 0.6923965796920823, dev_acc: 0.531055900621118, test_loss: 0.6924188355816171, test_acc: 0.526595744680851
e: 27, train_loss: 0.691931710422039, dev_loss: 0.6923903539684249, dev_acc: 0.5248447204968945, test_loss: 0.6924072588589174, test_acc: 0.5283687943262412
e: 28, train_loss: 0.6922085973620414, dev_loss: 0.6923392215130492, dev_acc: 0.5341614906832298, test_loss: 0.6923671724103021, test_acc: 0.5354609929078015
e: 29, train_loss: 0.6921842014193534, dev_loss: 0.6922627855902133, dev_acc: 0.546583850931677, test_loss: 0.692312535664714, test_acc: 0.5407801418439716
e: 30, train_loss: 0.6938292233347892, dev_loss: 0.6921961261248737, dev_acc: 0.5372670807453416, test_loss: 0.69227353404177, test_acc: 0.549645390070922
e: 31, train_loss: 0.6914637830853462, dev_loss: 0.692146893243612, dev_acc: 0.5434782608695652, test_loss: 0.692221666375796, test_acc: 0.549645390070922
e: 32, train_loss: 0.6935064471960067, dev_loss: 0.6921070867813892, dev_acc: 0.5403726708074534, test_loss: 0.6921827405268419, test_acc: 0.5585106382978723
e: 33, train_loss: 0.6948085733056069, dev_loss: 0.6921213224807882, dev_acc: 0.5403726708074534, test_loss: 0.6921583672998645, test_acc: 0.5514184397163121
e: 34, train_loss: 0.6929170789122582, dev_loss: 0.6920575453257709, dev_acc: 0.5434782608695652, test_loss: 0.692111092784726, test_acc: 0.5460992907801419
e: 35, train_loss: 0.6924350134134293, dev_loss: 0.6920333750869917, dev_acc: 0.5372670807453416, test_loss: 0.6920981123937783, test_acc: 0.5390070921985816
e: 36, train_loss: 0.6924579375982285, dev_loss: 0.6919635917459216, dev_acc: 0.5559006211180124, test_loss: 0.6920786450306574, test_acc: 0.5443262411347518
e: 37, train_loss: 0.6922373832464218, dev_loss: 0.6919423419496288, dev_acc: 0.5559006211180124, test_loss: 0.6920615328964612, test_acc: 0.5531914893617021
e: 38, train_loss: 0.6938401072621345, dev_loss: 0.691957777516442, dev_acc: 0.5559006211180124, test_loss: 0.6920651627559189, test_acc: 0.5585106382978723
e: 39, train_loss: 0.693418890595436, dev_loss: 0.6920210543256369, dev_acc: 0.5838509316770186, test_loss: 0.692082128520553, test_acc: 0.5602836879432624
e: 40, train_loss: 0.6917574394345284, dev_loss: 0.6919965860636338, dev_acc: 0.5714285714285714, test_loss: 0.6920705105818755, test_acc: 0.5549645390070922
e: 41, train_loss: 0.6919137840867042, dev_loss: 0.6919337873873503, dev_acc: 0.5714285714285714, test_loss: 0.6920524671779457, test_acc: 0.549645390070922
e: 42, train_loss: 0.6936752788424492, dev_loss: 0.6919257818541912, dev_acc: 0.5652173913043478, test_loss: 0.6920534628080138, test_acc: 0.5514184397163121
e: 43, train_loss: 0.6944098298549652, dev_loss: 0.691976513181414, dev_acc: 0.5745341614906833, test_loss: 0.6920892329926186, test_acc: 0.5514184397163121
e: 44, train_loss: 0.6921383420228958, dev_loss: 0.6919554727418082, dev_acc: 0.5838509316770186, test_loss: 0.6920806741249477, test_acc: 0.5585106382978723
e: 45, train_loss: 0.692567532479763, dev_loss: 0.6919270642795918, dev_acc: 0.5714285714285714, test_loss: 0.6920515248116027, test_acc: 0.5585106382978723
e: 46, train_loss: 0.692762070953846, dev_loss: 0.6919520393661831, dev_acc: 0.5652173913043478, test_loss: 0.6920500299397935, test_acc: 0.5549645390070922
e: 47, train_loss: 0.6921824018359184, dev_loss: 0.6919241553137762, dev_acc: 0.562111801242236, test_loss: 0.6920296078455364, test_acc: 0.549645390070922
e: 48, train_loss: 0.6924138078689576, dev_loss: 0.6918850439305632, dev_acc: 0.5652173913043478, test_loss: 0.6920261684250324, test_acc: 0.5585106382978723
e: 49, train_loss: 0.6925104104876518, dev_loss: 0.6918592865792861, dev_acc: 0.5869565217391305, test_loss: 0.6920009600989362, test_acc: 0.5602836879432624
e: 50, train_loss: 0.6918742264509201, dev_loss: 0.6918063637632761, dev_acc: 0.593167701863354, test_loss: 0.6919825151034281, test_acc: 0.5638297872340425
e: 51, train_loss: 0.6913689428567886, dev_loss: 0.6917907762231293, dev_acc: 0.5993788819875776, test_loss: 0.6919732243879467, test_acc: 0.5602836879432624
e: 52, train_loss: 0.6922471399903297, dev_loss: 0.6917855919147871, dev_acc: 0.6055900621118012, test_loss: 0.6919610795188458, test_acc: 0.5549645390070922
e: 53, train_loss: 0.6918000160455704, dev_loss: 0.6917550352789601, dev_acc: 0.6024844720496895, test_loss: 0.6919328496388509, test_acc: 0.5691489361702128
e: 54, train_loss: 0.6916329715251922, dev_loss: 0.6917134029154451, dev_acc: 0.6273291925465838, test_loss: 0.6919098740562479, test_acc: 0.5567375886524822
e: 55, train_loss: 0.6928528280258178, dev_loss: 0.6916667748682247, dev_acc: 0.6180124223602484, test_loss: 0.691865806989636, test_acc: 0.5585106382978723
e: 56, train_loss: 0.6927634009718895, dev_loss: 0.6916093533823949, dev_acc: 0.6024844720496895, test_loss: 0.6918489593774715, test_acc: 0.5602836879432624
e: 57, train_loss: 0.6918546366095543, dev_loss: 0.6915731650331746, dev_acc: 0.5993788819875776, test_loss: 0.6918192408609052, test_acc: 0.5602836879432624
e: 58, train_loss: 0.6916516676545144, dev_loss: 0.6915423353636487, dev_acc: 0.5962732919254659, test_loss: 0.6918063868656226, test_acc: 0.5602836879432624
e: 59, train_loss: 0.6929599236249924, dev_loss: 0.6915395150643698, dev_acc: 0.5900621118012422, test_loss: 0.6917898066289035, test_acc: 0.5620567375886525
e: 60, train_loss: 0.6917463476061821, dev_loss: 0.6915193659178218, dev_acc: 0.5962732919254659, test_loss: 0.6917764373282169, test_acc: 0.5638297872340425
e: 61, train_loss: 0.6923527521491051, dev_loss: 0.6914967840872936, dev_acc: 0.5869565217391305, test_loss: 0.6917639365221592, test_acc: 0.5691489361702128
e: 62, train_loss: 0.6932514860630036, dev_loss: 0.691574453937341, dev_acc: 0.5900621118012422, test_loss: 0.6917753202695374, test_acc: 0.5549645390070922
e: 63, train_loss: 0.691715406358242, dev_loss: 0.6915701946116383, dev_acc: 0.5962732919254659, test_loss: 0.6917478876545075, test_acc: 0.5673758865248227
e: 64, train_loss: 0.6915792821645736, dev_loss: 0.691477412392634, dev_acc: 0.5993788819875776, test_loss: 0.6917050321262779, test_acc: 0.5709219858156028
e: 65, train_loss: 0.6919259749650956, dev_loss: 0.6914032569953373, dev_acc: 0.5807453416149069, test_loss: 0.6916915466810795, test_acc: 0.574468085106383
e: 66, train_loss: 0.6917294124364853, dev_loss: 0.691346329931887, dev_acc: 0.5745341614906833, test_loss: 0.6916946328066765, test_acc: 0.5726950354609929
e: 67, train_loss: 0.6917290006875991, dev_loss: 0.6913227509267582, dev_acc: 0.5838509316770186, test_loss: 0.6916872724785027, test_acc: 0.5656028368794326
e: 68, train_loss: 0.6913880846500396, dev_loss: 0.691310557519427, dev_acc: 0.5807453416149069, test_loss: 0.691664902665091, test_acc: 0.5638297872340425
e: 69, train_loss: 0.6923871589899063, dev_loss: 0.6913107148608806, dev_acc: 0.593167701863354, test_loss: 0.6916373115270695, test_acc: 0.5656028368794326
e: 70, train_loss: 0.6914007003307343, dev_loss: 0.691277500450241, dev_acc: 0.5838509316770186, test_loss: 0.6916141554396203, test_acc: 0.5549645390070922
e: 71, train_loss: 0.6911227178573608, dev_loss: 0.6911729131796345, dev_acc: 0.5838509316770186, test_loss: 0.6915477359971256, test_acc: 0.5478723404255319
e: 72, train_loss: 0.6923215805888175, dev_loss: 0.6911530265156527, dev_acc: 0.5745341614906833, test_loss: 0.6915401457049323, test_acc: 0.549645390070922
e: 73, train_loss: 0.6938044316768647, dev_loss: 0.6911255361870949, dev_acc: 0.5869565217391305, test_loss: 0.6915239565972741, test_acc: 0.5585106382978723
e: 74, train_loss: 0.6904354759454727, dev_loss: 0.6910902739311597, dev_acc: 0.5900621118012422, test_loss: 0.6914912934633012, test_acc: 0.5602836879432624
e: 75, train_loss: 0.6924673531055451, dev_loss: 0.6910041141213837, dev_acc: 0.5993788819875776, test_loss: 0.6914498119066793, test_acc: 0.5549645390070922
e: 76, train_loss: 0.6920863901376724, dev_loss: 0.6909465352941003, dev_acc: 0.5869565217391305, test_loss: 0.6914209281721859, test_acc: 0.5602836879432624
e: 77, train_loss: 0.6927076077461243, dev_loss: 0.6909557418053195, dev_acc: 0.5838509316770186, test_loss: 0.6914157797681525, test_acc: 0.5549645390070922
e: 78, train_loss: 0.6922558603286744, dev_loss: 0.6909366896063645, dev_acc: 0.5714285714285714, test_loss: 0.6913837568557009, test_acc: 0.5585106382978723
e: 79, train_loss: 0.6898511931300163, dev_loss: 0.6908666630339179, dev_acc: 0.5745341614906833, test_loss: 0.6913077057676112, test_acc: 0.5602836879432624
e: 80, train_loss: 0.6926815725564957, dev_loss: 0.6908927154466973, dev_acc: 0.5900621118012422, test_loss: 0.6912828276977472, test_acc: 0.5656028368794326
e: 81, train_loss: 0.692347885131836, dev_loss: 0.6909594891234214, dev_acc: 0.6118012422360248, test_loss: 0.6913013423376895, test_acc: 0.5709219858156028
e: 82, train_loss: 0.6914393339753151, dev_loss: 0.6909680179557445, dev_acc: 0.6024844720496895, test_loss: 0.6912898430798916, test_acc: 0.574468085106383
e: 83, train_loss: 0.691889146387577, dev_loss: 0.690908756315338, dev_acc: 0.593167701863354, test_loss: 0.6912458022224143, test_acc: 0.5833333333333334
e: 84, train_loss: 0.6906909415721894, dev_loss: 0.6908071887048876, dev_acc: 0.5807453416149069, test_loss: 0.6912083799111928, test_acc: 0.5762411347517731
e: 85, train_loss: 0.6922439197897912, dev_loss: 0.6907761678192186, dev_acc: 0.593167701863354, test_loss: 0.6912059640208035, test_acc: 0.5673758865248227
e: 86, train_loss: 0.6938228067159653, dev_loss: 0.6907778830261704, dev_acc: 0.5900621118012422, test_loss: 0.6911909920526734, test_acc: 0.5673758865248227
e: 87, train_loss: 0.6927301979064941, dev_loss: 0.690799147445963, dev_acc: 0.593167701863354, test_loss: 0.6911971298515374, test_acc: 0.5691489361702128
e: 88, train_loss: 0.6918067716360092, dev_loss: 0.6908075311539336, dev_acc: 0.5838509316770186, test_loss: 0.6911860441273832, test_acc: 0.5638297872340425
e: 89, train_loss: 0.692686386168003, dev_loss: 0.6907800439722049, dev_acc: 0.5807453416149069, test_loss: 0.6911667127769889, test_acc: 0.5585106382978723
e: 90, train_loss: 0.6918433025479317, dev_loss: 0.6907746140260874, dev_acc: 0.5869565217391305, test_loss: 0.6911368522238224, test_acc: 0.5602836879432624
e: 91, train_loss: 0.6919974851608276, dev_loss: 0.6907875199125421, dev_acc: 0.5869565217391305, test_loss: 0.6911166480458375, test_acc: 0.5673758865248227
e: 92, train_loss: 0.6933421931266784, dev_loss: 0.6907755949112199, dev_acc: 0.5993788819875776, test_loss: 0.6911184582727176, test_acc: 0.5656028368794326
e: 93, train_loss: 0.6931458500027656, dev_loss: 0.6907557149111114, dev_acc: 0.5962732919254659, test_loss: 0.6911268491060176, test_acc: 0.5762411347517731
e: 94, train_loss: 0.6916885944008827, dev_loss: 0.6907814630070088, dev_acc: 0.5869565217391305, test_loss: 0.6911448023632063, test_acc: 0.5709219858156028
e: 95, train_loss: 0.6916460695266724, dev_loss: 0.6908000552135966, dev_acc: 0.5838509316770186, test_loss: 0.6911482690496648, test_acc: 0.5709219858156028
e: 96, train_loss: 0.6900260111689568, dev_loss: 0.6907417354006204, dev_acc: 0.5900621118012422, test_loss: 0.6911004737336585, test_acc: 0.5726950354609929
e: 97, train_loss: 0.6910287758708, dev_loss: 0.6906745361615412, dev_acc: 0.593167701863354, test_loss: 0.6910636552893523, test_acc: 0.5762411347517731
e: 98, train_loss: 0.6909538558125496, dev_loss: 0.6906782111025745, dev_acc: 0.5993788819875776, test_loss: 0.6910543926852815, test_acc: 0.5709219858156028
e: 99, train_loss: 0.6900167833566666, dev_loss: 0.6906618137167108, dev_acc: 0.5962732919254659, test_loss: 0.6910297108668808, test_acc: 0.5620567375886525

'''