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

from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model
from transformers import T5Tokenizer, T5TokenizerFast, AutoModelForCausalLM, T5Model
from transformers import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertJapaneseTokenizer, BertModel

class BertJapaneseTokenizerFast(BertJapaneseTokenizer):
  def __call__(self,text,text_pair=None,return_offsets_mapping=True,**kwargs):
    v=super().__call__(text=text,text_pair=text_pair,return_offsets_mapping=False,**kwargs)
    if return_offsets_mapping:
      import tokenizations
      if type(text)==str:
        z=zip([v["input_ids"].squeeze(0)],[text],[text_pair] if text_pair else [""])
      else:
        z=zip(v["input_ids"].squeeze(0),text,text_pair if text_pair else [""]*len(text))
      w=[]
      for a,b,c in z:
        a2b,b2a=tokenizations.get_alignments(self.convert_ids_to_tokens(a),b+c)
        x=[]
        for i,t in enumerate(a2b):
          if t==[]:
            s=(0,0)
            if a[i]==self.unk_token_id:
              j=[[-1]]+[t for t in a2b[0:i] if t>[]]
              k=[t for t in a2b[i+1:] if t>[]]+[[len(b+c)]]
              s=(j[-1][-1]+1,k[0][0])
          elif t[-1]<len(b):
            s=(t[0],t[-1]+1)
          else:
            s=(t[0]-len(b),t[-1]-len(b)+1)
          x.append(s)
        w.append(list(x))
      v["offset_mapping"]=w[0] if type(text)==str else w
    return v

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    run_mode = 't5-base'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.pn.220204'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    elif 'mbart' in model_name:
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = allocate_data_to_device(model, DEVICE)

    train_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt')
    test_datasets = get_datasets('/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-token.txt')
    train_sentences, train_labels = train_datasets['sentences'], train_datasets['positive_labels']
    train_sentences, train_labels, dev_sentences, dev_labels = train_sentences[:1000], train_labels[:1000], train_sentences[1000:], train_labels[1000:]
    test_sentences, test_labels = test_datasets['sentences'], test_datasets['positive_labels']
    dev_negative_labels = train_datasets['negative_labels'][1000:]
    test_negative_labels = test_datasets['negative_labels']


    output_layer = allocate_data_to_device(torch.nn.Linear(model.config.hidden_size, 1), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=2e-5, weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for s, l in zip(train_sentences, train_labels):
            inputs = tokenizer(s, return_tensors='pt')
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = output_layer(h1).squeeze(2)
            o1 = activation_function(o1) if with_activation_function else o1

            loss = loss_func(o1, torch.as_tensor([index], dtype=torch.long, device=DEVICE))
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
        for s, l, neg_l in zip(dev_sentences, dev_labels, dev_negative_labels):
            inputs = tokenizer(s, return_tensors='pt')
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            neg_index = get_label(inputs.data['offset_mapping'], neg_l) if 'tohoku' in model_name else get_label(encodings.offsets, neg_l)
            offset = inputs.data['offset_mapping']

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = output_layer(h1).squeeze(2)
            o1 = activation_function(o1) if with_activation_function else o1

            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            loss = loss_func(o1, torch.as_tensor([index], dtype=torch.long, device=DEVICE))
            dev_total_loss.append(loss.item())

            if o1[0][index] > o1[0][neg_index]:
                dev_tt += 1
            else:
                dev_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {l}, {torch.argmax(o1)}, {index}')

        dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
        dev_acc = dev_tt / (dev_tt + dev_ff)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for s, l in zip(test_sentences, test_labels):
            inputs = tokenizer(s, return_tensors='pt')
            encodings = None if 'tohoku' in model_name else inputs.encodings[0]
            index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            neg_index = get_label(inputs.data['offset_mapping'], neg_l) if 'tohoku' in model_name else get_label(encodings.offsets, neg_l)

            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.data.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            o1 = output_layer(h1).squeeze(2)
            o1 = activation_function(o1) if with_activation_function else o1

            # index = get_label(inputs.data['offset_mapping'], l) if 'tohoku' in model_name else get_label(encodings.offsets, l)
            loss = loss_func(o1, torch.as_tensor([index], dtype=torch.long, device=DEVICE))
            test_total_loss.append(loss.item())

            if o1[0][index] > o1[0][neg_index]:
                test_tt += 1
            else:
                test_ff += 1

            if with_print_logits:
                print(f'{o1.item()}, {l}, {torch.argmax(o1)}, {index}')

        test_total_loss = sum(test_total_loss) / len(test_total_loss)
        test_acc = test_tt / (test_tt + test_ff)

        print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, dev_acc: {dev_acc}, test_loss: {test_total_loss}, test_acc: {test_acc}')
        result_lines.append([e, train_total_loss, dev_total_loss, dev_acc, test_total_loss, test_acc])

    with Path(f'./result.pn.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')

if __name__ == '__main__':
    train_model()


'''
rinna-search



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