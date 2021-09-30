import os
import sys
sys.path.append(os.pardir)
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import T5Tokenizer, AutoModelForCausalLM

import torch
from torch.utils.data import DataLoader

from utils.WSCDataset import WSCDatasetForShots
torch.manual_seed(0)

device = 'cuda:0'

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").to(device)

wsc_train_batcher = DataLoader(WSCDatasetForShots(mode='train'), batch_size=1, shuffle=True)
wsc_test_batcher = DataLoader(WSCDatasetForShots(mode='test'), batch_size=1, shuffle=False)

def generate_examples(question_sentences, targets, answers):
    messages = []
    for question_sentence, target, answer in zip(question_sentences, targets, answers):
        message = [
            '=====',
            '問題文：' + question_sentence,
            '質問：' + f'問題文中の照応詞「{target}」は何を指しますか？',
            '正解：' + f'{answer}'
        ]
        messages.extend(message)

    return messages

def generate_question(question_sentence, target):
    message = [
        '=====',
        '問題文：' + question_sentence[0],
        '質問：' + f'問題文中の照応詞「{target[0]}」は何を指しますか？',
        '解答：'
    ]
    return message

for num_shots in [0, 1]:
    headers = ['照応解析の問題です。次の文章を注意深く読み、問題文中の「」で括られた照応詞の先行詞を同定してください。']
    if num_shots != 0:
        question_sentences, targets, answers = [], [], []
        for i, inputs in enumerate(wsc_train_batcher):
            if i == num_shots:
                break
            question_sentences.append(inputs[0][0].replace(inputs[3][0], f'「{inputs[3][0]}」'))
            targets.append(inputs[3][0])
            answers.append(inputs[2][0])
        headers += generate_examples(question_sentences, targets, answers)

    output_file = Path(f'../../results/WSC-ja-{num_shots}.tsv')

    with output_file.open('w') as f:
        for inputs in wsc_test_batcher:
            question_sentences, targets = [], []
            question_sentences.append(inputs[0][0].replace(inputs[3][0], f'「{inputs[3][0]}」'))
            targets.append(inputs[3][0])

            message = headers + generate_question(question_sentences, targets)

            question = '\n'.join(message)
            input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
            outputs = model.generate(input_ids,
                                     do_sample=True,
                                     max_length=len(question) + 50,
                                     temperature=1,
                                     repetition_penalty=0.8,
                                     length_penalty=0.5,
                                     top_p=0.9)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(question):]
            with_print = False
            print(f'ID: {int(inputs[4])}')
            if with_print:
                print(question)
                print(output_text)
                print(f'選択肢: {inputs[1][0][0]}, {inputs[1][1][0]}')
                print(f'解答: {inputs[2][0]}')
                print('********************')
            text = [
                str(int(inputs[4])),
                inputs[2][0],
                '|'.join([inputs[1][0][0], inputs[1][1][0]]),
                output_text,
                '|'.join(message)
            ]
            f.write('\t'.join(text))
            f.write('\n')