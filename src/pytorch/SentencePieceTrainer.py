from pathlib import Path
import io

import sentencepiece as spm
vocab_size = 3754

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

train_datasets = get_datasets(
    '/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/train-token.txt')
test_datasets = get_datasets(
    '/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-token.txt')
train_sentences, train_labels = train_datasets['sentences'], train_datasets['positive_labels']
train_sentences, train_labels, dev_sentences, dev_labels = train_sentences[:1000], train_labels[:1000], train_sentences[1000:], train_labels[1000:]
test_sentences, test_labels = test_datasets['sentences'], test_datasets['positive_labels']
dev_negative_labels = train_datasets['negative_labels'][1000:]
test_negative_labels = test_datasets['negative_labels']

with Path('../../data/WSC.train.txt').open('w') as f:
    for train_sentence in train_sentences:
        f.write(f'{train_sentence}\n')

spm.SentencePieceTrainer.train(input='../../data/WSC.train.txt', model_prefix='m.220422', vocab_size=vocab_size)

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.220422.model')

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
print(sp.decode_ids([209, 31, 9, 375, 586]))
