import os
from pathlib import Path
import transformers
import torch
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(0)
torch.manual_seed(0)
from transformers import T5TokenizerFast, T5Model

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.datasets import SentencesDataset
from torch.utils.data import DataLoader

def get_properties(mode):
    if mode == 'rinna-dev':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.triplet.dev', 100
    elif mode == 'rinna-test':
        return 'rinna/japanese-gpt2-medium', '../../results/wsc_sbert.rinna-japanese-gpt2-medium.triplet.test', 100
    elif mode == 'tohoku-bert-dev':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.triplet.dev', 100
    elif mode == 'tohoku-bert-test':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', '../../results/wsc_sbert.bert-base-japanese-whole-word-masking.triplet.test', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', '../../results/wsc_sbert.mbart-large-cc25.search.triplet', 100
    elif mode == 't5-base-dev':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.triplet.dev', 100
    elif mode == 't5-base-test':
        return 'megagonlabs/t5-base-japanese-web', '../../results/wsc_sbert.t5-base-japanese-web.triplet.test', 100
    elif mode =='rinna-roberta-best-dev':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.triplet.dev', 100
    elif mode == 'rinna-roberta-best-test':
        return 'rinna/japanese-roberta-base', '../../results/wsc_sbert.rinna-japanese-roberta-base.triplet.test', 100
    elif mode == 'nlp-waseda-roberta-base-japanese-dev':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.triplet.dev', 100
    elif mode == 'nlp-waseda-roberta-base-japanese-test':
        return 'nlp-waseda/roberta-base-japanese', '../../results/wsc_sbert.nlp-waseda-roberta-base-japanese.triplet.test', 100
    elif mode == 'rinna-japanese-gpt-1b-dev':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.triplet.dev', 100
    elif mode == 'rinna-japanese-gpt-1b-test':
        return 'rinna/japanese-gpt-1b', '../../results/wsc_sbert.rinna-japanese-gpt-1b.triplet.test', 100

def train_model():
    BATCH_SIZE = 16
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)

    run_mode = 'rinna-japanese-gpt-1b-test'
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.220302'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    transformer = models.Transformer(model_name)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling])
    model.to('cuda:0')

    triplet_reader = TripletReader("../../data/Winograd-Schema-Challenge-Ja-master/fixed")
    train_data = SentencesDataset(triplet_reader.get_examples(f'train-triplet.txt'), model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = TripletLoss(model=model, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=0.5)

    if 'test' in run_mode:
        test_data = SentencesDataset(triplet_reader.get_examples(f'test-triplet.txt'), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
        test_anchor, test_positive, test_negative = [t.texts[0] for t in test_dataloader.dataset],\
                                                    [t.texts[1] for t in test_dataloader.dataset],\
                                                    [t.texts[2] for t in test_dataloader.dataset]
        test_evaluator = TripletEvaluator(test_anchor, test_positive, test_negative)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                     evaluator=test_evaluator,
                     epochs=NUM_EPOCHS,
                     warmup_steps=WARMUP_STEPS,
                     output_path=OUTPUT_PATH)
    else:
        dev_data = SentencesDataset(triplet_reader.get_examples(f'dev-triplet.txt'), model=model)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=BATCH_SIZE)
        dev_anchor, dev_positive, dev_negative = [t.texts[0] for t in dev_dataloader.dataset], [t.texts[1] for t in dev_dataloader.dataset], [t.texts[2] for t in dev_dataloader.dataset]
        dev_evaluator = TripletEvaluator(dev_anchor, dev_positive, dev_negative)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                     evaluator=dev_evaluator,
                     epochs=NUM_EPOCHS,
                     warmup_steps=WARMUP_STEPS,
                     output_path=OUTPUT_PATH)






from sklearn.metrics.pairwise import cosine_similarity
def evaluate(test_set, test_vectors, model):
    test_data = SentencesDataset(triplet_reader.get_examples(f'test-triplet.txt'), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    test_anchor, test_positive, test_negative = [t.texts[0] for t in test_dataloader.dataset],\
                                                [t.texts[1] for t in test_dataloader.dataset],\
                                                [t.texts[2] for t in test_dataloader.dataset]
    test_evaluator = TripletEvaluator(test_anchor, test_positive, test_negative)




if __name__ == '__main__':
    train_model()