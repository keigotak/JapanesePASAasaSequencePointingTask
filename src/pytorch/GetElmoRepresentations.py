import os
import sys
sys.path.append('../')
import argparse
from pathlib import Path
from ElmoModel import ElmoModel
from utils.Datasets import get_datasets_in_sentences, get_datasets_in_sentences_test
import pickle
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='ELMo representation')
    parser.add_argument('--with_bccwj', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    arguments = parser.parse_args()
    with_bccwj = arguments.with_bccwj

    if arguments.device != 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.device
    device = torch.device("cpu")
    if arguments.device != 'cpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN = "train2"
    DEV = "dev"
    TEST = "test"
    train_label, train_args, train_preds, train_prop, train_vocab, train_word_pos, train_ku_pos, train_modes, train_word_pos_id, train_ku_pos_id, train_modes_id = get_datasets_in_sentences(
        TRAIN, with_bccwj=with_bccwj, with_bert=False)
    dev_label, dev_args, dev_preds, dev_prop, dev_vocab, dev_word_pos, dev_ku_pos, dev_modes, dev_word_pos_id, dev_ku_pos_id, dev_modes_id = get_datasets_in_sentences(
        DEV, with_bccwj=arguments.with_bccwj, with_bert=False)
    test_label, test_args, test_preds, test_prop, test_vocab, test_word_pos, test_ku_pos, test_modes, test_word_pos_id, test_ku_pos_id, test_modes_id = get_datasets_in_sentences(
        TEST, with_bccwj=arguments.with_bccwj, with_bert=False)

    with torch.no_grad():
        model = ElmoModel(device=device, elmo_with="allennlp")
        embeddings = {}
        for args in [train_args, dev_args, test_args]:
            sents = [''.join(arg.tolist()) for arg in args.tolist()]
            for sent, words in zip(sents, args):
                if sent not in embeddings.keys():
                    embeddings[sent] = [vecs.tolist() for vecs in model.get_word_embedding(words)]
    with Path('../../data/elmo.pkl').open('wb') as f:
        pickle.dump(embeddings, f)


def load():
    with Path('../../data/elmo.pkl').open('rb') as f:
        embeddings = pickle.load(f)
        cnt = 0
    for k, v in embeddings.items():
        print('{}'.format(k))
        print('{}'.format(v))
        if cnt > 10:
            break


if __name__ == '__main__':
    # main()
    load()

