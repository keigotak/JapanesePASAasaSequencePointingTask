from pathlib import Path
import numpy as np
import torch
from elmoformanylangs import Embedder


class ElmoModel:
    def __init__(self, device='cpu'):
        root_path = Path('../../data/elmo')
        self.embedding = Embedder(root_path)
        # self.embedding.use_cuda = False
        # if device != 'cpu':
        #     self.embedding.use_cuda = True
        self.embedding_dim = self.embedding.model.output_dim * 2

    def get_word_embedding(self, batch_words):
        return torch.tensor(self.embedding.sents2elmo(batch_words))

    def get_pred_embedding(self, batch_arg_embedding, batch_word_pos, word_pos_pred_idx):
        preds = []
        for arg, word_pos in zip(batch_arg_embedding, batch_word_pos):
            pred_pos = int((word_pos == word_pos_pred_idx).nonzero())
            pred_vec = arg[pred_pos].tolist()
            preds.append([pred_vec for _ in range(len(arg))])
        preds = torch.tensor(preds)
        return preds

    def state_dict(self):
        return self.embedding.state_dict()


if __name__ == "__main__":
    model = ElmoModel('cpu')
    sentences = np.array([["猫", "が", "好き", "です", "。"], ["私", "の", "父", "は", "カモ", "です", "。"], ["友人", "は", "ウサギ", "が", "好き", "です", "。"]])
    pos = [[3, 2, 1, 0, 1], [5, 4, 3, 2, 1, 0, 1], [5, 4, 3, 2, 1, 0, 1]]
    pred_idx = 0
    ret = model.get_word_embedding(sentences)
    print(ret)
    ret = model.get_pred_embedding(ret, pos, pred_idx)
    print(ret)
    ret = model.state_dict()
    pass
