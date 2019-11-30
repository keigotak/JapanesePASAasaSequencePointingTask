# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1)


class Model(nn.Module):
    def vocab_zero_padding(self, indexes, embeds):
        # print(indexes)
        indexes = indexes.long().cpu()
        for bi, index_list in enumerate(indexes):
            if self.vocab_padding_idx in index_list:
                embeds[bi][np.where(index_list == self.vocab_padding_idx)[0][0]:] = 0.0
        return embeds

    def vocab_zero_padding_bert(self, indexes, embeds):
        for bi, index_list in enumerate(indexes):
            if self.vocab_padding_idx in index_list:
                embeds[bi][np.where(index_list == self.vocab_padding_idx)[0][0]:] = 0.0
        return embeds

    def vocab_zero_padding_bert_bccwj(self, indexes, embeds):
        for bi, index_list in enumerate(indexes):
            if self.vocab_padding_idx in index_list:
                embeds[bi][np.where(index_list == self.vocab_padding_idx)[0][0]:] = 0.0
        return embeds


    def vocab_zero_padding_elmo(self, words, embeds):
        pad_word = '[PAD]'
        for bi, word_list in enumerate(words):
            if pad_word in word_list:
                embeds[bi][np.where(word_list == pad_word)[0][0]:] = 0.0
        return embeds

    @staticmethod
    def _reverse_tensor(tensor):
        return tensor[:, range(tensor.shape[1]-1, -1, -1), :]


if __name__ == "__main__":
    def _reverse_tensor(tensor):
        reversed_np = np.flip(tensor.cpu().detach().numpy(), 1).copy()
        reversed_tensor1 = torch.from_numpy(reversed_np)
        reversed_tensor2 = tensor[:, torch.arange(tensor.shape[1]-1, -1, -1), :]
        return reversed_tensor1, reversed_tensor2
    test_tensor = torch.Tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
    ret1, ret2 = _reverse_tensor(test_tensor)
    print(test_tensor)
    print(ret1)
    print(ret2)