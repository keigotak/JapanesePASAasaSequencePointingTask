# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Model import Model
from BertNictModel import BertNictModel

from collections import OrderedDict


class NictBertSequenceLabelingModel(Model):
    def __init__(self, word_pos_size=20,
                 ku_pos_size=20,
                 mode_size=2,
                 target_size=4,
                 device="cpu",
                 pos_embedding_dim=10,
                 mode_embedding_dim=2,
                 word_pos_pred_idx=0,
                 vocab_padding_idx=-1,
                 word_pos_padding_idx=-1,
                 ku_pos_padding_idx=-1,
                 mode_padding_idx=-1,
                 num_layers=1,
                 dropout_ratio=.1,
                 seed=1,
                 bidirectional=True, return_seq=False,
                 batch_first=True, continue_seq=False,
                 trainbert=False,
                 corpus='ntc',
                 with_db=False):
        super(NictBertSequenceLabelingModel, self).__init__()
        torch.manual_seed(seed)

        self.device = device
        if device != 'cpu':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.vocab_size = None
        self.embedding_dim = None
        self.word_embeddings = None
        self.word_embeddings = BertNictModel(device=device, trainable=trainbert, with_db=with_db, corpus=corpus)
        self.embedding_dim = self.word_embeddings.embedding_dim
        self.vocab_padding_idx = self.word_embeddings.get_padding_idx()

        self.pos_embedding_dim = pos_embedding_dim
        self.word_pos_embedding = nn.Embedding(word_pos_size, self.pos_embedding_dim, padding_idx=word_pos_padding_idx)
        self.ku_pos_embedding = nn.Embedding(ku_pos_size, self.pos_embedding_dim, padding_idx=ku_pos_padding_idx)
        self.mode_embedding_dim = mode_embedding_dim
        self.mode_embedding = nn.Embedding(mode_size, self.mode_embedding_dim, padding_idx=mode_padding_idx)

        self.word_pos_pred_idx = word_pos_pred_idx

        self.hidden_size = 2 * self.embedding_dim + 2 * self.pos_embedding_dim + self.mode_embedding_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.return_seq = return_seq
        self.continue_seq = continue_seq
        self.target_size = target_size
        self.mode = 'train'

        self.f_lstm1 = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bidirectional=False,
                              batch_first=self.batch_first)
        self.b_lstm1 = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bidirectional=False,
                              batch_first=self.batch_first)
        self.f_lstm2 = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bidirectional=False,
                              batch_first=self.batch_first)
        self.b_lstm2 = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bidirectional=False,
                              batch_first=self.batch_first)

        self.hidden2tag = nn.Linear(self.hidden_size, self.target_size)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, arg, pred, word_pos, ku_pos, mode, tag=None, epoch=None, index=None):
        # output shape: Batch, Sentence_length, word_embed_size
        if tag is not None:
            arg_rets = self.word_embeddings.get_word_embedding(arg, tag=tag, epoch=epoch, index=index)
        else:
            arg_rets = self.word_embeddings.get_word_embedding(arg)
        arg_embeds = arg_rets["embedding"]
        # arg_embeds = self.vocab_zero_padding_bert(arg_rets["id"], arg_rets["embedding"])
        pred_rets = self.word_embeddings.get_pred_embedding(arg_embeds, arg_rets["token"], word_pos, self.word_pos_pred_idx)
        pred_embeds = pred_rets["embedding"]

        # output shape: Batch, Sentence_length, pos_embed_size
        word_pos_embeds = self.word_pos_embedding(word_pos)
        ku_pos_embeds = self.ku_pos_embedding(ku_pos)

        # output shape: Batch, Sentence_length, mode_embed_size
        mode_embeds = self.mode_embedding(mode)

        # output shape: Batch, Sentence_length, 2 * word_embed_size + 2 * pos_embed_size
        concatten_embeds = torch.cat((arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds), dim=2)

        # output shape: Batch, Sentence_length, hidden_size
        f_lstm1_out, _ = self.f_lstm1(concatten_embeds)
        if self.dropout_ratio > 0:
            f_lstm1_out = self.dropout(f_lstm1_out)

        # output shape: Batch, Sentence_length, hidden_size
        residual_input2 = f_lstm1_out + concatten_embeds
        residual_input2 = self._reverse_tensor(residual_input2)
        b_lstm1_out, _ = self.b_lstm1(residual_input2)
        if self.dropout_ratio > 0:
            b_lstm1_out = self.dropout(b_lstm1_out)
        b_lstm1_out = self._reverse_tensor(b_lstm1_out)

        # output shape: Batch, Sentence_length, hidden_size
        residual_input3 = b_lstm1_out + f_lstm1_out
        f_lstm2_out, _ = self.f_lstm2(residual_input3)
        if self.dropout_ratio > 0:
            f_lstm2_out = self.dropout(f_lstm2_out)

        # output shape: Batch, Sentence_length, hidden_size
        residual_input4 = f_lstm2_out + b_lstm1_out
        residual_input4 = self._reverse_tensor(residual_input4)
        lstm_out, _ = self.b_lstm2(residual_input4)
        if self.dropout_ratio > 0:
            lstm_out = self.dropout(lstm_out)
        lstm_out = self._reverse_tensor(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space

    def load_weights(self, path):
        path = str(path.resolve())
        if '.h5' in path:
            path = path.replace('.h5', '')
        state_dict = torch.load(path + '.h5', map_location='cpu')
        state_dict_bert = torch.load(path + '_bert.h5', map_location='cpu')
        modified_state_dict_bert = OrderedDict()
        for k, v in state_dict_bert.items():
            modified_state_dict_bert['word_embeddings.model.' + k] = v
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    model = NictBertSequenceLabelingModel(trainbert=False)
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))
    model.forward(arg=[["ど", "う", "いたしまし", "て", "。"]],
                  pred=[["て", "て", "て", "て", "て"]],
                  word_pos=torch.Tensor([[3,2,1,0,11]]),
                  ku_pos=[[2,1,0,0,11]],
                  mode=[['','','','','']])
    # model.load_weights('../../results/pasa-bertsl-20191207-150951/model-0/epoch14-f0.8620.h5')
