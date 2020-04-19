# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Model import Model
from BertWithJumanModel import BertWithJumanModel

from collections import OrderedDict


class BertSequencePointingModel(Model):
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
                 batch_first=True, continue_seq=False):
        super(BertSequencePointingModel, self).__init__()
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
        self.vocab_padding_idx = vocab_padding_idx
        self.word_embeddings = BertWithJumanModel(device=device)
        self.embedding_dim = self.word_embeddings.embedding_dim

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

        self.l1_size = 0
        self.output_ga = nn.Linear(self.hidden_size, self.target_size)
        self.output_ni = nn.Linear(self.hidden_size, self.target_size)
        self.output_wo = nn.Linear(self.hidden_size, self.target_size)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, arg, pred, word_pos, ku_pos, mode):
        # output shape: Batch, Sentence_length, word_embed_size
        arg_rets = self.word_embeddings.get_word_embedding(arg)
        arg_embeds = self.vocab_zero_padding_bert(arg_rets["id"], arg_rets["embedding"])
        arg_embeds = self.append_zero_tensors(modified_size=len(arg[1]), current_tensor=arg_embeds)
        pred_rets = self.word_embeddings.get_pred_embedding(arg_embeds, arg_rets["token"], word_pos, self.word_pos_pred_idx)
        pred_embeds = pred_rets["embedding"]
        pred_embeds = self.append_zero_tensors(modified_size=len(pred[1]), current_tensor=pred_embeds)

        # output shape: Batch, Sentence_length, pos_embed_size
        word_pos_embeds = self.word_pos_embedding(word_pos)
        ku_pos_embeds = self.ku_pos_embedding(ku_pos)

        # output shape: Batch, Sentence_length, mode_embed_size
        mode_embeds = self.mode_embedding(mode)

        # output shape: Batch, Sentence_length, 2 * word_embed_size + 2 * pos_embed_size
        concatten_embeds = torch.cat((arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds), dim=2)
        # concatten_embeds = torch.empty(len(arg_embeds), len(arg_embeds[0]), self.hidden_size).to(self.device)
        # for bi, (args, preds, words, kus, modes) in enumerate(zip(arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds)):
        #     for ii, (arg, pred, word, ku, mode) in enumerate(zip(args, preds, words, kus, modes)):
        #         if str(self.device) != "cpu":
        #             arg = arg.to(self.device)
        #             pred = pred.to(self.device)
        #         concatten_embeds[bi][ii] = torch.cat((arg, pred, word, ku, mode), dim=0).to(self.device)

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

        linear_ga1 = lstm_out
        linear_ni1 = lstm_out
        linear_wo1 = lstm_out

        # output shape: Batch, Sentence_length, 1
        output_ga = self.output_ga(linear_ga1)
        output_ni = self.output_ni(linear_ni1)
        output_wo = self.output_wo(linear_wo1)

        # output shape: Batch, Sentence_length+1
        output_ga = torch.cat((torch.zeros(output_ga.shape[0], 1).float().to(output_ga.device),
                               output_ga.view(output_ga.shape[0], output_ga.shape[1])), dim=1)
        output_ni = torch.cat((torch.zeros(output_ni.shape[0], 1).float().to(output_ni.device),
                               output_ni.view(output_ni.shape[0], output_ni.shape[1])), dim=1)
        output_wo = torch.cat((torch.zeros(output_wo.shape[0], 1).float().to(output_wo.device),
                               output_wo.view(output_wo.shape[0], output_wo.shape[1])), dim=1)

        return output_ga, output_ni, output_wo

    def load_weights(self, path):
        path = str(path.resolve())
        if '.h5' in path:
            path = path.replace('.h5', '')
        state_dict = torch.load(path + '.h5', map_location='cpu')
        # state_dict_bert = torch.load(path + '_bert.h5', map_location='cpu')
        # modified_state_dict_bert = OrderedDict()
        # for k, v in state_dict_bert.items():
        #     modified_state_dict_bert['word_embeddings.model.' + k] = v
        self.load_state_dict(state_dict)

    def append_zero_tensors(self, modified_size, current_tensor):
        batch_size = current_tensor.shape[0]
        current_size = current_tensor.shape[1]
        dim = current_tensor.shape[2]
        if current_size < modified_size:
            tensor = torch.zeros((batch_size, (modified_size - current_size), dim), dtype=current_tensor.dtype, device=current_tensor.device, requires_grad=False)
            modified_tensor = torch.cat([current_tensor, tensor], dim=1)
            return modified_tensor
        else:
            return current_tensor



if __name__ == "__main__":
    model = BertSequencePointingModel()

