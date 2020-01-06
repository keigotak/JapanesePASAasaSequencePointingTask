# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Model import Model
from ElmoModel import ElmoModel
from BertWithJumanModel import BertWithJumanModel
from collections import OrderedDict


class BiGRUModel(Model):
    def __init__(self,
                 word_pos_size=20,
                 ku_pos_size=20,
                 mode_size=2,
                 target_size=4,
                 device='cpu',
                 vocab_size=-1,
                 embedding_dim=32,
                 pos_embedding_dim=10,
                 mode_embedding_dim=2,
                 word_pos_pred_idx=0,
                 vocab_padding_idx=-1,
                 word_pos_padding_idx=-1,
                 ku_pos_padding_idx=-1,
                 mode_padding_idx=-1,
                 vocab_null_idx=-1,
                 vocab_unk_idx=-1,
                 num_layers=1,
                 dropout_ratio=.1,
                 seed=1,
                 bidirectional=True,
                 return_seq=False,
                 pretrained_embedding='default',
                 batch_first=True,
                 norm='none',
                 continue_seq=False,
                 pretrained_weights=None,
                 with_train_embedding=True,
                 add_null_word=True):
        super(BiGRUModel, self).__init__()
        self.device = device
        self.pretrained_embedding = pretrained_embedding
        self.with_train_embedding = with_train_embedding

        torch.manual_seed(seed)
        # if you are suing GPU
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
        self.vocab_null_idx = vocab_null_idx
        self.vocab_unk_idx = vocab_unk_idx
        if pretrained_embedding == 'default':
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.word_embeddings.padding_idx = vocab_padding_idx
        elif pretrained_embedding == 'bert':
            self.word_embeddings = BertWithJumanModel(device=device, trainable=with_train_embedding)
            self.embedding_dim = self.word_embeddings.embedding_dim
            self.vocab_padding_idx = self.word_embeddings.get_padding_idx()
        elif pretrained_embedding == 'elmo':
            self.word_embeddings = ElmoModel(device=device)
            self.embedding_dim = self.word_embeddings.embedding_dim
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_weights)
            self.word_embeddings.padding_idx = vocab_padding_idx
            self.word_embeddings.weight.requires_grad = with_train_embedding
            self.vocab_size = self.word_embeddings.num_embeddings
            self.embedding_dim = self.word_embeddings.embedding_dim

        self.pos_embedding_dim = pos_embedding_dim
        self.word_pos_embedding = nn.Embedding(word_pos_size, self.pos_embedding_dim, padding_idx=word_pos_padding_idx)
        self.ku_pos_embedding = nn.Embedding(ku_pos_size, self.pos_embedding_dim, padding_idx=ku_pos_padding_idx)
        self.mode_embedding_dim = mode_embedding_dim
        self.mode_embedding = nn.Embedding(mode_size, self.mode_embedding_dim, padding_idx=mode_padding_idx)

        self.word_pos_pred_idx = word_pos_pred_idx
        self.word_pos_padding_idx = word_pos_padding_idx

        self.hidden_size = 2 * self.embedding_dim + 2 * self.pos_embedding_dim + self.mode_embedding_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.norm = norm
        self.return_seq = return_seq
        self.continue_seq = continue_seq
        self.add_null_word = add_null_word
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

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, arg, pred, word_pos, ku_pos, mode):
        # output shape: Batch, Sentence_length, word_embed_size
        if self.pretrained_embedding == 'bert':
            arg_rets = self.word_embeddings.get_word_embedding(arg)
            arg_embeds = self.vocab_zero_padding_bert(arg_rets["id"], arg_rets["embedding"])
            pred_rets = self.word_embeddings.get_pred_embedding(arg_embeds,
                                                                arg_rets["token"],
                                                                word_pos,
                                                                self.word_pos_pred_idx)
            pred_embeds = pred_rets["embedding"]
        elif self.pretrained_embedding == 'elmo':
            arg_rets = self.word_embeddings.get_word_embedding(arg)
            arg_embeds = self.vocab_zero_padding_elmo(arg, arg_rets)
            pred_rets = self.word_embeddings.get_pred_embedding(arg_embeds, word_pos, self.word_pos_pred_idx)
            pred_embeds = pred_rets
        else:
            arg_embeds = self.word_embeddings(arg)
            arg_embeds = self.vocab_zero_padding(arg, arg_embeds)
            pred_embeds = self.word_embeddings(pred)
            pred_embeds = self.vocab_zero_padding(pred, pred_embeds)

        # output shape: Batch, Sentence_length, pos_embed_size
        word_pos_embeds = self.word_pos_embedding(word_pos)
        ku_pos_embeds = self.ku_pos_embedding(ku_pos)

        # output shape: Batch, Sentence_length, mode_embed_size
        mode_embeds = self.mode_embedding(mode)

        # output shape: Batch, Sentence_length, 2 * word_embed_size + 2 * pos_embed_size
        if self.pretrained_embedding == 'bert': # bertの場合はsubword化されたものなので後でdeviceに乗せる
            concatten_embeds = torch.empty(len(arg_embeds), len(arg_embeds[0]), self.hidden_size).to(self.device)
            for bi, (args, preds, words, kus, modes) in enumerate(
                zip(arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds)):
                for ii, (arg, pred, word, ku, mode) in enumerate(zip(args, preds, words, kus, modes)):
                    if str(self.device) != "cpu":
                        arg = arg.to(self.device)
                        pred = pred.to(self.device)
                    concatten_embeds[bi][ii] = torch.cat((arg, pred, word, ku, mode), dim=0).to(self.device)
        else:
            concatten_embeds = torch.cat((arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds), 2)

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

        return lstm_out

    def load_weights(self, path):
        path = str(path.resolve())
        if '.h5' in path:
            path = path.replace('.h5', '')
        state_dict = torch.load(path + '.h5', map_location='cpu')

        if self.pretrained_embedding == 'bert':
            state_dict_bert = torch.load(path + '_bert.h5', map_location='cpu')
            modified_state_dict_bert = OrderedDict()
            for k, v in state_dict_bert.items():
                modified_state_dict_bert['word_embeddings.model.' + k] = v
            state_dict.update(modified_state_dict_bert)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    model = BiGRUModel(pretrained_embedding='bert')
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))
