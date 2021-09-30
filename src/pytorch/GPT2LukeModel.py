# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Model import Model
from BertWithJumanModel import BertWithJumanModel
import MeCab

import sys
import os
sys.path.append(os.pardir)
from utils.GetCollocatedSentences import GetNextSentences, GetPreviousSentences
from sentence_transformers import SentenceTransformer


class GPT2LukeModel(Model):
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
                 train_embedding=True,
                 with_bccwj=False):
        super(GPT2LukeModel, self).__init__()
        torch.manual_seed(seed)

        self.device = device
        if device != 'cpu':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.vocab_size = None
        self.embedding_dim = None
        self.word_embeddings = None
        self.word_embeddings = GPT2Model(device=device, trainable=train_embedding)
        self.embedding_dim = self.word_embeddings.embedding_dim
        self.vocab_padding_idx = self.word_embeddings.get_padding_idx()

        self.pos_embedding_dim = pos_embedding_dim
        self.word_pos_embedding = nn.Embedding(word_pos_size, self.pos_embedding_dim, padding_idx=word_pos_padding_idx)
        self.ku_pos_embedding = nn.Embedding(ku_pos_size, self.pos_embedding_dim, padding_idx=ku_pos_padding_idx)
        self.mode_embedding_dim = mode_embedding_dim
        self.mode_embedding = nn.Embedding(mode_size, self.mode_embedding_dim, padding_idx=mode_padding_idx)

        self.word_pos_pred_idx = word_pos_pred_idx

        self.with_pos_embedding = True
        if self.with_pos_embedding:
            self.hidden_size = 2 * self.embedding_dim + 2 * self.pos_embedding_dim + self.mode_embedding_dim
        else:
            self.hidden_size = 2 * self.embedding_dim + self.mode_embedding_dim

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

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

        self.with_collocated_sentence = None
        if self.with_collocated_sentence == 'next':
            ns = GetNextSentences(with_bccwj=with_bccwj)
            self.collocated_sentences = {key: ns.get_collocated_sentences(key) for key in ['train', 'dev', 'test']}
            self.mecab = MeCab.Tagger("-Ochasen")
        elif self.with_collocated_sentence == 'prev':
            ps = GetPreviousSentences(with_bccwj=with_bccwj)
            self.collocated_sentences = {key: ps.get_collocated_sentences(key) for key in ['train', 'dev', 'test']}
            self.mecab = MeCab.Tagger("-Ochasen")

        self.hidden2sentence = nn.Linear(self.embedding_dim, self.word_embeddings.max_seq_length + 1)

    def forward(self, arg, pred, word_pos, ku_pos, mode):
        sentence_length = len(arg[0])
        appended_args = []
        case_token_positions = []

        for i, words in enumerate(arg):
            words = [word for word in words if word != "[PAD]"]
            words = words + ['[SEP]', pred[i][0], '[GA]', '[WO]', '[NI]']
            appended_args.append(words)
            case_token_positions.append({'[GA]': words.index('[GA]'), '[WO]': words.index('[WO]'), '[NI]': words.index('[NI]')})
        arg = appended_args

        # output shape: Batch, Sentence_length, word_embed_size
        arg_rets = self.word_embeddings.get_word_embedding(arg)
        arg_embeds = self.vocab_zero_padding_bert(arg_rets["id"], arg_rets["embedding"])
        arg_embeds = self.append_zero_tensors(modified_size=sentence_length, current_tensor=arg_embeds)

        batch_case_pointing_logits = []
        for case_token in ['[GA]', '[WO]', '[NI]']:
            case_pointing_logits = []
            for i, pos in enumerate(case_token_positions):
                case_token_embedding = arg_embeds[i][pos[case_token]]
                case_pointing_logit = self.hidden2sentence(case_token_embedding)
                case_pointing_logits.append(case_pointing_logit)
            batch_case_pointing_logits.append(torch.stack(case_pointing_logits))
        batch_case_pointing_logits = torch.stack(batch_case_pointing_logits)



        # pred_rets = self.word_embeddings.get_pred_embedding(arg_embeds, arg_rets["token"], word_pos, self.word_pos_pred_idx)
        # pred_embeds = pred_rets["embedding"]
        # pred_embeds = self.append_zero_tensors(modified_size=sentence_length, current_tensor=pred_embeds)
        #
        # # output shape: Batch, Sentence_length, pos_embed_size
        # word_pos_embeds = self.word_pos_embedding(word_pos)
        # ku_pos_embeds = self.ku_pos_embedding(ku_pos)
        #
        # # output shape: Batch, Sentence_length, mode_embed_size
        # mode_embeds = self.mode_embedding(mode)
        #
        # if self.with_pos_embedding:
        #     # output shape: Batch, Sentence_length, 2 * word_embed_size + 2 * pos_embed_size
        #     concatten_embeds = torch.cat((arg_embeds, pred_embeds, word_pos_embeds, ku_pos_embeds, mode_embeds), dim=2)
        # else:
        #     # output shape: Batch, Sentence_length, 2 * word_embed_size
        #     concatten_embeds = torch.cat((arg_embeds, pred_embeds, mode_embeds), dim=2)
        #
        # # output shape: Batch, Sentence_length, hidden_size
        # f_lstm1_out, _ = self.f_lstm1(concatten_embeds)
        # if self.dropout_ratio > 0:
        #     f_lstm1_out = self.dropout(f_lstm1_out)
        #
        # # output shape: Batch, Sentence_length, hidden_size
        # residual_input2 = f_lstm1_out + concatten_embeds
        # residual_input2 = self._reverse_tensor(residual_input2)
        # b_lstm1_out, _ = self.b_lstm1(residual_input2)
        # if self.dropout_ratio > 0:
        #     b_lstm1_out = self.dropout(b_lstm1_out)
        # b_lstm1_out = self._reverse_tensor(b_lstm1_out)
        #
        # # output shape: Batch, Sentence_length, hidden_size
        # residual_input3 = b_lstm1_out + f_lstm1_out
        # f_lstm2_out, _ = self.f_lstm2(residual_input3)
        # if self.dropout_ratio > 0:
        #     f_lstm2_out = self.dropout(f_lstm2_out)
        #
        # # output shape: Batch, Sentence_length, hidden_size
        # residual_input4 = f_lstm2_out + b_lstm1_out
        # residual_input4 = self._reverse_tensor(residual_input4)
        # lstm_out, _ = self.b_lstm2(residual_input4)
        # if self.dropout_ratio > 0:
        #     lstm_out = self.dropout(lstm_out)
        # lstm_out = self._reverse_tensor(lstm_out)
        #
        # if self.with_sentence_transformer:
        #     tag_space = self.hidden2tag(torch.cat((lstm_out, torch.stack(collocated_sentence_embeddings).unsqueeze(1).repeat(1, lstm_out.shape[1], 1).to(lstm_out.device)), dim=2))
        # else:
        #     tag_space = self.hidden2tag(lstm_out)

        return batch_case_pointing_logits

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
    model = GPT2LukeModel(train_embedding=False)
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))
    # model.load_weights('../../results/pasa-bertsl-20191207-150951/model-0/epoch14-f0.8620.h5')

    tensor = torch.FloatTensor(1, 187, 768)
    tensor.to(device='cpu')
    new_tensor = model.append_zero_tensors(264, tensor)
    print(new_tensor.shape)

    tensor = torch.FloatTensor(1, 264, 768)
    tensor.to(device='cpu')
    new_tensor = model.append_zero_tensors(264, tensor)
    print(sum((tensor == new_tensor).numpy().flatten().tolist()) == 264*768)