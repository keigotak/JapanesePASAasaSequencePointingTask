# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from Model import Model
torch.manual_seed(1)


class CombinationModel(Model):
    def __init__(self, vocab_size, word_pos_size, ku_pos_size, mode_size,
                 l1_size, device,
                 l2_size=32, embedding_dim=32, pos_embedding_dim=10, mode_embedding_dim=2,
                 vocab_padding_idx=-1, word_pos_padding_idx=-1, ku_pos_padding_idx=-1, mode_padding_idx=-1,
                 vocab_null_idx=-1, vocab_unk_idx=-1,
                 num_layers=1, dropout_ratio=.1,
                 bidirectional=True, return_seq=False, pretrained_embedding='default',
                 batch_first=True, norm='none', continue_seq=False, pretrained_weights=None, add_null_word=True):
        super(CombinationModel, self).__init__()

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
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_weights)
            self.word_embeddings.padding_idx = vocab_padding_idx
            self.word_embeddings.weight.requires_grad = True
            self.vocab_size = self.word_embeddings.num_embeddings
            self.embedding_dim = self.word_embeddings.embedding_dim

        self.pos_embedding_dim = pos_embedding_dim
        self.word_pos_embedding = nn.Embedding(word_pos_size, self.pos_embedding_dim, padding_idx=word_pos_padding_idx)
        self.ku_pos_embedding = nn.Embedding(ku_pos_size, self.pos_embedding_dim, padding_idx=ku_pos_padding_idx)
        self.mode_embedding_dim = mode_embedding_dim
        self.mode_embedding = nn.Embedding(mode_size, self.mode_embedding_dim, padding_idx=mode_padding_idx)

        self.hidden_size = 2 * self.embedding_dim + 2 * self.pos_embedding_dim + self.mode_embedding_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.device = device
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.norm = norm
        self.return_seq = return_seq
        self.continue_seq = continue_seq
        self.add_null_word = add_null_word

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

        self.target_size_seq = 4
        self.hidden2tag = nn.Linear(self.hidden_size, self.target_size_seq)

        self.l1_size = l1_size
        self.l2_size = l2_size
        if self.add_null_word:
            self.l1_size += 1
        self.linear_ga1 = nn.Linear(self.l1_size, self.l2_size)
        self.linear_ni1 = nn.Linear(self.l1_size, self.l2_size)
        self.linear_wo1 = nn.Linear(self.l1_size, self.l2_size)

        self.target_size_ptr = l1_size
        self.output_ga = nn.Linear(self.l2_size, self.target_size_ptr)
        self.output_ni = nn.Linear(self.l2_size, self.target_size_ptr)
        self.output_wo = nn.Linear(self.l2_size, self.target_size_ptr)
        if not self.add_null_word:
            self.output_ga = nn.Linear(self.l2_size, self.target_size_ptr, bias=False)
            self.output_ni = nn.Linear(self.l2_size, self.target_size_ptr, bias=False)
            self.output_wo = nn.Linear(self.l2_size, self.target_size_ptr, bias=False)

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
        ret = {}

        # output shape: Batch, Sentence_length, word_embed_size
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

        # output shape: Batch, Sentence_length, target_size
        output_seq = torch.log_softmax(self.hidden2tag(lstm_out), dim=2)

        output_seq_ga = output_seq[:, :, 0]
        output_seq_ni = output_seq[:, :, 1]
        output_seq_wo = output_seq[:, :, 2]

        b_size = output_seq.shape[0]
        s_size = output_seq.shape[1]
        if s_size < self.l1_size:
            output_seq_ga = torch.cat((output_seq[:, :, 0], torch.zeros([b_size, (self.l1_size - s_size)]).to(self.device)), dim=1)
            output_seq_ni = torch.cat((output_seq[:, :, 1], torch.zeros([b_size, (self.l1_size - s_size)]).to(self.device)), dim=1)
            output_seq_wo = torch.cat((output_seq[:, :, 2], torch.zeros([b_size, (self.l1_size - s_size)]).to(self.device)), dim=1)

        # output shape: Batch, Sentence_length, l1_size
        linear_ga1 = torch.tanh(self.linear_ga1(output_seq_ga))
        linear_ni1 = torch.tanh(self.linear_ni1(output_seq_ni))
        linear_wo1 = torch.tanh(self.linear_wo1(output_seq_wo))

        # output shape: Batch, Sentence_length, l1_size
        if self.dropout_ratio > 0:
            linear_ga1 = self.dropout(linear_ga1)
            linear_ni1 = self.dropout(linear_ni1)
            linear_wo1 = self.dropout(linear_wo1)

        # output shape: Batch, Sentence_length, 1
        output_ptr_ga = self.output_ga(linear_ga1)
        output_ptr_ni = self.output_ni(linear_ni1)
        output_ptr_wo = self.output_wo(linear_wo1)

        if self.add_null_word:
            # output shape: Batch, Sentence_length
            output_ptr_ga = output_ptr_ga.view(output_ptr_ga.shape[0], output_ptr_ga.shape[1])
            output_ptr_ni = output_ptr_ni.view(output_ptr_ni.shape[0], output_ptr_ni.shape[1])
            output_ptr_wo = output_ptr_wo.view(output_ptr_wo.shape[0], output_ptr_wo.shape[1])
        else:
            # output shape: Batch, Sentence_length+1
            output_ptr_ga = torch.cat((torch.zeros(output_ptr_ga.shape[0], 1).float().to(output_ptr_ga.device), output_ptr_ga.view(output_ptr_ga.shape[0], output_ptr_ga.shape[1])), dim=1)
            output_ptr_ni = torch.cat((torch.zeros(output_ptr_ni.shape[0], 1).float().to(output_ptr_ni.device), output_ptr_ni.view(output_ptr_ni.shape[0], output_ptr_ni.shape[1])), dim=1)
            output_ptr_wo = torch.cat((torch.zeros(output_ptr_wo.shape[0], 1).float().to(output_ptr_wo.device), output_ptr_wo.view(output_ptr_wo.shape[0], output_ptr_wo.shape[1])), dim=1)

        output_ptr_ga = torch.log_softmax(output_ptr_ga, dim=1)
        output_ptr_ni = torch.log_softmax(output_ptr_ni, dim=1)
        output_ptr_wo = torch.log_softmax(output_ptr_wo, dim=1)

        return output_seq, output_ptr_ga, output_ptr_ni, output_ptr_wo
