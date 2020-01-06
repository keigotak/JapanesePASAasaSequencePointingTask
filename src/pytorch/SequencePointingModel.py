# -*- coding: utf-8 -*-
from BiGRUModel import BiGRUModel
import torch
import torch.nn as nn


class SequencePointingModel(BiGRUModel):
    def __init__(self,
                 word_pos_size=20,
                 ku_pos_size=20,
                 mode_size=2,
                 target_size=1,
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
        super().__init__(word_pos_size=word_pos_size,
                         ku_pos_size=ku_pos_size,
                         mode_size=mode_size,
                         target_size=target_size,
                         device=device,
                         vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         pos_embedding_dim=pos_embedding_dim,
                         mode_embedding_dim=mode_embedding_dim,
                         word_pos_pred_idx=word_pos_pred_idx,
                         vocab_padding_idx=vocab_padding_idx,
                         word_pos_padding_idx=word_pos_padding_idx,
                         ku_pos_padding_idx=ku_pos_padding_idx,
                         mode_padding_idx=mode_padding_idx,
                         num_layers=num_layers,
                         dropout_ratio=dropout_ratio,
                         seed=seed,
                         bidirectional=bidirectional,
                         return_seq=return_seq,
                         pretrained_embedding=pretrained_embedding,
                         batch_first=batch_first,
                         norm=norm,
                         continue_seq=continue_seq,
                         pretrained_weights=pretrained_weights,
                         with_train_embedding=with_train_embedding,
                         add_null_word=add_null_word)

        self.with_linear = False
        self.l1_size = 0
        self.output_ga = nn.Linear(self.hidden_size, self.target_size)
        self.output_ni = nn.Linear(self.hidden_size, self.target_size)
        self.output_wo = nn.Linear(self.hidden_size, self.target_size)


    def forward(self, arg, pred, word_pos, ku_pos, mode):
        lstm_out = super().forward(arg, pred, word_pos, ku_pos, mode)

        # output shape: Batch, Sentence_length, 1
        output_ga = self.output_ga(lstm_out)
        output_ni = self.output_ni(lstm_out)
        output_wo = self.output_wo(lstm_out)

        if self.add_null_word:
            # output shape: Batch, Sentence_length
            output_ga = output_ga.view(output_ga.shape[0], output_ga.shape[1])
            output_ni = output_ni.view(output_ni.shape[0], output_ni.shape[1])
            output_wo = output_wo.view(output_wo.shape[0], output_wo.shape[1])
        else:
            # output shape: Batch, Sentence_length+1
            output_ga = torch.cat((torch.zeros(output_ga.shape[0], 1).float().to(output_ga.device), output_ga.view(output_ga.shape[0], output_ga.shape[1])), dim=1)
            output_ni = torch.cat((torch.zeros(output_ni.shape[0], 1).float().to(output_ni.device), output_ni.view(output_ni.shape[0], output_ni.shape[1])), dim=1)
            output_wo = torch.cat((torch.zeros(output_wo.shape[0], 1).float().to(output_wo.device), output_wo.view(output_wo.shape[0], output_wo.shape[1])), dim=1)

        return output_ga, output_ni, output_wo


if __name__ == "__main__":
    model = SequencePointingModel(pretrained_embedding='bert')
    for k, v in model.named_parameters():
        print("{}, {}, {}".format(v.requires_grad, v.size(), k))
