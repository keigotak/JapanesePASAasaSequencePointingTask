from pathlib import Path
import logging
logging.basicConfig(level=logging.FATAL,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

import numpy as np
import torch

from elmoformanylangs import Embedder

from typing import Dict, List
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.elmo import Elmo, batch_to_ids

def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids

# def text_to_instance(word_list, label):
#     tokens = [Token(word) for word in word_list]
#     word_sentence_field = TextField(tokens, {"tokens":SingleIdTokenIndexer()})
#     char_sentence_field = TextField(tokens, {'char_tokens': char_indexer})
#     fields = {"tokens":word_sentence_field, 'char_tokens': char_sentence_field}
#     if label is not None:
#         label_field = LabelField(label, skip_indexing=True)
#         fields["label"] = label_field
#     return Instance(fields)

class CustomELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    We allow to add optional additional special tokens with designated
    character ids with ``tokens_to_add``.
    """
    max_word_length = 50

    def __init__(self, tokens_to_add: Dict[str, int] = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

        # setting special token
        self.beginning_of_sentence_character = self.tokens_to_add['<bos>']  # <begin sentence>
        self.end_of_sentence_character = self.tokens_to_add['<eos>']  # <end sentence>
        self.beginning_of_word_character = self.tokens_to_add['<bow>']  # <begin word>
        self.end_of_word_character = self.tokens_to_add['<eow>']  # <end word>
        self.padding_character = self.tokens_to_add['<pad>']  # <padding>
        self.oov_character = self.tokens_to_add['<oov>']

        self.max_word_length = 50

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars

        self.beginning_of_sentence_characters = _make_bos_eos(
            self.beginning_of_sentence_character,
            self.padding_character,
            self.beginning_of_word_character,
            self.end_of_word_character,
            self.max_word_length,
        )
        self.end_of_sentence_characters = _make_bos_eos(
            self.end_of_sentence_character,
            self.padding_character,
            self.beginning_of_word_character,
            self.end_of_word_character,
            self.max_word_length,
        )

        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = self.end_of_word_character
        elif word == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = self.end_of_sentence_characters
        else:
            word = word[: (self.max_word_length - 2)]
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            for k, char in enumerate(word, start=1):
                char_ids[k] = self.tokens_to_add[char] if char in self.tokens_to_add else self.oov_character
            char_ids[len(word) + 1] = self.end_of_word_character

        # +1 one for masking
        # return [c + 1 for c in char_ids]
        return char_ids

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


@TokenIndexer.register("custom_elmo_characters")
class CustomELMoTokenCharactersIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.
    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    tokens_to_add : ``Dict[str, int]``, optional (default=``None``)
        If not None, then provides a mapping of special tokens to character
        ids. When using pre-trained models, then the character id must be
        less then 261, and we recommend using un-used ids (e.g. 1-32).
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: str = "elmo_characters",
        tokens_to_add: Dict[str, int] = None,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._mapper = CustomELMoCharacterMapper(tokens_to_add)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[List[int]]]:
        # TODO(brendanr): Retain the token to index mappings in the vocabulary and remove this

        # https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/wordpiece_indexer.py#L113

        texts = [token for token in tokens]

        if any(text is None for text in texts):
            raise ConfigurationError(
                "ELMoTokenCharactersIndexer needs a tokenizer " "that retains text"
            )
        return {index_name: [self._mapper.convert_word_to_char_ids(text) for text in texts]}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {}

    @staticmethod
    def _default_value_for_padding():
        return [0] * CustomELMoCharacterMapper.max_word_length

    @overrides
    def as_padded_tensor(
        self,
        tokens: Dict[str, List[List[int]]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val, desired_num_tokens[key], default_value=self._default_value_for_padding
                )
            )
            for key, val in tokens.items()
        }



class ElmoModel:
    def __init__(self, device='cpu', elmo_with="allennlp"):
        self.elmo_with = elmo_with
        if self.elmo_with == "allennlp":
            root_path = Path('../../data/elmo/converted')
            option_file = root_path / 'allennlp_config.json'
            weight_file = root_path / 'allennlp_elmo.hdf5'
            self.num_output_representations = 2
            self.embedding = Elmo(option_file, weight_file, num_output_representations=self.num_output_representations)
            self.embedding_dim = 1024 * self.num_output_representations

            with open('../../data/elmo/converted/char_for_allennlp.dic') as f:
                char_dic = {line.split('\t')[0]: int(line.split('\t')[1].strip('\n')) for line in f}

            self.char_indexer = CustomELMoTokenCharactersIndexer(tokens_to_add=char_dic)
            self.vocab = Vocabulary()
        else:
            root_path = Path('../../data/elmo')
            self.embedding = Embedder(root_path)
            if device != 'cpu':
                self.embedding.use_cuda = True
                self.embedding.model.use_cuda = True
                self.embedding.model.encoder.use_cuda = True
                self.embedding.model.token_embedder.use_cuda = True
                self.embedding.model.to(device)
            self.embedding_dim = self.embedding.config['encoder']['projection_dim'] * 2

    def get_word_embedding(self, batch_words):
        rets = []
        if self.elmo_with == "allennlp":
            for words in batch_words:
                indexes = self.char_indexer.tokens_to_indices(words, self.vocab, "tokens")
                embedding = self.embedding(torch.Tensor([indexes["tokens"]]).long())
                rets.append(torch.cat((embedding['elmo_representations'][0], embedding['elmo_representations'][1]), dim=2)[0])
        else:
            rets = self.embedding.sents2elmo(batch_words)
            rets = torch.tensor(rets)
        return rets

    def get_pred_embedding(self, batch_arg_embedding, batch_word_pos, word_pos_pred_idx):
        preds = []
        print('{}, {}, {}'.format(len(batch_arg_embedding), len(batch_arg_embedding[0]), len(batch_arg_embedding[0][0])))
        print('{}, {}, {}'.format(type(batch_arg_embedding), type(batch_arg_embedding[0]), type(batch_arg_embedding[0][0])))
        for arg, word_pos in zip(batch_arg_embedding, batch_word_pos):
            if type(word_pos) != list:
                word_pos = word_pos.tolist()
            pred_pos = word_pos.index(word_pos_pred_idx)
            pred_vec = arg[pred_pos].tolist()
            preds.append([pred_vec for _ in range(len(arg))])
        preds = torch.tensor(preds)
        return preds

    def state_dict(self):
        return self.embedding.state_dict()


if __name__ == "__main__":
    model = ElmoModel(device='cpu', elmo_with="allennlp")
    sentences = np.array([["猫", "が", "好き", "です", "。", "　", "[PAD]"], ["私", "の", "父", "は", "カモ", "です", "。"], ["友人", "は", "ウサギ", "が", "好き", "です", "。"]])
    pos = [[3, 2, 1, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 1], [5, 4, 3, 2, 1, 0, 1]]
    pred_idx = 0
    ret = model.get_word_embedding(sentences)
    print(ret)
    ret = model.get_pred_embedding(ret, pos, pred_idx)
    print(ret)
    pass

