from pathlib import Path
import pickle

class GetCollocatedSentences:
    def __init__(self, num_sentence=1):
        self.collocated_sentences = {}
        self.num_sentence = num_sentence

    def get_collocated_sentences(self, mode='all'):
        if mode == 'all':
            collocated_sentences = {}
            for mode in self.collocated_sentences.keys():
                collocated_sentences.update(self.collocated_sentences[mode])
        else:
            if self.num_sentence == 1:
                return self.collocated_sentences[mode]
            elif self.num_sentence == -1:
                collocated_multiple_sentences = {}
                for current_sentence, val in self.collocated_sentences[mode].items():
                    collocated_words = val.copy()
                    collocated_sentence = ''.join(collocated_words)
                    while len(collocated_words) < 1024:
                        if collocated_sentence in self.collocated_sentences[mode].keys():
                            collocated_words.extend(self.collocated_sentences[mode][collocated_sentence])
                            collocated_sentence = ''.join(self.collocated_sentences[mode][collocated_sentence])
                        else:
                            break
                    collocated_multiple_sentences[current_sentence] = collocated_words.copy()
                return collocated_multiple_sentences
            elif self.num_sentence == -2:
                collocated_multiple_sentences = {}
                for current_sentence, val in self.collocated_sentences[mode].items():
                    collocated_words = val.copy()
                    collocated_sentence = ''.join(collocated_words)
                    if collocated_sentence in self.collocated_sentences[mode].keys():
                        collocated_words.extend(self.collocated_sentences[mode][collocated_sentence])
                        collocated_sentence = ''.join(self.collocated_sentences[mode][collocated_sentence])
                        words = None
                        while True:
                            if collocated_sentence in self.collocated_sentences[mode].keys():
                                words = self.collocated_sentences[mode][collocated_sentence].copy()
                                collocated_sentence = ''.join(words)
                                continue
                            else:
                                if words is not None:
                                    collocated_words.extend(words.copy())
                                break
                        collocated_multiple_sentences[current_sentence] = collocated_words.copy()

                    collocated_multiple_sentences[current_sentence] = collocated_words.copy()
                return collocated_multiple_sentences
            else:
                collocated_multiple_sentences = {}
                for current_sentence, val in self.collocated_sentences[mode].items():
                    collocated_words = val.copy()
                    collocated_sentence = ''.join(collocated_words)
                    for _ in range(self.num_sentence - 1):
                        if collocated_sentence in self.collocated_sentences[mode].keys():
                            collocated_words.extend(self.collocated_sentences[mode][collocated_sentence])
                            collocated_sentence = ''.join(self.collocated_sentences[mode][collocated_sentence])
                        else:
                            break
                    collocated_multiple_sentences[current_sentence] = collocated_words.copy()
                return collocated_multiple_sentences


class GetNextSentences(GetCollocatedSentences):
    def __init__(self, with_bccwj=False, num_sentence=1):
        super().__init__(num_sentence=num_sentence)
        for mode in ['train', 'dev', 'test']:
            if with_bccwj:
                with Path(f'../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/next_sentence_{mode}_bccwj.pkl').open('rb') as f:
                    self.collocated_sentences[mode] = pickle.load(f)
            else:
                with Path(f'../../data/NTC_dataset/next_sentence_{mode}.pkl').open('rb') as f:
                    self.collocated_sentences[mode] = pickle.load(f)

    # def get_collocated_sentences(self, mode='all'):
    #     if mode == 'all':
    #         next_sentences = {}
    #         for mode in self.collocated_sentences.keys():
    #             next_sentences.update(self.collocated_sentences[mode])
    #         return next_sentences
    #     else:
    #         return self.collocated_sentences[mode]

class GetPreviousSentences(GetCollocatedSentences):
    def __init__(self, with_bccwj=False, num_sentence=1):
        super().__init__(num_sentence=num_sentence)
        for mode in ['train', 'dev', 'test']:
            if with_bccwj:
                with Path(f'../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/previous_sentence_{mode}_bccwj.pkl').open('rb') as f:
                    self.collocated_sentences[mode] = pickle.load(f)
            else:
                with Path(f'../../data/NTC_dataset/previous_sentence_{mode}.pkl').open('rb') as f:
                    self.collocated_sentences[mode] = pickle.load(f)

    # def get_collocated_sentences(self, mode='all'):
    #     if mode == 'all':
    #         previous_sentences = {}
    #         for mode in self.previous_sentences.keys():
    #             previous_sentences.update(self.previous_sentences[mode])
    #         return previous_sentences
    #     else:
    #         return self.previous_sentences[mode]


if __name__ == '__main__':
    # cls = GetNextSentences(with_bccwj=True, num_sentence=2)
    # ns = cls.get_collocated_sentences(mode='train')
    # print(len(ns))

    cls = GetPreviousSentences(with_bccwj=True, num_sentence=-2)
    ps = cls.get_collocated_sentences(mode='train')
    print(len(ps))