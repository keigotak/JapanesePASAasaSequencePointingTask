from pathlib import Path
import pickle


class GetNextSentences:
    def __init__(self, with_bccwj=False):
        self.next_sentences = {}
        for mode in ['train', 'dev', 'test']:
            if with_bccwj:
                with Path(f'../../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/next_sentence_{mode}_bccwj.pkl').open('rb') as f:
                    self.next_sentences[mode] = pickle.load(f)
            else:
                with Path(f'../../data/NTC_dataset/next_sentence_{mode}.pkl').open('rb') as f:
                    self.next_sentences[mode] = pickle.load(f)

    def get_next_sentences(self, mode='all'):
        if mode == 'all':
            next_sentences = {}
            for mode in self.next_sentences.keys():
                next_sentences.update(self.next_sentences[mode])
            return next_sentences
        else:
            return self.next_sentences[mode]


if __name__ == '__main__':
    cls = GetNextSentences(with_bccwj=True)
    ns = cls.get_next_sentences()
    print(len(ns))