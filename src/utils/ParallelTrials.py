from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from pathlib import Path
import pickle
import os
from time import sleep


class ParallelTrials:
    def __init__(self, path):
        super(ParallelTrials, self).__init__()
        self.trial_pkl_path = path
        self.trials = Trials()
        if not self.trial_pkl_path.exists():
            with self.trial_pkl_path.open('wb') as f:
                pickle.dump(self.trials, f)

        if self.trial_pkl_path.exists() and self.get_access():
            with self.trial_pkl_path.open('rb') as f:
                self.trials = pickle.load(f)

    def add(self, _trials, save=False):
        if self.trial_pkl_path.exists() and self.get_access():
            with self.trial_pkl_path.open('rb') as f:
                self.trials = pickle.load(f)
            best_score = 0.0
            if len(self.trials) != 0:
                best_score = self.trials.best_trial['result']['loss']
            for i in range(len(_trials) - 1, -1, -1):
                tid = 0
                if len(self.trials) != 0:
                    tid = self.trials.tids[-1] + 1
                _dynamic_trials = _trials._dynamic_trials[i]

                if self.is_on_trials(_dynamic_trials):
                    continue

                _dynamic_trials['tid'] = tid
                _dynamic_trials['misc']['tid'] = tid
                for key in _dynamic_trials['misc']['idxs']:
                    _dynamic_trials['misc']['idxs'][key][0] = tid
                _ids = tid
                score = _dynamic_trials['result']['loss']

                self.trials._dynamic_trials.append(_dynamic_trials)
                self.trials._ids.add(_ids)
                self.trials._trials.append(_dynamic_trials)
                if score < best_score:
                    for key, value in _dynamic_trials['misc']['vals'].items():
                        self.trials.argmin[key] = value[0]

                    for key in self.trials.best_trial:
                        self.trials.best_trial[key] = _dynamic_trials[key]
                for key in self.trials.idxs.keys():
                    self.trials.idxs[key].append(_ids)
                    self.trials.idxs_vals[0][key].append(_ids)
                for key, value in _dynamic_trials['misc']['vals'].items():
                    self.trials.idxs_vals[1][key].append(value[0])
                    self.trials.vals[key].append(value[0])
                self.trials.miscs.append(_dynamic_trials)
                self.trials.results.append(_dynamic_trials['result'])
                self.trials.specs.append(None)
                self.trials.tids.append(_ids)

        if self.trial_pkl_path.exists() and self.get_access() and save:
            with self.trial_pkl_path.open('wb') as f:
                pickle.dump(self.trials, f)

    def get_access(self):
        accessed = False
        if self.trial_pkl_path.exists():
            for i in range(180):
                try:
                    os.rename(str(self.trial_pkl_path.resolve()), str(self.trial_pkl_path.resolve()))
                except OSError as e:
                    sleep(1)
                else:
                    accessed = True
                    break
        return accessed

    def is_on_trials(self, _trial):
        for i in range(len(self.trials._dynamic_trials) - 1, -1, -1):
            if self.trials._dynamic_trials[i]['book_time'] == _trial['book_time'] and self.trials._dynamic_trials[i]['refresh_time'] == _trial['refresh_time']:
                return True
        return False

    def get_best_score(self):
        if len(self.trials) != 0:
            return self.trials.best_trial['result']['loss']
        return 0.0


if __name__ == '__main__':
    # ptrials = ParallelTrials(path=Path('../../results/trials/pointer1.pkl'))
    ptrials = ParallelTrials(path=Path('../../results/trials/lstm.pkl'))

    # test_path = Path('../../results/trials/pointer2.pkl')
    test_path = Path('../../results/trials/lstm1.pkl')
    test_trials = Trials()
    with test_path.open('rb') as f:
        test_trials = pickle.load(f)

    # test_path = Path('../../results/trials/pointer.pkl')
    # test_trials = Trials()
    # with test_path.open('rb') as f:
    #     test_trials = pickle.load(f)
    ptrials.add(test_trials)

    # save_path = Path('../../results/trials/pointer_merged.pkl')
    # with save_path.open('wb') as f:
    #     pickle.dump(ptrials.trials, f)

    pass