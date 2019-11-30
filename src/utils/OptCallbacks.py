# -*- coding: utf-8 -*-
from pathlib import Path


class OptCallbacks:
    def __init__(self):
        self.counter = 0
        self.params = []
        self.path = None

    def count_up(self):
        self.counter += 1

    def get_count(self):
        return self.counter

    def set_model_param(self, _params):
        _params = [str(i) for i in _params]
        self.params = _params

    def get_model_save_dir(self, _tag, _now):
        dir_tag = _tag + "-{0:%Y%m%d-%H%M%S}".format(_now)
        _path = Path('../../results').joinpath(dir_tag)
        _path = _path.joinpath('model-{}'.format(self.counter))
        self.path = _path
        _path.mkdir(exist_ok=True)
        return _path, str(_path.resolve())

    def save_model_info(self, _params):
        self.set_model_param(_params)
        path = self.path.joinpath('info.txt')
        with path.open('w') as f:
            _line = '\n'.join(self.params)
            f.write(_line + '\n')
