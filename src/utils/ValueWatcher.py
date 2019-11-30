# -*- coding: utf-8 -*-


class ValueWatcher:
    def __init__(self):
        self.value = 0.0
        self.counter = 0

    def set(self, data):
        self.value = data
        self.counter = 0

    def add(self, value):
        self.counter += 1
        self.value += value

    def get_ave(self):
        if self.counter == 0:
            return 0.0
        ret = float(self.value) / self.counter
        return ret

    def maximum(self, data):
        if self.value < data:
            self.value = data
            return True
        return False

    def reset(self):
        self.counter = 0
        self.value = 0.0