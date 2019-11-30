# -*- coding: utf-8 -*-


class DevChecker:
    def __init__(self, patients=10):
        self.counter = 0
        self.patients = patients

    def is_dev(self):
        self.counter += 1
        if self.counter > self.patients:
            self.counter = 0
            return True
        return False