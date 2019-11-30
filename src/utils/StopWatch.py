# -*- coding: utf-8 -*-
import time


class StopWatch:
    def __init__(self):
        self.base = None

    def start(self):
        self.base = time.time()

    def stop(self):
        return time.time() - self.base