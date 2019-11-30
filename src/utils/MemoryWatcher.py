# -*- coding: utf-8 -*-
import torch


class MemoryWatcher:
    def __init__(self):
        self.divice = None
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.divice = torch.device('cpu')
        self.available = torch.cuda.is_available()
        self.current_memory = None
        self.maximum_memory = self.get_maximum_memory()
        self.current_cache = None
        self.maximum_cache = self.get_maximum_cache()

    def get_current_memory(self):
        if not self.available:
            return 0
        self.current_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2
        return self.current_memory

    def get_maximum_memory(self):
        if not self.available:
            return 0
        self.maximum_memory = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        return self.maximum_memory

    def get_current_cache(self):
        if not self.available:
            return 0
        self.current_cache = torch.cuda.memory_cached(self.device) / 1024 ** 2
        return self.current_cache

    def get_maximum_cache(self):
        if not self.available:
            return 0
        self.maximum_cache = torch.cuda.max_memory_cached(self.device) / 1024 ** 2
        return self.maximum_cache

    def info(self):
        self.get_current_memory()
        self.get_current_cache()
        return "[Memory] {0:.2}MB / {1:.2}MB [Cache] {2:.2}MB / {3:.2}MB".format(self.current_memory, self.maximum_memory, self.current_cache, self.maximum_cache)

    def release_cache(self):
        torch.cuda.empty_cache()