# -*- coding: utf-8 -*-


class CheckPoint:
    def __init__(self):
        self.target_data = None
        self.current_data = None

    def is_minimum(self, data):
        self.current_data = data
        if self.target_data is None:
            self.target_data = data
        else:
            if self.current_data <= self.target_data:
                self.target_data = self.current_data
                return True
        return False

    def is_maximum(self, data):
        self.current_data = data
        if self.target_data is None:
            self.target_data = data
        else:
            if self.current_data >= self.target_data:
                self.target_data = self.current_data
                return True
        return False