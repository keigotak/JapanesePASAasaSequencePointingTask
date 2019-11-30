# -*- coding: utf-8 -*-
import math


class EarlyStop:
    def __init__(self, patients=10, go_minus=True):
        self.patients = patients
        self.counter = 0
        self.target_data = None
        self.current_data = None
        self.go_minus = go_minus

    def is_over(self):
        if self.patients < 0:
            return False

        if self.counter >= self.patients:
            return True
        return False

    def is_counted_delay(self, data):
        self.current_data = data
        if self.target_data is None:
            self.target_data = data
        else:
            if math.isnan(self.current_data):
                self.counter += 1
            else:
                if self.go_minus:
                    if self.target_data > self.current_data:
                        self.counter += 1
                    else:
                        self.counter = 0
                else:
                    if self.target_data < self.current_data:
                        self.counter += 1
                    else:
                        self.counter = 0
        self.target_data = self.current_data
        return self.is_over()

    def is_minimum_delay(self, data):
        self.current_data = data
        if self.target_data is None:
            self.target_data = data
        else:
            if math.isnan(self.current_data):
                self.counter += 1
            else:
                if self.current_data >= self.target_data:
                    self.counter += 1
                else:
                    self.counter = 0
                self.target_data = min(self.current_data, self.target_data)
        return self.is_over()

    def is_maximum_delay(self, data):
        self.current_data = data
        if self.target_data is None:
            self.target_data = data
        else:
            if math.isnan(self.current_data):
                self.counter += 1
            else:
                if self.current_data <= self.target_data:
                    self.counter += 1
                else:
                    self.counter = 0
                self.target_data = max(self.current_data, self.target_data)
        return self.is_over()


if __name__ == "__main__":
    es = EarlyStop()
    es.is_minimum_delay(float("nan"))
    es.is_minimum_delay(float("nan"))
