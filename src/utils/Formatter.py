# -*- coding: utf-8 -*-
import re
import MeCab


class Formatter:
    def __init__(self):
        self.m = MeCab.Tagger("-Owakati")

    def parse(self, sentence):
        sentence = self.m.parse(sentence)
        sentence = sentence.split()
        return sentence

    @staticmethod
    def strip(sentence):
        return re.sub('\s', '', sentence)
