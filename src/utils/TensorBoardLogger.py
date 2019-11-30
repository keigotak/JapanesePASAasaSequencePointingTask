# -*- coding: utf-8 -*-
import tensorboardX as tbx


class TensorBoardLogger:
    def __init__(self, log_dir=None):
        self.writer = tbx.SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()