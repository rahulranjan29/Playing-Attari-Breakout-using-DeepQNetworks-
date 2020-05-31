# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:10:02 2020

@author: Rahul Verma
"""

import os
import torch

from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer