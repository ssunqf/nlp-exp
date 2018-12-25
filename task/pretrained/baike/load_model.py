#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch

from .train import Config, Trainer


if __name__ == '__main__':
    trainer = Trainer.create(Config())