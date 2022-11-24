# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
import torch.optim as optim

@dataclass
class Config:
    """Class to store program configuration parameters"""
    def __init__(self, args):
        self.benchmark = args.benchmark
        self.widths = args.width
        self.verbose: bool = args.quiet
        self.scalar: bool = args.scalar
        self.learning_mode = args.stopping_criterion['mode']
        self.target_error = args.stopping_criterion['target-error']
        self.loss_stop = float(args.stopping_criterion['loss-stop'])
        self.loss_grad_stop = float(args.stopping_criterion['loss-grad-stop'])
        self.loss_grad_grad_stop = float(args.stopping_criterion['loss-grad-grad-stop'])
        if args.optimizer['type'] == 'AdamW':
            self.optimizer = optim.AdamW
            self.lr = float(args.optimizer['lr'])
        elif args.optimizer['type'] == 'SGD':
            self.optimizer = optim.SGD
            self.lr = float(args.optimizer['lr'])
            self.momentum = float(args.optimizer['momentum'])
        self.iterative = args.iterative
        self.reduction = float(args.reduction)
        self.timeout = args.timeout
        self.timeout_duration = int(args.timeout_duration)
        self.seed = args.seed
        self.repeat = args.repeat
        self.save_net = args.save_net
        self.output_type = args.output_type
        self.output_file = args.output_file
        self.bounded_time = args.bounded_time
        self.time_horizon = args.time_horizon
        self.initial = args.initial
        self.forbidden = args.forbidden

