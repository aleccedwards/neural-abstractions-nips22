# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

import torch

from cegis.nn import ReluNet
from benchmarks import read_benchmark
from cli import get_default_config

config = get_default_config()


class testNN(unittest.TestCase):
    def setUp(self) -> None:
        bench = read_benchmark('lin')
        self.dimension = bench.dimension
        self.width = [20]
        self.net = ReluNet(bench, self.width, 0.1, config)

    def test_input_output(self):
        data = torch.randn(100, self.dimension)
        output = self.net(data)
        self.assertSequenceEqual(output.shape, (100, self.dimension))

    def test_width(self):
        params = list(self.net.parameters())
        self.assertEqual(self.width[0], params[0].shape[0])


if __name__ == "__main__":
    unittest.main()
