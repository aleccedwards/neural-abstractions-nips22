# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

import torch
import z3
import numpy as np

from cegis.verifier import z3_replacements, Z3Verifier
from cegis.nn import ReluNet
from utils import *
from cegis.translator import Translator
from cli import get_default_config
from benchmarks import read_benchmark

config = get_default_config()

class testTranslatorZ3(unittest.TestCase):
    def setUp(self) -> None:
        self.width = [3,3]
        bench = read_benchmark('lin')
        self.dimension = bench.dimension
        torch.manual_seed(0)
        self.net = ReluNet(bench, self.width, 0.1, config)
        x0, x1 = [z3.Real("x%d" % i) for i in range(self.dimension)]
        self.translator = Translator((x0, x1), Z3Verifier.relu)

    def test_output(self):
        output = self.translator.translate(self.net)

    def test_output_similarity(self):
        X = torch.randn(10, self.dimension)
        true_output = self.net.model(X).detach()
        translation = self.translator.translate(self.net, dp=20)
        translated_output = np.array(
            [
                [
                    float(
                        (
                            z3_replacements(
                                translation[0, 0],
                                self.translator.input_vars,
                                np.array(x.unsqueeze(1)),
                            )
                        ).as_fraction()
                    ),
                    float(
                        (
                            z3_replacements(
                                translation[1, 0],
                                self.translator.input_vars,
                                np.array(x.unsqueeze(1)),
                            )
                        ).as_fraction()
                    ),
                ]
                for x in X
            ]
        )
        self.assertTrue(np.allclose(true_output, translated_output))



if __name__ == "__main__":
    unittest.main()
