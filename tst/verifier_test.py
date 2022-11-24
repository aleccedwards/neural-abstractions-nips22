# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

import numpy as np

from domains import Sphere
from cegis.verifier import Z3Verifier


class testZ3Verifer(unittest.TestCase):
    def setUp(self) -> None:
        self.dimension = 3
        self.x = Z3Verifier.new_vars(self.dimension)
        self.error = 0.01
        s = Sphere([0, 0, 0], 10)
        self.verifier = Z3Verifier(self.x, self.dimension, s.generate_domain)
        
    def test_positive(self):
        true_f = 5 * np.random.rand(3, 3) * np.array(self.x).reshape(-1, 1)
        shifted_f = true_f - self.error / 2
        res, _ = self.verifier.verify(true_f, shifted_f, self.error)
        self.assertTrue(res)

    def test_negative(self):
        true_f = 5 * np.random.rand(3, 3) * np.array(self.x).reshape(-1, 1)
        shifted_f = true_f - np.sqrt(self.error) * 2
        res, _ = self.verifier.verify(true_f, shifted_f, self.error)
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
