# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List

import numpy as np
import pandas as pd

from benchmarks import read_benchmark
from cegis.translator import Translator
from utils import Timeout
from cegis.verifier import DRealVerifier


class BoundsFinder:
    """Class to help find upper and lower bounds of abstraction"""
    def __init__(
        self,
    ) -> None:
        self.highest_sat = 0.0
        self.lowest_unsat = float("inf")

    def get_interval(self) -> List[float]:
        """Express interval as a list"""
        return [self.highest_sat, self.lowest_unsat]

    def get_interval_size(self) -> float:
        """Return size of interval"""
        return self.lowest_unsat - self.highest_sat

    def get_interval_midpoint(self) -> float:
        """Return midpoint of interval"""
        return (self.lowest_unsat + self.highest_sat) / 2

    def set_highest_sat(self, val: float):
        """Set highest sat value"""
        self.highest_sat = val

    def set_lowest_unsat(self, val: float):
        """Set lowest unsat value"""
        self.lowest_unsat = val

    def reset_bounds(self):
        """Reset bounds to default values"""
        self.highest_sat = 0.0
        self.lowest_unsat = float("inf")

    def find_interval(self, benchmark, net, verifier: DRealVerifier, target_size=0.01):
        """Finds error interval to a given target size

        Iteratively checks an error bound and stores if result is SAT or UNSAT
        to determine lower and upper bounds of approximation error.

        Args:
            benchmark (benchmarks.Benchmark): benchmark dynamical model
            net (_type_): network of abstraction
            verifier (DRealVerifier): verifier to use
            target_size (float, optional): target size of interval. Defaults to 0.01.
        """
        while self.get_interval_size() > target_size:
            next_error = self.get_interval_midpoint()
            true_f = np.array(benchmark.f(verifier.vars)).reshape(-1, 1)
            res, _ = verifier.verify(true_f, net, epsilon=next_error)
            if res:
                # query unsat
                self.set_lowest_unsat(next_error)
            else:
                # query sat
                self.set_highest_sat(next_error)


if __name__ == "__main__":
    errors = []
    widths = []
    lower_bounds = []
    upper_bounds = []
    e = [0.1, 0.01, 0.001, 0.0001]
    w = [ 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200,]
    bf = BoundsFinder()
    for error in e:
        for width in w:
            b = read_benchmark("lv")
            vars = DRealVerifier.new_vars(b.dimension)
            v = DRealVerifier(vars, b.dimension, b.get_domain, error)
            net = None #load_model(b.name, error, width, b.dimension)
            t = Translator(vars, DRealVerifier.relu)
            translation = t.translate(net)
            true_f = np.array(b.f(vars)).reshape(-1, 1)
            res, _ = v.verify(true_f, translation, epsilon=error)
            if not res:
                continue
            bf.set_lowest_unsat(error)

            try:
                with Timeout(seconds=600):
                    bf.find_interval(b, translation, v, target_size=0.001)
            except TimeoutError:
                continue
            lb, ub = bf.get_interval()
            widths.append(width)
            errors.append(error)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
            results = {"e": errors, "w": widths, "lb": lower_bounds, "ub": upper_bounds}
            df = pd.DataFrame.from_dict(results)
            df.to_csv("results/bounds-lv.csv", index=False)
            bf.reset_bounds()
