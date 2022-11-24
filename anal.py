# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os.path
from argparse import Namespace
import csv
import warnings
from typing import List

import torch

from cegis.nn import ReconstructedRelu
from benchmarks import Benchmark
from utils import Timeout
from cegis.verifier import DRealVerifier
from cegis.translator import Translator
from cegis.bounds_finder import BoundsFinder


class CSVWriter:
    """Class for writing results to csv file."""
    def __init__(self, filename: str, headers: List[str]) -> None:
        """Initializes CSVWriter.

        If the file does not exist, it will be created here
        and the header will be written to it.

        Args:
            filename (str): filename of csv file.
            headers (List[str]): headers for csv file
        """        
        self.headers = headers
        self.filename = filename
        if not os.path.isfile(self.filename):
            self.write_header_to_csv()

    def write_header_to_csv(self):
        """Creates csv file and writes a header to it."""
        with open(self.filename, "a") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.headers, delimiter=",", lineterminator="\n"
            )
            writer.writeheader()

    def write_row_to_csv(self, values: List):
        """ Writes values to row in CSV file."""
        if len(values) != len(self.headers):
            warnings.warn("More values to write than columns in csv.")
        with open(self.filename, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(values)


class Analyser:
    """Object to calculate and record stats of abstraction."""
    def __init__(self, abstraction) -> None:
        """initializes Analyser.

        Args:
            abstraction (NeuralAbstraction): abstraction to analyse
        """
        self.filename = "results/"
        self.abstraction = abstraction
        self.error_calculator = ErrorCalculator(abstraction.benchmark, abstraction.reconstructed_net)

    def record(self, filename: str, config, res: str, T: float):
        """records results of abstraction to csv file.

        Args:
            filename (str): csv file to write to
            config (_type_): program configuration
            res (str): result of synthesis
            T (float): duration of synthesis
        """
        self.filename = self.filename + filename + ".csv"
        headers = [
            "Benchmark",
            "Width",
            "Partitions",
            "Error_1_norm",
            "Error_2_norm",
            "Error_inf_norm",
            "Error",
            "Result",
            "Est_Mean_se",
            "Est_Max_se",
            "Seed",
            "Method",
            "Time"
        ]
        writer = CSVWriter(self.filename, headers)
        if config.scalar:
            method = "SFNA"
        else:
            method = "FNA"
        if config.iterative:
            method = "I" + method
        mse, Mse = self.estimate_errors()
        p = len(self.abstraction.locations)
        error = torch.tensor((self.abstraction.error))
        writer.write_row_to_csv(
            [
                self.abstraction.benchmark.name,
                config.widths,
                p,
                round(error.norm(p=1).item(), ndigits=5),
                round(error.norm(p=2).item(), ndigits=5),
                round(error.norm(p=torch.inf).item(), ndigits=5),
                self.abstraction.error,
                res,
                round(mse, ndigits=5),
                round(Mse, ndigits=5),
                config.seed,
                method,
                round(T, ndigits=5)
            ]
        )

    def estimate_errors(self) -> float:
        """Estimates mean and max error of abstraction.

        Returns:
           mse float: mean squared error
           Mse float: max squared error
        """        
        return (
            self.error_calculator.estimate_mse(),
            self.error_calculator.estimate_max_square_error(),
        )

class ErrorCalculator:
    """Class for estimating errors of abstractions."""
    def __init__(self, benchmark: Benchmark, abstraction, p=10000) -> None:
        """initializes ErrorCalculator.

        Args:
            benchmark (Benchmark): _description_
            abstraction (_type_): _description_
            p (int, optional): _description_. Defaults to 10000.
        """
        self.benchmark = benchmark
        self.abstraction = abstraction
        self.p = p

    def estimate_mse(self) -> float:
        """Estimates mean square error of abstraction"""
        data = self.benchmark.domain.generate_data(self.p)
        y = torch.tensor(list(map(self.benchmark.f, data)))
        y_est = self.abstraction(data).T
        mse_error = torch.nn.MSELoss(reduction="mean")
        mse = mse_error(y, y_est)
        return mse.item()

    def estimate_max_square_error(self) -> float:
        """Estimates max square error of abstraction."""
        data = self.benchmark.domain.generate_data(self.p)
        y = torch.tensor(list(map(self.benchmark.f, data)))
        y_est = self.abstraction(data).T
        mse_error = torch.nn.MSELoss(reduction="none")
        max_se = mse_error(y, y_est).max()
        return max_se.item()


def get_error_bounds(benchmark: Benchmark, args: Namespace, res: str, net):
    """Determines upper and lower bounds for approximation error of abstraction.

    Uses on the BoundsFinder class to determine the bounds. 

    Assumes that error is a float so is deprecated at present.

    Args:
        benchmark (Benchmark): benchmark dynamical model
        args (Namespace): command line arguments
        res (str): result of synthesis
        net (_type_): network of abstraction

    Returns:
        bounds (List(floats)): lower and upper bounds
    """
    bf = BoundsFinder()
    if res == "S":
        bf.set_lowest_unsat(args.error)
    else:
        return None, None
    vars = DRealVerifier.new_vars(benchmark.dimension)
    v = DRealVerifier(vars, benchmark.dimension, benchmark.get_domain, error=args.error)
    t = Translator(vars, DRealVerifier.relu)
    abstraction = t.translate(net)
    with Timeout(seconds=120):
        try:
            bf.find_interval(benchmark, abstraction, v, target_size=0.001)
        except TimeoutError:
            pass
    return bf.get_interval()


if __name__ == "__main__":
    from benchmarks import Linear
    b = Linear()
    net = ReluNet(b.dimension, 5, 0.05)
    ec = ErrorCalculator(b, net)
    mse = ec.estimate_mse()
    Mse = ec.estimate_max_square_error()
    print("Mean square error = {}, \nMax square error = {}".format(mse, Mse))
