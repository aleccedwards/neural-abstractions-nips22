# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from re import X
from typing import Callable, List, Union

import dreal
import sympy as sp
from interval import imath, interval
import numpy as np
import torch
import sysplot
from domains import Rectangle, Sphere


class Benchmark:
    """Non-linear dynamical model"""

    def __init__(self) -> None:
        self.dimension: int = None
        self.name: str = None
        self.domain: Union[Rectangle, Sphere] = None

    def f(v: List) -> List:
        """function to evaluate dynamic model at given point

        Evaluates (x_0, ..., x_n) -> f(x_0, ..., x_n-1)

        Args:
            v (List): list of variables (x_0, ..., x_n)

        Returns:
            List: vector of values of f(x_0, ..., x_n-1)
        """
        x, y = v
        return [-x + y, -y]

    def get_domain(self, x: List, _And: Callable):
        """Returns symbolic formula for domain in terms of x.

        Args:
            x (List): symbolic variables.
            _And (Callable): And function for symbolic formula.

        Returns:
            domain: symbolic formula for domain.
        """
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        """Returns data points uniformly distributed over a slightly larger rectangle.

        Args:
            batch_size (int): nuber of points to sample.
            bloat (float, optional):  additive increase in size of rectangle. Defaults to 0.1.

        Returns:
            torch.Tensor: sampled data
        """
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        """Plots the benchmark and a neural network's output.

        Args:
            net (ReLUNet): Neural network to plot.
            name (str, optional):
                Name of file to save plot to. If None, then plot is show.
                Defaults to None.
        """
        sysplot.plot_vector_fields(net, self, [-1, 1], [-1, 1], name=name)

    def get_image(self):
        """Find the image of the function wrt the domain."""
        if isinstance(self.domain, Sphere):
            raise ValueError("Intervals not implemented for spherical domains")
        return self.f(self.domain.as_intervals())

    def normalise(self):
        """Normalise dynamics to [-1, 1]^d."""
        scale = []
        for dim in self.image:
            scale.append((dim[0].sup - dim[0].inf) / 2)
        self.scale = scale
        self.name += "-normalised"

    def unnormalise(self):
        """Unnormalise dynamics"""
        self.scale = [1 for i in range(self.dimension)]
        self.name = self.name[:-11]

    def get_scaling(self):
        """Determine scaling for normalisation.

        Returns:
            shift (np.ndarray): shift for normalisation.
            scale (np.ndarray): scale for normalisation.
        """
        scales = []
        shifts = []
        for dim in self.image:
            shifts.append(dim.midpoint[0].inf)
            scales.append((dim[0].sup - dim[0].inf) / 2)
        return np.array(shifts).reshape(-1, 1), list(scales)

    def f_intervals(self, v):
        """Evaluate model using interval arithmetic.

        Relies on PyInterval."""
        return self.f(v)


class Linear(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Linear"
        self.short_name = "lin"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]
        self.image = self.get_image()
        # self.normalise()

    def f(self, v):
        x, y = v
        f = [-x + y, -y]
        return [fi / si for fi, si in zip(f, self.scale)]

    def get_domain(self, x: List, _And):
        """
        Returns smt (symbolic) domain object
        """
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        sysplot.plot_vector_fields(net, self, [-1, 1], [-1, 1], name=name)


class SteamGovernor(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "Steam"
        self.short_name = "steam"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        self.image = self.get_image()
        # self.normalise()

    def f(self, v):
        x, y, z = v
        try:
            f = [
                y,
                z**2 * np.sin(x) * np.cos(x) - np.sin(x) - 3 * y,
                -(np.cos(x) - 1),
            ]
        except TypeError:
            try:
                f = [
                    y,
                    z**2 * dreal.sin(x) * dreal.cos(x) - dreal.sin(x) - 3 * y,
                    -(dreal.cos(x) - 1),
                ]
            except TypeError:
                f = [
                    y,
                    z**2 * sp.sin(x) * sp.cos(x) - sp.sin(x) - 3 * y,
                    -(sp.cos(x) - 1),
                ]
        return [fi / si for fi, si in zip(f, self.scale)]

    def f_intervals(self, v):
        x, y, z = v
        f = [
            y,
            z**2 * imath.sin(x) * imath.cos(x) - imath.sin(x) - 3 * y,
            -(imath.cos(x) - 1),
        ]
        return [fi / si for fi, si in zip(f, self.scale)]

    def get_image(self):
        return self.f_intervals(self.domain.as_intervals())


class JetEngine(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Jet Engine"
        self.short_name = "jet"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        self.image = self.get_image()
        # self.normalise()

    def f(self, v):
        x, y = v
        f = [-y - 1.5 * x**2 - 0.5 * x**3 - 0.1, 3 * x - y]
        return [fi / si for fi, si in zip(f, self.scale)]


class NL1(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz1"
        self.short_name = "NL1"
        self.domain = Rectangle([0, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n, bloat=0)

    def f(self, v):
        x, y = v
        x = x
        y = y
        try:
            f = [y, np.sqrt(x)]
            # f = [y**2 + 0.5*y, np.sqrt(x)]
        except TypeError:
            try:
                f = f = [y, dreal.sqrt(x)]
            except TypeError:
                f = [
                    y - sp.sqrt(x),
                ]
        return [fi / si for fi, si in zip(f, self.scale)]


class NL2(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz2"
        self.short_name = "NL2"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def f(self, v):
        x, y = v
        x = x
        y = y
        try:
            if isinstance(x, np.ndarray):
                f = [x**2 + y, np.power(np.power(x, 2), 1 / 3) - x]
            else:
                f = [x**2 + y, torch.pow(torch.pow(x, 2), 1 / 3) - x]
        except TypeError:
            try:
                f = f = [x**2 + y, dreal.pow(dreal.pow(x, 2), 1 / 3) - x]
            except TypeError:
                f = [-x*y + y, sp.Pow(sp.Pow(x, 1), 1 / 3) - x]
        return [fi / si for fi, si in zip(f, self.scale)]


class WaterTank(Benchmark):
    def __init__(self) -> None:
        self.dimension = 1
        self.name = "Water-tank"
        self.short_name = "tank"
        self.domain = Rectangle([0], [2])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n, bloat=0)
    
    def f(self, v):
        x = v
        try:
            if isinstance(x, np.ndarray):
                f = [1.5 - np.sqrt(x)]
            else:
                f = [1.5 - torch.sqrt(x)]
        except TypeError:
            try:
                x, = v
                f = f = [1.5 - dreal.sqrt(x)]
            except TypeError:
                f = [1.5 - sp.sqrt(x)]
        return [fi / si for fi, si in zip(f, self.scale)]
    
    def plotting(self, net, name: str = None):
        return

class Exponential(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Exponential"
        self.short_name = "exp"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        self.image = self.get_image()
        # self.normalise()#

    def f(self, v):
        x, y = v
        x = x
        y = y
        try:
            f = [-np.sin(np.exp(y**3 + 1)) - y**2, -x]
        except TypeError:
            try:
                f = f = [-dreal.sin(dreal.exp(y**3 + 1)) - y**2, -x]
            except TypeError:
                f = [sp.sin(sp.exp(y + 1)), sp.exp(sp.sin(x + 1))]
        return [fi / si for fi, si in zip(f, self.scale)]

    def f_intervals(self, v):
        x, y = v
        f = [imath.sin(imath.exp(y)), imath.exp(imath.sin(x))]
        return [fi / si for fi, si in zip(f, self.scale)]



def read_benchmark(name: str):
    """Reads a benchmark from a string and returns a Benchmark object

    Args:
        name (str): corresponding to a benchmark name

    Returns:
        benchmark (Benchmark): benchmark object
    """
    if name == "lin":
        return Linear()
    elif name == "exp" or name == "Exponential":
        return Exponential()
    elif name == "steam":
        return SteamGovernor()
    elif name == "jet":
        return JetEngine()
    elif name == "buck":
        return NL1()
    elif name == "nl2":
        return NL2()
    elif name == "tank":
        return WaterTank()


if __name__ == "__main__":
    for n in ("nl2",):
        b = read_benchmark(n)
        # b.normalise()
        print(b.name)
        # print(b.get_image())
        if b.dimension == 2:
            from matplotlib import pyplot as plt

            sysplot.plot_benchmark(b)
            plt.gca().set_aspect("equal")
            sysplot.show()
