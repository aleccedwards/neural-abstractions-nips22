# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
from typing import List
from interval import interval


def square_init_data(domain, batch_size):
    """Samples points uniformly over a rectangular domain.

    Args:
        domain (List[float]): 
            lower and upper bounds of the domain.
        batch_size (int): number of points to sample.

    Returns:
        data (torch.Tensor): sampled data
    """

    r1 = torch.tensor(domain[0])
    r2 = torch.tensor(domain[1])
    square_uniform = (r1 - r2) * torch.rand(batch_size, len(domain[0])) + r2
    return square_uniform


def n_dim_sphere_init_data(centre, radius, batch_size):
    """Generates points in a n-dim sphere: X**2 <= radius**2
    
        Adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        method 20: Muller generalised

    Args:
        centre (List): centre of sphere in R^n
        radius (float): radius of sphere
        batch_size (int): number of points to sample.

    Returns:
        data (torch.Tensor): sampled data
    """

    dim = len(centre)
    u = torch.randn(
        batch_size, dim
    )  # an array of d normally distributed random variables
    norm = torch.sum(u ** 2, dim=1) ** (0.5)
    r = radius * torch.rand(batch_size, dim) ** (1.0 / dim)
    x = torch.div(r * u, norm[:, None]) + torch.tensor(centre)

    return x


class Rectangle:
    """Hypercube in R^n"""
    def __init__(self, lb: List[float], ub: List[float]):
        """Initiaises hypercube in R^n.

        Expresses hypercube as lower and upper bounds.

        Args:
            lb (List[float]): list of lower bounds.
            ub (List[float]): list of upper bounds.
        """
        self.name = "rectangle"
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dimension = len(lb)

    def generate_domain(self, x: List, _And):
        """Returns symbolic formula for domain in terms of x.

        Args:
            x (List): symbolic variables.
            _And (_type_): And function for symbolic formula.

        Returns:
            domain: symbolic formula for domain.
        """
        lower = _And(*[self.lower_bounds[i] <= x[i] for i in range(self.dimension)])
        upper = _And(*[x[i] <= self.upper_bounds[i] for i in range(self.dimension)])
        return _And(lower, upper)

    def generate_data(self, batch_size: int) -> torch.Tensor:
        """Returns data points uniformly distributed over the rectangle.

        Args:
            batch_size (int): number of points to sample.

        Returns:
            torch.Tensor: sampled data
        """        
        return square_init_data([self.lower_bounds, self.upper_bounds], batch_size)

    def generate_bloated_data(self, batch_size: int, bloat=0.1) -> torch.Tensor:
        """Returns data points uniformly distributed over a slightly larger rectangle.

        Args:
            batch_size (int): nuber of points to sample.
            bloat (float, optional):  additive increase in size of rectangle. Defaults to 0.1.

        Returns:
            torch.Tensor: sampled data
        """        
        return square_init_data(
            [
                [lb - bloat for lb in self.lower_bounds],
                [ub + bloat for ub in self.upper_bounds],
            ],
            batch_size=batch_size,
        )

    def as_intervals(self) -> List[interval]:
        """Expresses rectangle as intervals."""
        return [interval(dx) for dx in zip(self.lower_bounds, self.upper_bounds)]

    def check_interior(self, S: torch.Tensor) -> torch.Tensor:
        """Checks if points in S are in interior of rectangle.

        Returns tensor of booleans indicating whether each point is in interior.

        Args:
            S (torch.Tensor): data points to check.

        Returns:
            torch.Tensor: tensor of boolean values.
        """
        lb = torch.tensor(self.lower_bounds)
        ub = torch.tensor(self.upper_bounds)
        res = torch.logical_and(S > lb, S < ub).all(dim=1)
        return res

    def scale(self, factor: List):
        """Scales each dimension of rectangle by factor in list"""
        lb = [bound * s for (bound, s) in zip(self.lower_bounds, factor)]
        ub = [bound * s for (bound, s) in zip(self.upper_bounds, factor)]
        return Rectangle(lb, ub)

    def shift(self, shift):
        """Shifts each dimension of rectangle by shift in list"""
        lb = [bound + s for (bound, s) in zip(self.lower_bounds, shift)]
        ub = [bound + s for (bound, s) in zip(self.upper_bounds, shift)]
        return Rectangle(lb, ub)


class Sphere:
    """Hypersphere in R^n"""
    def __init__(self, centre: List[float], radius: float):
        """Initiaises hypersphere in R^n.

        Args:
            centre (List[float]): centre of sphere.
            radius (float): radius of sphere.
        """        
        self.name = "sphere"
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)

    def generate_domain(self, x: List, _And):
        """Returns symbolic formula for domain in terms of x.

        Args:
            x (List): symbolic variables.
            _And (_type_): And function for symbolic formula.

        Returns:
            domain: symbolic formula for domain.
        """
        return _And(
            sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
            <= self.radius ** 2
        )

    def generate_data(self, batch_size):
        """Returns data points uniformly distributed over the rectangle.

        Args:
            batch_size (int): number of points to sample.

        Returns:
            torch.Tensor: sampled data
        """  
        return n_dim_sphere_init_data(self.centre, self.radius ** 2, batch_size)


if __name__ == "__main__":
    r = Rectangle([-1, -1], [1, 1])
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.array(r.generate_bloated_data(1000, bloat=0.1)).reshape(-1, 2)
    res = np.array(r.check_interior(torch.tensor(x)))
    x = x[res]
    r2 = r.scale([2, 3])
    r2 = r2.shift([1, 3])
    x = np.array(r2.generate_data(1000))
    plt.plot(x[:, 0], x[:, 1], "x")
    print(r2.lower_bounds, r2.upper_bounds)

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.show()
