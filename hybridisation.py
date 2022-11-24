# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from itertools import product
from math import prod
from typing import List

import sympy as sp
import numpy as np
import torch
from interval import interval, imath, fpu
from anal import CSVWriter

from benchmarks import read_benchmark
from domains import Rectangle


class MeshGrid:
    """Mesh grid in R^n"""
    def __init__(self, rectangle: Rectangle, epsilon) -> None:
        """Initialize a mesh grid in R^n

        Args:
            rectangle (doamins.Rectangle): 
                Rectangle domain in which the mesh grid is defined
            epsilon (_type_): 
                size of the mesh grid (length of the grid cells)
        """        
        self.domain = rectangle
        self.epsilon = epsilon
        self.tiks = self.get_tiks()
        self.h = self.get_h()

    def get_tiks(self):
        """Return number of cells in each dimension"""
        tiks = []
        for i in range(self.domain.dimension):
            tiks.append(
                int(
                    (self.domain.upper_bounds[i] - self.domain.lower_bounds[i])
                    / self.epsilon
                )
            )
        return tiks

    def get_total_partitions(self):
        """Return total number of cells in mesh"""
        return prod(self.tiks)

    def get_h(self):
        """Return the largest cell dimension
        
        Unsure how this is different to epsilon"""
        max_h = 0
        for i in range(self.domain.dimension):
            h = (self.domain.upper_bounds[i] - self.domain.lower_bounds[i]) / self.tiks[
                i
            ]
            max_h = h if h > max_h else h
        return max_h

    def get_mode(self, s):
        """Return index location of cell that s lies in"""
        mode = []
        for i, x in enumerate(s):
            assert x <= self.domain.upper_bounds[i]
            assert x >= self.domain.lower_bounds[i]
            j = int(
                (self.tiks[i] * (x - self.domain.lower_bounds[i]))
                / (self.domain.upper_bounds[i] - self.domain.lower_bounds[i])
            )
            mode.append(j)
        return mode


class RectangularCell:
    """Single cell in the mesh grid"""
    def __init__(self, cell) -> None:
        """Initialize a rectangular cell

        Args:
            cell (interval): cell as intervals
        """        """Initialize a rectangular cell"""
        self.cell = cell

    def get_spread(self, cell_dimension):
        """Get length of cell in a given dimension"""
        return cell_dimension[0].sup - cell_dimension[0].inf

    def get_max_spread(self):
        """Get maximum spread of cell over all dimensions"""
        max_spread = 0
        for inclusion_dim in self.cell:
            spread = self.get_spread(inclusion_dim)
            max_spread = spread if spread > max_spread else max_spread
        return max_spread

    def get_corners(self) -> List[List[float]]:
        """Return corners of cell"""
        bounds = [tuple(self.cell[i])[0] for i in range(len(self.cell))]
        corners = product(*bounds)
        return list(corners)

    def get_furthest_corner_dist(
        self, corners: torch.Tensor, point: torch.Tensor, norm
    ):
        """Finds the distance between a point and the furthest corner of the cell.

        Update: This should just call get_corners rather than have as arg

        Args:
            corners (torch.Tensor): corners of cell
            point (torch.Tensor): point to find distance to
            norm (int): norm to use for distance calculation

        Returns:
            dist (float): distance between point and furthest corner
        """    
        d = 0
        for corner in corners:
            dist = (corner - point).norm(p=norm).item() ** 2
            if d < dist:
                d = dist
        return d

    def get_max_corner_norm(self, corners, norm):
        """Gets the further distance between any two corners of the cell.

        Equivalent to max spread when norm = inf
        Args:
            corners (torch.Tensor): corners of cell
            norm (int): norm to use for distance calculation

        Returns:
            _type_: _description_
        """
        d = 0
        for corner1 in corners:
            for corner2 in corners:
                dist = (corner1 - corner2).norm(p=norm).item()
                if d < dist:
                    d = dist
        return d


class SimplicialMesh:
    """Simplicial mesh"""
    def __init__(self, rectangle, epsilon) -> None:
        """Initialize a simplicial mesh

        Args:
            rectangle (domains.Rectangle): 
                Rectangle on which the simplicial mesh is constructed
            epsilon (_type_): size of the simplices
        """
        self.rectangular_mesh = MeshGrid(rectangle, epsilon)
        self.h = self.rectangular_mesh.h

    def get_total_partitions(self):
        """Return total number of simplices in mesh"""
        return prod(self.rectangular_mesh.tiks) * self.rectangular_mesh.domain.dimension


class RectangularAutomata:
    """Object to represent a rectangular automaton"""
    name = "RA"

    def __init__(self, benchmark, epsilon) -> None:
        """Initialize a rectangular automaton

        Args:
            benchmark (benchmarks.Benchmark): 
                benchmark the automaton is to abstract
            epsilon (float): size of rectangular cells
        """        
        self.benchmark = benchmark
        self.domain = self.benchmark.domain
        self.epsilon = epsilon
        self.mesh = MeshGrid(self.domain, self.epsilon)

    def get_error(self, norm=2):
        """Return estimate of error of automaton"""
        return self.estimate_errors(norm)[1]

    def hybridise_point(self, s: torch.Tensor, norm):
        """Return the error associated with a point.

        Finds the errors between the Rectangular Automaton and the 
        benchmark at a given point s.

        Args:
            s (torch.Tensor): point to evaluate
            norm (int): norm to use for error calculation

        Returns:
            max_error_est (float): 
                distance between point and furthest corner in its corresponding cell
            max_spread (float): largest spread of corresponding cell
        """
        assert s.size(0) == self.domain.dimension
        tiks = self.mesh.tiks

        mode = []
        for i, x in enumerate(s):
            assert x <= self.domain.upper_bounds[i]
            assert x >= self.domain.lower_bounds[i]
            j = int(
                (tiks[i] * (x - self.domain.lower_bounds[i]))
                / (self.domain.upper_bounds[i] - self.domain.lower_bounds[i])
            )
            mode.append(j)

        inp = []
        for i in range(self.domain.dimension):
            eps = (self.domain.upper_bounds[i] - self.domain.lower_bounds[i]) / tiks[i]
            inp.append(
                interval([eps * mode[i], eps * mode[i] + eps])
                + self.domain.lower_bounds[i]
            )

        cell = RectangularCell(self.benchmark.f_intervals(inp))
        corners = torch.tensor(cell.get_corners())
        max_spread = cell.get_max_corner_norm(corners, norm)
        max_error_est = cell.get_furthest_corner_dist(
            corners, torch.stack(self.benchmark.f(s)).unsqueeze(1).T, norm
        )
        return max_error_est, max_spread

    def estimate_errors(self, norm, n=10000):
        """Estimate maximum error of automaton.

        Randomly samples n points in the domain and finds the maximum error
        between the automaton and the benchmark out of the n points.

        Args:
            norm (int): norm to use for error calculation
            n (int, optional): number to use. Defaults to 10000.

        Returns:
            max_error_est (float): 
                distance between point and furthest corner in its corresponding cell
            max_spread (float): largest spread of corresponding cell
        """
        S = self.benchmark.domain.generate_data(n)
        max_e = 0
        max_spread = 0
        for s in S:
            e, spread = self.hybridise_point(s, norm)
            max_e = e if e > max_e else max_e
            max_spread = spread if spread > max_spread else max_spread
        return max_e, max_spread


class SimplicialAffine:
    """Simplicial affine Hybridisation"""
    name = "ASM"

    def __init__(self, benchmark, eps) -> None:
        """Initialize a simplicial affine hybridisation

        Args:
            benchmark (benchmarks.Benchmark): 
                benchmark the automaton is to abstract
            eps (_type_): size of simplices
        """        
        self.benchmark = benchmark
        self.eps = eps
        self.mesh = SimplicialMesh(benchmark.domain, eps)

    def get_error(self, norm=2):
        """Calculate error of automaton
        
        Error as defined by the paper:
        https://doi.org/10.1007/s00236-006-0035-7"""
        # Norm not used for this error bound
        return self.get_error_bound()

    def get_error_bound(self):
        max_K = self.get_K_max()
        n = self.benchmark.dimension
        return (max_K * n**2) / (2 * (n + 1) ** 2) * self.mesh.h**2

    def get_K_max(self):
        x = [
            sp.symbols("x" + str(i), real=True) for i in range(self.benchmark.dimension)
        ]
        symbolic_f = sp.Matrix(self.benchmark.f(x))
        max_K = 0
        for fi in symbolic_f:
            k = self.get_Ki(fi, x)
            kl = sp.lambdify(
                x,
                k,
                modules=[
                    {
                        "sin": imath.sin,
                        "cos": imath.cos,
                        "exp": imath.exp,
                        "sqrt": imath.sqrt,
                        "log": imath.log,
                        "Abs": abs,
                    },
                    "math",
                ],
            )
            k_upper = self.get_upper_bound(kl)
            max_K = k_upper if k_upper > max_K else max_K
        return max_K

    def get_upper_bound(self, K):
        d = self.benchmark.domain.as_intervals()
        K_bounds = K(*d)
        if isinstance(K_bounds, (int, float)):
            return K_bounds
        else:
            return K_bounds[0].sup

    def get_Ki(self, fi, x):
        val = 0
        for xi in x:
            for xj in x:
                val += sp.Abs(sp.diff(sp.diff(fi, xi), xj))
        return val


class RectangularMultiAffine:
    """Rectangular multi affine hybridisation"""
    name = "MARM"

    def __init__(self, benchmark, eps) -> None:
        """Initialize a rectangular multi affine hybridisation

        Args:
            benchmark (benchmarks.Benchmark): 
                benchmark the automaton is to abstract
            eps (_type_): size of rectangular cells
        """        
        self.benchmark = benchmark
        self.eps = eps
        self.mesh = MeshGrid(benchmark.domain, eps)

    def get_error(self, norm=2):
        """Calculate error of automaton
        
        Error as defined by the paper:
        https://doi.org/10.1007/s00236-006-0035-7"""
        return self.get_error_bound(norm)

    def get_error_bound(self, norm):
        Mx = self.get_M(norm)
        M_max = self.get_max_M(Mx)
        return M_max / 8 * self.mesh.h**2

    def get_max_M(self, M):
        d = self.benchmark.domain.as_intervals()
        M_bounds = M(*d)
        if isinstance(M_bounds, (int, float)):
            return M_bounds
        elif isinstance(M_bounds, np.ndarray):
            return M_bounds.max()
        else:
            return M_bounds[0].sup

    def get_M(self, norm):
        x = [
            sp.symbols("x" + str(i), real=True) for i in range(self.benchmark.dimension)
        ]
        symbolic_f = sp.Matrix(self.benchmark.f(x))
        J = symbolic_f.jacobian(x)
        if norm == 2:
            norm = "fro"
        M = sum([sp.diff(J, x[i]).norm(ord=norm) for i in range(len(x))])
        lM = sp.lambdify(
            x,
            M,
            modules=[
                {
                    "sin": imath.sin,
                    "cos": imath.cos,
                    "exp": imath.exp,
                    "sqrt": imath.sqrt,
                    "Abs": abs,
                    "Max": fpu.max,
                    "max": fpu.max,
                },
                "numpy",
            ],
        )
        return lM


def analyse_hybridisations(benchmarks: List[str], epsilon: List[float]):
    """Calculate and record error bounds for all hybridisations for a list of benchmarks and sizes.

    Results are recorded to a csv file names "results/hybridisation-results.csv".
    
    Args:
        benchmarks (List[str]): list of benchmark names corresponding to benchmark objects.
        epsilon (List[float]): list of mesh sizes to use
    """    
    header = ["Benchmark", "Method", "h", "Partitions", "Error_1_norm", "Error_2_norm"]
    writer = CSVWriter("results/hybridisation-results.csv", header)
    methods = [SimplicialAffine]
    for benchmark in benchmarks:
        benchmark = read_benchmark(benchmark)
        # benchmark.normalise()
        torch.manual_seed(1)
        for eps in epsilon:
            for method in methods:
                hybridisation = method(benchmark, eps)
                e = []
                for norm in (2,):
                    e_n = hybridisation.get_error(norm=norm)
                    e.append(e_n)
                N_P = hybridisation.mesh.get_total_partitions()
                writer.write_row_to_csv(
                    [benchmark.name, method.name, hybridisation.mesh.h, N_P, *e]
                )


if __name__ == "__main__":
    analyse_hybridisations(
        ["jet", "exp"],
        [1.0, 0.5, 0.25],
    )
    analyse_hybridisations(
        ["steam"],
        [1.0, 0.5]
    )
