# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Callable, Tuple, List
from matplotlib import pyplot as plt
import numpy as np
import torch
from pandas import read_csv
import seaborn as sns


def show():
    """Convenience function for pyplot.show()"""
    plt.show()


def plot_error_surface(benchmark, abstraction, n=100):
    """Plot the error surface of a neural abstraction.

    Plots the error between the abstraction and the concrete model
    over the domain.

    Args:
        benchmark (benchmarks.Benchmark): 
            benchmark object of concrete model
        abstraction (???): neural abstraction
        n (int, optional): number of data points to plot. Defaults to 100.
    """
    assert benchmark.dimension == 2
    x = np.linspace(
        benchmark.domain.lower_bounds[0] + 0.1,
        benchmark.domain.upper_bounds[0] + 0.1,
        n,
    )
    y = np.linspace(
        benchmark.domain.lower_bounds[1] + 0.1,
        benchmark.domain.upper_bounds[1] + 0.1,
        n,
    )
    xx, yy = np.meshgrid(x, y)
    dxf, dyf = benchmark.f([xx, yy])
    gg = np.vstack([xx.ravel(), yy.ravel()])
    dv = abstraction(torch.tensor(gg).T.float())
    dxn = np.float64(dv[:, 0].detach().numpy().reshape(n, n))
    dyn = np.float64(dv[:, 1].detach().numpy().reshape(n, n))

    dx_e = (dxf - dxn) ** 2
    dy_e = (dyf - dyn) ** 2
    E = dx_e + dy_e
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(xx, yy, E, cmap=plt.cm.coolwarm)
    plt.show()


def plot_benchmark(benchmark):
    """Plot the phase plane of a benchmark."""
    if benchmark.dimension != 2:
        raise ValueError("Plotting for dim > 2 not supported")
    else:
        xr = [benchmark.domain.lower_bounds[0],
              benchmark.domain.upper_bounds[0]]
        yr = [benchmark.domain.lower_bounds[1],
              benchmark.domain.upper_bounds[1]]
        return plot_vector_field(benchmark.f, xr, yr)


def plot_nd_benchmark(benchmark, vars: List[bool]):
    """Plots the phase plane of a >2d system by constraining all 
        but two states to a fixed value (currently zero)

    Args:
        benchmark (benchmarks.Benchmark): 
        Benchmark object of concrete model
        vars (List[bool]): Which variables to plot. Other 
            variables are fixed to zero.

    Raises:
        ValueError: If vars contains more than 2 True values.

    Returns:
        streamplot: plot of the vector field of the benchmark.
    """    
    """
    Plots the phase plane of a >2d system by constraining all but two states to a fixed value (currently zero)
    """
    state_indices = [i for i, B in enumerate(vars) if B is True]
    if len(state_indices) != 2:
        raise ValueError("Can only vary two states")
    xr = [
        benchmark.domain.lower_bounds[state_indices[0]],
        benchmark.domain.upper_bounds[state_indices[0]],
    ]
    yr = [
        benchmark.domain.lower_bounds[state_indices[1]],
        benchmark.domain.upper_bounds[state_indices[1]],
    ]
    xx = np.linspace(xr[0], xr[1], 50)
    yy = np.linspace(yr[0], yr[1], 50)
    XX, YY = np.meshgrid(xx, yy)
    VV = [np.ones_like(XX) * v for v in vars]
    VV[state_indices[0]] = XX
    VV[state_indices[1]] = YY
    FF = benchmark.f(VV)
    dx = FF[state_indices[0]]
    dy = FF[state_indices[1]]
    return plt.streamplot(
        XX,
        YY,
        dx,
        dy,
        linewidth=0.5,
        density=1.0,
        arrowstyle="fancy",
        arrowsize=1.5,
        color="tab:blue",
        maxlength=0.3,
    )


def plot_vector_field(
    f: Callable, xrange: Tuple[float, float], yrange: Tuple[float, float]
):
    """Plots the phase plane of a 2d vector field.

    Args:
        f (Callable): function that returns a 2d vector field (x, y) -> (dx, dy)
        xrange (Tuple[float, float]): lower and upper bound of x
        yrange (Tuple[float, float]): lower and upper bound of y

    Returns:
        streamplot: phase plane of the vector field f
    """
    # Plot phase plane of function f
    # fig = plt.figure()
    ax = plt.gca()
    xx = np.linspace(xrange[0], xrange[1], 50)
    yy = np.linspace(yrange[0], yrange[1], 50)
    XX, YY = np.meshgrid(xx, yy)
    dx, dy = f([XX, YY])
    color = np.sqrt((np.hypot(dx, dy)))
    # color = 10000* np.ones_like(color)
    # plt.quiver(dx, dy)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    for axis in ['top', 'left', 'right', 'bottom']:
        ax.spines[axis].set_linewidth(2)
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])
    

    
    return plt.streamplot(
        XX,
        YY,
        dx,
        dy,
        linewidth=0.8,
        density=1.5,
        arrowstyle="fancy",
        arrowsize=1.5,
        color=color,
        cmap=sns.color_palette("ch:s=0,rot=-.4,l=0.8,d=0", as_cmap=True,)
    )


def plot_nn_vector_field(net, xrange: Tuple[float, float], yrange: Tuple[float, float]):
    """Plots the phase plane of a 2d vector field represented by a neural network.

    Args:
        net (ReLUNet): Neural network that returns a 2d vector field (x, y) -> (dx, dy)
        xrange (Tuple[float, float]): lower and upper bound of x
        yrange (Tuple[float, float]): lower and upper bound of y

    Returns:
        streamplot: plot of phase plane of the vector field
    """
    # Plot phase plane
    # fig = plt.figure()
    ax = plt.gca()
    xx = np.linspace(xrange[0], xrange[1], 50)
    yy = np.linspace(yrange[0], yrange[1], 50)
    XX, YY = np.meshgrid(xx, yy)
    gg = np.vstack([XX.ravel(), YY.ravel()])
    dv = net(torch.tensor(gg).T.float())
    dv = dv.T if dv.shape[0] != 2500 else dv
    dx = np.float64(dv[:, 0].detach().numpy().reshape(50, 50))
    dy = np.float64(dv[:, 1].detach().numpy().reshape(50, 50))
    color = np.sqrt((np.hypot(dx, dy)))
    # color = 10000* np.ones_like(color)
    # plt.quiver(dx, dy)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    for axis in ['top', 'left', 'right', 'bottom']:
        ax.spines[axis].set_linewidth(2)
    
    return plt.streamplot(
        XX,
        YY,
        dx,
        dy,
        linewidth=0.8,
        density=1.5,
        arrowstyle="fancy",
        arrowsize=1.5,
        color=color,
        cmap=sns.color_palette("ch:l=0.8,d=0", as_cmap=True,)
    )


def plot_vector_fields(
    net,
    benchmark,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    name: str = None,
):
    """Overlays plot of benchmark and NN vector fields.

    Args:
        net (ReLUNet): Neural network that returns a 2d vector field (x, y) -> (dx, dy)
        benchmark (_type_):
            benchmark object that represents a 2d vector field (x, y) -> (dx, dy)
        xrange (Tuple[float, float]): lower and upper bound of x
        yrange (Tuple[float, float]): lower and upper bound of y
        name (str, optional): 
            Name for saving to file. If None then plot is shown. Defaults to None.
    """
    fig1 = plot_vector_field(benchmark.f, xrange, yrange)
    fig2 = plot_nn_vector_field(net, xrange, yrange)
    if name:
        plt.savefig("results/plots/" + str(name) + ".pdf", format="pdf")
    else:
        # plot_error_surface(benchmark, net, n=5000)
        plt.show()


def plot_results_table():
    """Results table plot. Deprecated."""
    data = read_csv("results/results-table.csv", skiprows=[0])
    data.insert(1, "Partitions", 2 ** data["Width"].astype(float))
    ax = data.plot(
        "Partitions",
        ["Max SE", "Max SE.1", "Max SE.2", "Max SE.3"],
        label=[r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"],
        ylabel="Max SE",
    )
    plt.yscale("log")
    plt.xscale("log")
    # plt.show()
    plt.savefig("results/plots/Max_square_erros.png")


def plot_results_file(file: str):
    """Plots results from a file. Deprecated."""
    data = read_csv(file)
    ax = data.plot("Width", "Error", marker="x", liexpyle="none")
    plt.show()


if __name__ == "__main__":
    import benchmarks
    from cegis import nn
    b = benchmarks.read_benchmark("lin")
    N = nn.ReluNet(b, [5], 0.1)
    plot_nn_vector_field(N, [-1, 1], [-1, 1])
    show()
