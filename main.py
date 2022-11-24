# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import time
import warnings 

import torch

from benchmarks import read_benchmark
from cegis.cegis import Cegis, ScalarCegis
from cli import get_config, parse_command
import polyhedrons
from utils import save_net_dict, interpolate_error, get_partitions
from anal import Analyser
from neural_abstraction import NeuralAbstraction
from config import Config

def main(config: Config):
    benchmark = read_benchmark(config.benchmark)
    if config.target_error == 0:
        config.target_error = interpolate_error(
            benchmark.name, get_partitions(benchmark, config.widths, config.scalar))
    c = Cegis(benchmark, config.target_error, config.widths,
                config)
    T = config.timeout_duration if config.timeout else float("inf")

    t0 = time.perf_counter()
    if config.iterative:
        res, net, e = c.synthesise_iteratively(t=T)
    else:
        res, net, e = c.synthesise_with_timeout(t=T)
    t1 = time.perf_counter()
    delta_t = t1 - t0
    if config.save_net:
        nets = net
        for i, N in enumerate(nets):
            save_net_dict(config.fname + "dim=" + str(i), N)
    # benchmark.plotting(net[0])

    NA = NeuralAbstraction(net, e, benchmark)
    print("Learner Timers: {} \n".format(c.learner[0].get_timer()))
    print("Verifier Timers: {} \n".format(c.verifier.get_timer()))
    print("Abstraction Timers: {} \n".format(NA.get_timer()))
    print("The abstraction consists of {} modes".format(len(NA.locations)))
    if "xml" in config.output_type:
        if config.initial:
            XI = polyhedrons.vertices2polyhedron(config.initial)
        else:
            XI = None
        if config.forbidden:
            XU = polyhedrons.vertices2polyhedron(config.forbidden) # Currently unused
        NA.to_xml(config.output_file, bounded_time=config.bounded_time, T=config.time_horizon, initial_state=XI)
    if "csv" in config.output_type:
        a = Analyser(NA)
        a.record(config.output_file, config, res, delta_t)
    if "plot" in config.output_type:
        if benchmark.dimension !=2:
            warnings.warn("Attempted to plot for n-dimensional system")
        else:  
            NA.plot(label=True)
    if "pkl" in config.output_type:
        NA.to_pkl(config.output_file)



if __name__ == "__main__":
    c = get_config()
    torch.set_num_threads(1)
    for i in range(int(c.repeat)):
        c.seed += i
        torch.manual_seed(0 + c.seed)
        main(c)
