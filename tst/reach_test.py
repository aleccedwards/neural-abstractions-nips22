import unittest

import torch
import numpy as np
import cdd

import reach
import neural_abstraction as na
import polyhedrons

import cegis.nn as nn
import cegis.cegis as cegis
import benchmarks
from cli import get_default_config


class TestEulerIntegrator(unittest.TestCase):
    def setUp(self) -> None:
        def f(v):
            x, y = v
            return [x**2 + y, y**3 - x**2]

        self.T = 0.05
        self.euler_integrator = reach.EulerIntegrator(f, self.T)

    def test_euler_integrator(self):
        x = [0.1, 0.1]
        res = self.euler_integrator.integrate(x)
        self.assertTrue(isinstance(res, list))
        self.assertAlmostEqual(res[0], 0.1 + self.T * (0.1**2 + 0.1), places=4)
        self.assertAlmostEqual(res[1], 0.1 + self.T * (0.1**3 - 0.1**2), places=4)


class TestScipyIntegrator(unittest.TestCase):
    def setUp(self) -> None:
        def f(v):
            x, y = v
            return [x**2 + y, y**3 - x**2]

        self.euler_integrator = reach.ScipyIntegrator(f)

    def test_scipy_integrator(self):
        x = [0.1, 0.1]
        res = self.euler_integrator.integrate(1.0, x)
        t = res.t
        y = res.y


class TestEstimateTimeBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(2)
        cls.bench = benchmarks.read_benchmark("lin")
        cls.widths = [5, 3]
        nets = [nn.ReluNet(cls.bench, cls.widths, None, None)]
        cls.W, cls.b = na.get_Wb(nets)
        cls.na = na.NeuralAbstraction(nets, 0, cls.bench)
        # cls.na.plot(label=True)
        activation = cls.na.locations["5"]
        cls.P = cls.na.invariants["5"]
        cls.mode = cls.na.modes["5"]

    def test_estimate_time_bounds(self):
        TP = polyhedrons.convert2template(self.P, polyhedrons.get_2d_template(4))
        res = reach.estimate_time_bounds(self.na, self.mode, TP)

    def test_estimate_lower_bound(self):
        S = np.array(cdd.Matrix(self.P.V, number_type="float"))[:, 1:]
        t, lP = reach.estimate_lower_time_bound(self.na, self.mode, S)
        self.assertIsInstance(t, float)
        self.assertIsInstance(lP, set)
        self.assertTrue(lP)  # Check set is non-empty

    def test_estimate_upper_bound(self):
        S = np.array(cdd.Matrix(self.P.V, number_type="float"))[:, 1:]
        t, lP = reach.estimate_upper_time_bound(self.na, self.mode, S)
        self.assertIsInstance(t, float)
        self.assertIsInstance(lP, set)
        self.assertTrue(lP)  # Check set is non-empty


class TestReach(unittest.TestCase):
    ## Regression Test
    @classmethod
    def setUpClass(cls) -> None:
        config = get_default_config()
        config.benchmark = "flipflop"
        config.widths = [10, 10]
        config.verbose = False
        benchmark = benchmarks.read_benchmark(config.benchmark)
        c = cegis.Cegis(benchmark, config.target_error, config.widths, config)

        cls.res, net, e = c.synthesise_with_timeout()
        cls.NA = na.NeuralAbstraction(net, e, benchmark)
        cls.NA.plot(label=True)

    def test(self):
        print(self.res)
        self.assertTrue(self.res=="S")


if __name__ == "__main__":
    unittest.main()
