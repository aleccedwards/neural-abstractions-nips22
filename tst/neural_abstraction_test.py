import os
import unittest
import pickle

import numpy as np
from matplotlib import pyplot as plt
import torch

import neural_abstraction as na
import reach
import cegis.nn as nn
import benchmarks
import domains


class testIncrement(unittest.TestCase):
    def test_increment_to_end(self):
        act = [0] * 3
        res = False
        for i in range(2 ** len(act) - 1):
            res = na.increment(act)
            self.assertFalse(res)
        res = na.increment(act)
        self.assertTrue(res)


class testGetDomainConstraints(unittest.TestCase):
    def test_get_domain_constraints2D(self):
        R = domains.Rectangle([-1, -1], [1, 1])
        constr = na.get_domain_constraints(R, 2)
        correct_result = np.array(
            [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
        )
        np.testing.assert_equal(constr, correct_result)


class testGetConstraintMatrices1Layer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1)
        cls.bench = benchmarks.read_benchmark("lin")
        cls.widths = [5, 3]
        nets = [nn.ReluNet(cls.bench, cls.widths, None, None)]
        cls.W, cls.b = na.get_Wb(nets)

    def test_construct_empty_constraint_matrices(self):
        ineq, eq = na.construct_empty_constraint_matrices(self.W, self.b)
        ineq_shape_correct = [
            sum(self.widths),
            sum(self.widths[:-1]) + self.bench.dimension + 1,
        ]
        eq_shape_correct = [
            sum(self.widths[:-1]),
            sum(self.widths[:-1]) + self.bench.dimension + 1,
        ]
        np.testing.assert_equal(ineq.shape, ineq_shape_correct)
        np.testing.assert_equal(eq.shape, eq_shape_correct)

    def test_get_constraints(self):
        ineq, eq = na.construct_empty_constraint_matrices(self.W, self.b)
        act = [1, 0, 0] * 5
        ineq, eq = na.get_constraints(ineq, eq, self.W, self.b, act)
        correct_ineq = np.array(
            [
                [
                    0.19612379,
                    0.36434609,
                    -0.3121016,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.0348828,
                    0.1370808,
                    -0.33189401,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.2582553,
                    0.66569638,
                    -0.42406419,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.2755602,
                    -0.1454698,
                    0.3597362,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0515543,
                    -0.0982999,
                    0.0865811,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -0.19138931,
                    0.0,
                    0.0,
                    0.1929014,
                    0.14330889,
                    -0.0214133,
                    -0.26659641,
                    -0.243077,
                ],
                [
                    -0.2078253,
                    0.0,
                    0.0,
                    -0.43717399,
                    0.27723691,
                    0.1249353,
                    0.42420691,
                    0.29518771,
                ],
                [
                    -0.43881401,
                    0.0,
                    0.0,
                    0.40747321,
                    0.42521441,
                    0.2157055,
                    -0.3927035,
                    0.0744919,
                ],
            ]
        )
        correct_eq = np.array(
            [
                [
                    -0.19612379,
                    -0.36434609,
                    0.3121016,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.2755602,
                    0.1454698,
                    -0.3597362,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )
        np.testing.assert_allclose(ineq, correct_ineq, atol=1e-6)
        np.testing.assert_allclose(eq, correct_eq, atol=1e-6)


class testNeuralAbstraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = benchmarks.read_benchmark("lin")
        cls.widths = [5, 3]
        nets = [nn.ReluNet(cls.bench, cls.widths, None, None)]
        cls.W, cls.b = na.get_Wb(nets)
        cls.na = na.NeuralAbstraction(nets, 0, cls.bench)

    def test_pickle(self):
        fi = "tst/test_na.pkl"
        with open(fi, "wb") as f:
            pickle.dump(self.na, f)
        NA = pickle.load(open(fi, "rb"))
        os.remove(fi)

class testModeNoDisturbance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(2)
        cls.bench = benchmarks.read_benchmark("lin")
        cls.widths = [5, 3]
        nets = [nn.ReluNet(cls.bench, cls.widths, None, None)]
        cls.W, cls.b = na.get_Wb(nets)
        cls.na = na.NeuralAbstraction(nets, 0, cls.bench)
        # cls.na.plot(label=True)
        key = "5"
        activation = cls.na.locations[key]
        W, b = na.get_Wb(nets)
        cls.mode = na.Mode(cls.na.flows[key], cls.na.invariants[key])

    def test_check_containment(self):
        x = [0.6, 0.6]
        self.assertFalse(self.mode.contains(x))
        x = [0.2, 0]
        self.assertTrue(self.mode.contains(x))

    def test_flow(self):
        # Drift test
        A, c = self.na.flows["5"]
        x = [0, 0]
        y = self.mode.flow(x)
        self.assertEqual(y[0], c[0])
        self.assertEqual(y[1], c[1])
        x = [0.4, 0.1]
        y = self.mode.flow(x)
        self.assertEqual(y[0], A[0, 0] * x[0] + A[0, 1] * x[1] + c[0])
        self.assertEqual(y[1], A[1, 0] * x[0] + A[1, 1] * x[1] + c[1])

    @unittest.skip("Visual plot test")
    def test_integrate(self):
        I = reach.ScipyIntegrator(self.mode.flow)
        x = [0.4, 0.1]
        for i in range(10):
            res = I.integrate(1, x)
            Y = res.y
            plt.plot(Y[0, :], Y[1, :])
        plt.show()


class testModeDisturbance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(2)
        cls.bench = benchmarks.read_benchmark("lin")
        cls.widths = [5, 3]
        nets = [nn.ReluNet(cls.bench, cls.widths, None, None)]
        cls.W, cls.b = na.get_Wb(nets)
        cls.na = na.NeuralAbstraction(nets, 0, cls.bench)
        # cls.na.plot(label=True)
        key = "5"
        activation = cls.na.locations[key]
        W, b = na.get_Wb(nets)
        cls.mode = na.Mode(cls.na.flows[key], cls.na.invariants[key], disturbance=[0.02, 0.02])

    def test_flow(self):
        A, c = self.na.flows["5"]
        x = [0, 0]
        y = self.mode.flow(x)
        self.assertTrue(y[0] - c[0] <= 0.1)
        self.assertTrue(y[1] - c[1] <= 0.1)

    @unittest.skip("Visual plot test")
    def test_integrate(self):
        I = reach.ScipyIntegrator(self.mode.flow)
        x = [0.4, 0.1]
        for i in range(10):
            res = I.integrate(1, x)
            Y = res.y
            plt.plot(Y[0, :], Y[1, :])
        plt.show()


if __name__ == "__main__":
    unittest.main()
