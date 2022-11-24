import unittest
from copy import deepcopy
import pickle
import os

import numpy as np
from matplotlib import pyplot as plt
import z3

import polyhedrons
import neural_abstraction as na


class TestGrahamScan(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()


class TestPolyhedron(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        W = [np.array([[1.5, 1], [-0.75, 1], [-1.25, 1], [1, 1]]), np.random.rand(2, 4)]
        b = [np.array([-3, 0.5, -3 / 4, 1 / 4]), np.random.rand(2, 1)]
        act = [0, 1, 0, 1]
        ineq, eq = na.construct_empty_constraint_matrices(W, b)
        ineq, eq = na.get_constraints(ineq, eq, W, b, act)
        cls.P = polyhedrons.Polyhedron(ineq, L=eq)

    def test_contains(self):
        x = [[0.5, 0.5]]
        self.assertTrue(self.P.contains(x))
        x = [[-0.5, -0.5]]
        self.assertFalse(self.P.contains(x))

    def test_pickle(self):
        fi = "tst/test_polyhedron.pkl"
        with open(fi, "wb") as f:
            pickle.dump(self.P, f)
        with open(fi, "rb") as f:
            P = pickle.load(f)
        self.assertTrue(P.__dict__ == self.P.__dict__)
        os.remove(fi)

    def test_intersetion(self):
        P2 = polyhedrons.vertices2polyhedron([[0, 0], [1, 0], [0, 1]])
        P3 = polyhedrons.Polyhedron.intersection(P2, self.P)
        P2.plot()
        self.P.plot(color='black')
        P3.plot(color='b')
        plt.show()

    def test_max_norms(self):
        res = self.P.max_1_norm()
        self.assertAlmostEqual(res, 57/22)
        res = self.P.max_inf_norm()
        self.assertAlmostEqual(res, 39/22)

    def test_to_str(self):
        s = self.P.to_str()
        res = "3 + -1.5 * x0 + -1 * x1 >= 0\n0.5 + -0.75 * x0 + 1 * x1 >= 0\n0.75 + 1.25 * x0 + -1 * x1 >= 0\n0.25 + 1 * x0 + 1 * x1 >= 0"
        self.assertEqual(s, res)


class TestTemplatePolyhedron(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        W = [np.array([[1.5, 1], [-0.75, 1], [-1.25, 1], [1, 1]]), np.random.rand(2, 4)]
        b = [np.array([-3, 0.5, -3 / 4, 1 / 4]), np.random.rand(2, 1)]
        act = [0, 1, 0, 1]
        ineq, eq = na.construct_empty_constraint_matrices(W, b)
        ineq, eq = na.get_constraints(ineq, eq, W, b, act)
        cls.P = polyhedrons.Polyhedron(ineq, L=eq)
        cls.TP = polyhedrons.convert2template(cls.P, polyhedrons.get_2d_template(6))

    def test_contains(self):
        self.TP.plot()
        plt.show()
        x = [[0.5, 0.5]]
        self.assertTrue(self.TP.contains(x))
        x = [[-0.5, -0.5]]
        self.assertFalse(self.TP.contains(x))

    def test_subset(self):
        small_TP = deepcopy(self.TP)
        small_TP.bound = [bound * 0.8 for bound in self.TP.bound]
        self.assertTrue(small_TP.is_subset(self.TP))
        self.assertFalse(self.TP.is_subset(small_TP))

        large_TP = deepcopy(self.TP)
        large_TP.bound = [bound * 1.2 for bound in self.TP.bound]
        self.assertTrue(self.TP.is_subset(large_TP))
        self.assertFalse(large_TP.is_subset(self.TP))

        # Non-strict subset
        self.assertTrue(self.TP.is_subset(self.TP))

    def test_as_smt(self):
        TP_simple = polyhedrons.convert2template(self.P, polyhedrons.get_2d_template(4))
        TP_smt = TP_simple.as_smt([z3.Real("x" + str(i)) for i in range(2)], z3.And)
        self.assertIsInstance(TP_smt, z3.BoolRef)

    def test_box_sample(self):
        x = self.TP.box_sample(100)
        x = np.array(x)
        self.TP.plot()
        plt.plot(x[:,0], x[:, 1], 'x')
        plt.show()

    def test_hitandrun_sample(self):
        x = self.TP.hitandrunsample(100)
        x = np.array(x)
        self.TP.plot()
        plt.plot(x[:,0], x[:, 1], 'x')
        plt.show()

class TestConversion2Template(unittest.TestCase):
    def test_convert2template(self):
        W = [np.array([[1.5, 1], [-0.75, 1], [-1.25, 1], [1, 1]]), np.random.rand(2, 4)]
        b = [np.array([-3, 0.5, -3 / 4, 1 / 4]), np.random.rand(2, 1)]
        act = [0, 1, 0, 1]
        ineq, eq = na.construct_empty_constraint_matrices(W, b)
        ineq, eq = na.get_constraints(ineq, eq, W, b, act)
        P = polyhedrons.Polyhedron(ineq, L=eq)
        template = polyhedrons.get_2d_template(8)
        print(template)
        TP = polyhedrons.convert2template(P, template)
        TP.plot(color="tab:blue")
        P.plot()
        plt.show()


class TestConversion2Polyhedron(unittest.TestCase):
    def test_convert_to_polyhedron(self):
        W = [np.array([[1.5, 1], [-0.75, 1], [-1.25, 1], [1, 1]]), np.random.rand(2, 4)]
        b = [np.array([-3, 0.5, -3 / 4, 1 / 4]), np.random.rand(2, 1)]
        act = [0, 1, 0, 1]
        ineq, eq = na.construct_empty_constraint_matrices(W, b)
        ineq, eq = na.get_constraints(ineq, eq, W, b, act)
        P = polyhedrons.Polyhedron(ineq, L=eq)
        template = polyhedrons.get_2d_template(8)
        print(template)
        TP = polyhedrons.convert2template(P, template)
        P_TP = polyhedrons.convert2polyhedron(TP)
        TP.plot(color="tab:blue")
        P.plot()
        P_TP.plot(color="tab:green")
        plt.show()


class TestVertices2Polyhedron(unittest.TestCase):
    def test_v2p(self):
        vertices = [
            [0, 0],
            [0.5, 0.5],
            [0.5, 0],
        ]
        P = polyhedrons.vertices2polyhedron(vertices)
        P.plot()
        plt.show()


if __name__ == "__main__":
    unittest.main()
