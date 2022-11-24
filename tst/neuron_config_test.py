# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest
from cli import get_default_config
import cegis.nn as nn
from benchmarks import Linear, NP3
import torch
import numpy as np
import neural_abstraction as na

CONFIG = get_default_config()


class test2DSingleLayerScalar(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 2

        self.net = [
            nn.ScalarReluNet(Linear(), [2], 0.0, CONFIG) for i in range(self.dim)
        ]
        with torch.no_grad():
            self.net[0].model[0].weight = torch.nn.Parameter(
                torch.tensor([[1.5, 1], [-0.75, 1]])
            )
            self.net[0].model[0].bias = torch.nn.Parameter(torch.tensor([-3, 0.5]))
            self.net[1].model[0].weight = torch.nn.Parameter(
                torch.tensor([[-1.25, 1], [1, 1]])
            )
            self.net[1].model[0].bias = torch.nn.Parameter(torch.tensor([-0.75, 0.25]))
        # N = na.NeuralAbstraction(self.net, 0.0, Linear())
        # N.plot()

    def test_halfspace_config(self):
        W, b = na.get_Wb(self.net)
        W1, b1 = W[0], b[0]
        domain_const = na.get_domain_constraints(Linear().domain, self.dim)
        inactive_hp = na.check_fixed_hyperplanes(self.dim, domain_const, W1, b1)
        correct_res = {0: 0}  # First hyperplane should be fixed inactive
        self.assertDictEqual(inactive_hp, correct_res)

    def test_enumerate_activations(self):
        W, b = na.get_Wb(self.net)
        configs = na.get_active_configurations(Linear().domain, self.dim, W, b)
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(configs, correct_configs)

    def test_explore_tree(self):
        W, b = na.get_Wb(self.net)
        tree = na.NeuronTree(W, b, Linear().domain)
        tree.explore_tree()
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(tree.configs, correct_configs)

    def test_allsat(self):
        W, b = na.get_Wb(self.net)
        configs = na.all_sat(W, b, Linear().domain)
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(configs, correct_configs)


class test2DSingleLayerFullyConnected(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 2
        self.net = [nn.ReluNet(Linear(), [4], 0.0, CONFIG)]
        with torch.no_grad():
            self.net[0].model[0].weight = torch.nn.Parameter(
                torch.tensor([[1.5, 1], [-0.75, 1], [-1.25, 1], [1, 1]])
            )
            self.net[0].model[0].bias = torch.nn.Parameter(
                torch.tensor([-3, 0.5, -0.75, 0.25])
            )

    def test_halfspace_config(self):
        W, b = na.get_Wb(self.net)
        W1, b1 = W[0], b[0]
        domain_const = na.get_domain_constraints(Linear().domain, self.dim)
        inactive_hp = na.check_fixed_hyperplanes(self.dim, domain_const, W1, b1)
        correct_res = {0: 0}  # First hyperplane should be fixed inactive
        self.assertDictEqual(inactive_hp, correct_res)

    def test_enumerate_activations(self):
        W, b = na.get_Wb(self.net)
        configs = na.get_active_configurations(Linear().domain, self.dim, W, b)
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(configs, correct_configs)

    def test_explore_tree(self):
        W, b = na.get_Wb(self.net)
        tree = na.NeuronTree(W, b, Linear().domain)
        tree.explore_tree()
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(tree.configs, correct_configs)

    def test_allsat(self):
        W, b = na.get_Wb(self.net)
        configs = na.all_sat(W, b, Linear().domain)
        correct_configs = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ]
        self.assertCountEqual(configs, correct_configs)


class test2DDeepScalar(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 2
        self.net = [nn.ScalarReluNet(Linear(), [2, 2], 0.0, CONFIG)
                    for i in range(self.dim)]


class test2DDeepFullyConnected(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 2
        self.net = [nn.ReluNet(Linear(), [3, 2], 0.0, CONFIG)]
        with torch.no_grad():
            self.net[0].model[0].weight = torch.nn.Parameter(
                torch.tensor([[1.5, 1], [-0.75, 1], [-1.25, 1]])
            )
            self.net[0].model[0].bias = torch.nn.Parameter(
                torch.tensor(
                    [
                        -3,
                        0.5,
                        -0.75,
                    ]
                )
            )
            self.net[0].model[2].weight = torch.nn.Parameter(
                torch.tensor([[0.6, 0.7, 1.4], [-0.1, 2, 1.7]])
            )
            self.net[0].model[2].bias = torch.nn.Parameter(torch.tensor([0.2, 2.0]))

    def test_enumerate_activations(self):
        W, b = na.get_Wb(self.net)
        configs = na.get_active_configurations(Linear().domain, self.dim, W, b)
        print("enumerate {}".format(configs))

    def test_explore_tree(self):
        W, b = na.get_Wb(self.net)
        tree = na.NeuronTree(W, b, Linear().domain)
        tree.explore_tree()
        print("tree: {} ".format(tree.configs))

    def test_allsat(self):
        W, b = na.get_Wb(self.net)
        configs = na.all_sat(W, b, Linear().domain)
        print("allsat: {}".format(configs))
        # self.assertCountEqual(configs, correct_configs)


class test3DSingleLayarScalar(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 3
        self.net = [nn.ScalarReluNet(NP3(), [1], 0.0, CONFIG)
                    for i in range(self.dim)]
        with torch.no_grad():
            self.net[0].model[0].weight = torch.nn.Parameter(
                torch.tensor([[-1.0, -1, -1]])
            )
            self.net[0].model[0].bias = torch.nn.Parameter(torch.tensor([4.0]))
            self.net[1].model[0].weight = torch.nn.Parameter(
                torch.tensor([[-1.0, -1, -1]])
            )
            self.net[1].model[0].bias = torch.nn.Parameter(torch.tensor([2.0]))
            self.net[2].model[0].weight = torch.nn.Parameter(
                torch.tensor([[-1.0, 1, -1]])
            )
            self.net[2].model[0].bias = torch.nn.Parameter(torch.tensor([1.0]))

    def test_halfspace_config(self):
        # pass
        W1 = [net.model[0].weight.detach().numpy() for net in self.net]
        b1 = [net.model[0].bias.detach().numpy() for net in self.net]
        W1 = np.concatenate(W1)
        b1 = np.concatenate(b1)
        domain_const = na.get_domain_constraints(NP3().domain, self.dim)
        inactive_hp = na.check_fixed_hyperplanes(self.dim, domain_const, W1, b1)
        correct_res = {0: 1}  # First hyperplane should be fixed active
        self.assertDictEqual(inactive_hp, correct_res)

    def test_enumerate_activations(self):
        W, b = na.get_Wb(self.net)
        configs = na.get_active_configurations(NP3().domain, self.dim, W, b)
        # I think the correct configs are this but double check
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(correct_configs, configs)

    def test_explore_tree(self):
        W, b = na.get_Wb(self.net)
        tree = na.NeuronTree(W, b, NP3().domain)
        tree.explore_tree()
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(correct_configs, tree.configs)

    def test_allsat(self):
        W, b = na.get_Wb(self.net)
        configs = na.all_sat(W, b, NP3().domain)
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(configs, correct_configs)


class test3DSingleLayerFullyConnected(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 3
        self.net = [nn.ReluNet(NP3(), [3], 0.0, CONFIG)]
        with torch.no_grad():
            self.net[0].model[0].weight = torch.nn.Parameter(
                torch.tensor([[-1.0, -1, -1], [-1.0, -1, -1], [-1.0, 1, -1]])
            )
            self.net[0].model[0].bias = torch.nn.Parameter(
                torch.tensor([4.0, 2.0, 1.0])
            )

    def test_halfspace_config(self):
        # pass
        W1 = [net.model[0].weight.detach().numpy() for net in self.net]
        b1 = [net.model[0].bias.detach().numpy() for net in self.net]
        W1 = np.concatenate(W1)
        b1 = np.concatenate(b1)
        domain_const = na.get_domain_constraints(NP3().domain, self.dim)
        inactive_hp = na.check_fixed_hyperplanes(self.dim, domain_const, W1, b1)
        correct_res = {0: 1}  # First hyperplane should be fixed active
        self.assertDictEqual(inactive_hp, correct_res)

    def test_enumerate_activations(self):
        W, b = na.get_Wb(self.net)
        configs = na.get_active_configurations(NP3().domain, self.dim, W, b)
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(correct_configs, configs)

    def test_explore_tree(self):
        W, b = na.get_Wb(self.net)
        tree = na.NeuronTree(W, b, NP3().domain)
        tree.explore_tree()
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(correct_configs, tree.configs)

    def test_allsat(self):
        W, b = na.get_Wb(self.net)
        configs = na.all_sat(W, b, NP3().domain)
        correct_configs = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.assertCountEqual(configs, correct_configs)


class testConsistency(unittest.TestCase):
    def test(self):
        torch.manual_seed(1)
        np.random.seed(4)  # BUg when seed = 5, i =8
        max_layers = 3
        max_neurons = 10
        ndim = 2   
        n_trials = 3 
        for i in range(n_trials):
            n_layers = np.random.randint(1, max_layers + 1)
            n_neurons = np.random.randint(1, max_neurons, n_layers).tolist()
            net = [nn.ReluNet(Linear(), n_neurons, 0.0, CONFIG)]
            W, b = na.get_Wb(net)
            d = Linear().domain
            tree = na.NeuronTree(W, b, d)
            enum_configs = na.get_active_configurations(d, ndim, W, b)
            tree_configs = tree.explore_tree()
            allsat_configs = na.all_sat(W, b, d)
            self.assertCountEqual(enum_configs, allsat_configs)
            self.assertCountEqual(enum_configs, tree_configs)


if __name__ == "__main__":
    torch.manual_seed(1)
    unittest.main()
