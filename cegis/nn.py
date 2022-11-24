# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import OrderedDict, List
from matplotlib import pyplot as plt

import torch
import numpy as np
from torch import Tensor, nn, optim

from utils import timer, Timer
import neural_abstraction as na

T = Timer()

def vprint(m, v: bool):
    """Print first argument if second argument is True."""
    if v:
        print(m)


class ReluNet(nn.Module):
    """torch network of arbitrary size with ReLU activation functions

    Args:
        nn (Module): inherits from torch.nn.module
    """

    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        """Initialise ReLU net object

        Args:
            benchmark (Benchmark): Benchmark object describing model to be abstracted
            width (List[int]): size of hidden layers of network
            error (float): target error (potentially redundant)
            config (_type_): configuration object of program
        """
        super().__init__()
        self.width = width
        self.benchmark = benchmark
        self.loss = nn.MSELoss(reduction="mean")
        self.relu = nn.ReLU
        self.error = torch.tensor(error) if error else error
        self.model = nn.Sequential(self.get_structure(width))
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        """forward method for nn.module"""
        return self.model(x)

    @timer(T)
    def learn(self, S: Tensor, y: Tensor, optimizer: optim.Optimizer):
        """Trains neural abstraction.

        Trains neural abstraction using the corresponding method described by
        the configuration object.

        Args:
            S (Tensor): Training Data
            y (Tensor): _description_
            optimizer (optim.Optimizer): _description_

        """
        if self.config.learning_mode == "error":
            return self.learn_error(S, y, optimizer)
        else:
            return self.learn_min(S, y, optimizer)

    def learn_min(self, S: Tensor, y: Tensor, optimizer: optim.Optimizer) -> None:
        """Trains neural abstraction to a target local minima.

        Performs gradient descent using the optimizer parameter, using stopping criterion from the self.config object.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data
            optimizer (optim.Optimizer): Torch optimizer object

        """
        stop = False
        grad_prev = float("inf")
        interior_points = self.benchmark.domain.check_interior(S)
        grad = []
        grad_grad = []
        l = []
        max_x = []
        max_y = []
        i = 0
        global fig_count 
        fig_count = 0
        while not stop:
            s = 0
            i += 1
            if i % 100 == 1:
                fig = na.NeuralAbstraction((self,), 0,  self.benchmark).plot(show=False)
                plt.savefig(f"plots/({fig_count})_nl2.png")
                fig_count += 1
                plt.clf()
            
            loss = self.loss(self(S), y)
            loss.backward()
            optimizer.step()
            e = abs((self(S) - y)[interior_points]).max(axis=0)[0]
            # max_x.append(e[0].item())
            # max_y.append(e[1].item())

            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                s += p.grad.data.norm(2).item()
            grad.append(s)
            d_grad = s - grad_prev
            grad_grad.append(d_grad)
            grad_prev = s
            l.append(loss.item())
            optimizer.zero_grad()
            # if (d_grad > 5E-4 and s < 1) or i > 21000:
            vprint(loss.item(), self.config.verbose)
            if (
                loss.item() < self.config.loss_stop
                and s < self.config.loss_grad_stop
                and d_grad < self.config.loss_grad_grad_stop
            ):
                stop = True
        e = abs((self(S) - y)[interior_points]).max(axis=0)[0]
        e = (e / 0.8).tolist()
        e = [round(max(ei, 0.005), ndigits=3) for ei in e]
        return e

    def learn_error(self, S: Tensor, y: Tensor, optimizer: optim.Optimizer) -> None:
        """Trains neural abstraction to a target maximum error threshold.

        Performs gradient descent using the optimizer parameter, using error based stopping criterion from the self.config object.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data
            optimizer (optim.Optimizer):  Torch optimizer object
        """
        n_error = float("inf")
        split = int(0.9 * S.shape[0])
        S_train, S_val = S[:split], S[split:]
        y_train, y_val = y[:split], y[split:]

        # print(f'seed: {torch.rand(1)}')
        iter = 0
        titer = 0
        max_train_error_array = []
        mean_train_error_array = []
        max_val_error_array = []
        mean_val_error_array = []
        l_array_train = []
        l_val = []
        interior_points_train = self.benchmark.domain.check_interior(S_train)
        interior_points_val = self.benchmark.domain.check_interior(S_val)
        while n_error > 0:
            iter += 1
            loss = self.loss(self(S), y)

            error = (
                ((self(S_train) - y_train).abs() > 0.80 * self.error)[
                    interior_points_train
                ]
                .any(dim=1)
                .sum()
            )

            n_error = error.sum().item()

            if iter == 100:
                iter = 0
                titer += 1
                e_train = (abs(self(S_train) - y_train))[interior_points_train]
                e_val = (abs(self(S_val) - y_val))[interior_points_val]
                max_train_error_array.append(e_train.max(axis=0)[0].detach().numpy())
                mean_train_error_array.append(e_train.mean(axis=0).detach().numpy())
                max_val_error_array.append(e_val.max(axis=0)[0].detach().numpy())
                mean_val_error_array.append(e_val.mean(axis=0).detach().numpy())
                l_array_train.append(loss.item())
                l_val.append(self.loss(self(S_val), y_val).item())

                vprint(f"ME = {loss}, N_error = {n_error}", self.config.verbose)
            loss.backward()
            # print(optimizer.param_groups[0]['params'][1].grad)
            s = 0
            optimizer.step()
            optimizer.zero_grad()
        try:
            # max_train_error_array = np.array(max_train_error_array)
            # mean_train_error_array = np.array(mean_train_error_array)
            # max_val_error_array = np.array(max_val_error_array)
            # mean_val_error_array = np.array(mean_val_error_array)
            l_array_train = np.array(l_array_train)
            # l_val = np.array(l_val)
            # plt.plot(max_train_error_array[:,0], label='x1-train-max')
            # plt.plot(max_train_error_array[:,1], label='x2-train-max')
            # plt.plot(max_val_error_array[:,0], label='x1-val-max')
            # plt.plot(max_val_error_array[:,1], label='x2-val-max')
            # plt.plot(l_array_train, label="loss_train")
            # plt.plot(l_val, label='loss_val')
            # plt.yscale('log')
            # plt.legend()
            # plt.show()
        except IndexError:
            pass

    def get_structure(self, width: List[int]) -> OrderedDict:
        """returns ordered dictionary of hidden layers with relu activations based on width list

        Args:
            width (List[int]): size of hidden layers of net

        Returns:
            OrderedDict: (label: layer) pairs representing neural network structure in order.
        """
        input_layer = [
            (
                "linear-in",
                nn.Linear(
                    self.benchmark.dimension,
                    width[0],
                ),
            )
        ]
        out = [
            (
                "linear-out",
                nn.Linear(
                    width[-1],
                    self.benchmark.dimension,
                ),
            )
        ]
        relus = [("relu" + str(i + 1), self.relu()) for i in range(len(width))]
        lins = [
            ("linar" + str(i + 1), nn.Linear(width[i], width[i + 1]))
            for i in range(len(width) - 1)
        ]
        z = [None] * (2 * len(width) - 1)
        z[::2] = relus
        z[1::2] = lins
        structure = OrderedDict(input_layer + z + out)
        return structure

    @staticmethod
    def get_timer():
        return T


class ScalarReluNet(ReluNet):
    """Variant Relu Net that only has a single output neuron"""

    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        super().__init__(benchmark, width, error, config)
        self.model = nn.Sequential(self.get_structure(width))

    def get_structure(self, width: List[int]) -> OrderedDict:
        """Returns ordered dictionary of network structure based on width list

        Args:
            width (List[int]): size of hidden layers of net

        Returns:
            OrderedDict: (label: layer) pairs representing neural network structure in order.
        """
        input_layer = [
            (
                "linear-in",
                nn.Linear(
                    self.benchmark.dimension,
                    width[0],
                ),
            )
        ]
        out = [("linear-out", nn.Linear(width[-1], 1))]
        relus = [("relu" + str(i + 1), self.relu()) for i in range(len(width))]
        lins = [
            ("linear" + str(i + 1), nn.Linear(width[i], width[i + 1]))
            for i in range(len(width) - 1)
        ]
        z = [None] * (2 * len(width) - 1)
        z[::2] = relus
        z[1::2] = lins
        structure = OrderedDict(input_layer + z + out)
        return structure


class ReconstructedRelu(nn.Module):
    """Reconstructs ScalarReluNet networks into a single vector-valued function.

    Args:
        nn (nn.module): inherits from torch.nn.Module
    """

    def __init__(
        self,
        scalar_nets,
    ) -> None:
        super().__init__()
        self.scalar_nets = scalar_nets

    def forward(self, x):
        return (
            torch.stack([self.scalar_nets[i](x) for i in range(len(self.scalar_nets))])
            .squeeze()
            .T
        )
