# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Tuple, List
import copy

import numpy as np
import torch
from torch import Tensor

from benchmarks import Benchmark
from cegis.nn import ReconstructedRelu, ReluNet, ScalarReluNet
from cegis.translator import Translator
from cegis.verifier import DRealVerifier, Z3Verifier
from utils import Timeout, check_timeout_loc, vprint

SUCCESS = "S"
FAILURE = "F"
FAILURE_IN_VERIFIER = "FV"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Cegis:
    """Object for initialising CEGIS Algorithm for neural abstractions
    and storing the results"""
    def __init__(
        self, benchmark: Benchmark, error: float, width: List[int], config
    ) -> None:
        """initialise Cegis object

        Args:
            benchmark (Benchmark): dynamical model to be abstracted
            error (float): target error
            width (List[int]): structure of hidden layers of neural network
            config (config.Config): configuration object
        """
        self.benchmark = benchmark
        self.error = [error] * benchmark.dimension if error else error
        if config.scalar:
            self.learner = [
                ScalarReluNet(self.benchmark, width, self.error, config)
                for i in range(benchmark.dimension)
            ]
        else:
            self.learner = [ReluNet(self.benchmark, width, self.error, config)]
        verifier_type = DRealVerifier
        self.x = verifier_type.new_vars(self.benchmark.dimension)
        self.verifier = verifier_type(
            self.x,
            self.benchmark.dimension,
            self.benchmark.get_domain,
            verbose=config.verbose,
        )
        self.translator = Translator(self.x, self.verifier.relu)
        self.config = config

    def synthesise(self) -> Tuple[str, ReluNet]:
        """Run CEGIS algorithm and return the result

        Returns:
            res (str): result of CEGIS algorithm (S, F, FV)
            learner (List[ReLUNet]): Trained neural networks
            error (List[float]): error bounds for each dimension
        """        
        chunk_size = 1 if self.config.scalar else self.benchmark.dimension
        S = self.benchmark.get_data()
        Sdot = torch.tensor(list(map(self.benchmark.f, S)))
        truef = np.array(self.benchmark.f(self.x)).reshape(-1, 1)
        Sdot = list(chunks(Sdot.T, chunk_size))
        truef = list(chunks(truef, chunk_size))
        error = []
        for i in range(len(self.learner)):
            found = False
            learner = self.learner[i]
            # optim = torch.optim.SGD(learner.model.parameters(), lr=0.05, momentum=0.99)
            optim = torch.optim.AdamW(learner.model.parameters(), lr=1e-3)
            sdot = Sdot[i].T
            truefi = truef[i]
            while not found:
                learner_error = learner.learn(S, sdot, optim)
                vprint(learner_error, self.config.verbose)
                self.error = learner_error if learner_error else self.error
                candidate = self.translator.translate(learner)
                found, cex = self.verifier.verify(truefi, candidate, epsilon=self.error)
                if not found:
                    S , sdot = self.augment(S, sdot, cex)
            self.learner[i] = learner
            error.extend(self.error)

        return (SUCCESS, self.learner, error)

    def synthesise_with_timeout(self, t=float("inf")):
        """Attempt to synthesise the abstraction with a timeout.

        Relies on the SIGALRM signal to terminate the synthesis.
        UNIX only.

        Checks if the synthesis did not complete within the timeout. 
        If so, returns as failure with infinite error.

        Args:
            t (int, optional): time before timeout. Defaults to float("inf").

        Returns:
            res (str): result of CEGIS algorithm (S, F, FV)
            learner (List[ReLUNet]): Trained neural networks
            error (List[float]): error bounds for each dimension
        """ 
        if t == float("inf"):
            return self.synthesise()
        else:
            try:
                with Timeout(seconds=t):
                    return self.synthesise()
            except TimeoutError:
                # FV ==> time out in Verifier; F ==> General timeout
                res = FAILURE_IN_VERIFIER if check_timeout_loc() else FAILURE
                net = self.learner
                return (res, net, [torch.inf])

    def augment(self, S: Tensor, Sdot: Tensor, cex: Tensor) -> Tuple[Tensor, Tensor]:
        """adds counterexamples to dataset and returns new dataset

        Args:
            S (Tensor): data of domain
            Sdot (Tensor): data of f(x)
            cex (Tensor): counterexample points in domain 

        Returns:
            Tuple[Tensor, Tensor]: Augmented datasets S and Sdot
        """
        cex = self.sample(cex)
        S = torch.cat((S, cex))
        cex_dot = torch.tensor(list(map(self.benchmark.f, cex)))
        Sdot = torch.cat((Sdot, cex_dot))
        return S, Sdot

    def sample(self, cex: Tensor, n: int = 5) -> Tensor:
        """Augment a point with n samples from nearby.

        Samples from a Gaussian distribution with mean 0 and standard deviation 0.01.

        Args:
            cex (Tensor): data point to sample around
            n (int, optional): number of data points to return. Defaults to 5.

        Returns:
            Tensor: tensor containing original point and n samples
        """
        aug_cex = torch.zeros((1 + n, cex.shape[1]))
        aug_cex[0, :] = cex
        for i in range(1, n + 1):
            aug_cex[i, :] = torch.normal(cex, 0.01)
        return aug_cex

    def synthesise_iteratively(self, t=20, reduction=0.6) -> Tuple[str, ReluNet]:
        """Synthesise the abstraction iteratively, reducing target error on success.

        Each iteration runs CEGIS with a timeout of t seconds. If successful,
        the target error is reduced and the next iteration is run. If the
        timeout occurs, the procedure ends and the most recent successful 
        result is returned.

        Args:
            t (int, optional): 
            timeout for each synthesis iteration. Defaults to 20.
            reduction (float, optional): 
            factor to reduce error by on success. Defaults to 0.6.

        Returns:
            res (str): result of CEGIS algorithm (S, F, FV)
            learner (List[ReLUNet]): Trained neural networks
            error (List[float]): error bounds for each dimension
        """
        reduction = self.config.reduction
        timedout = False
        lowest_successful_error = float("inf")
        while not timedout:
            res, l, error = self.synthesise_with_timeout(t=t)
            timedout = True if res != SUCCESS else False
            if not timedout:
                succ_net = copy.deepcopy(l)
                lowest_successful_error = error
                self.error = [round(reduction * ei, 6) for ei in self.error]
                self.verifier.error = self.error
                for learner in self.learner:
                    learner.error = torch.tensor(self.error)
        if lowest_successful_error == float("inf"):
            res = FAILURE
            succ_net = l
        else:
            res = SUCCESS
        return res, succ_net, lowest_successful_error


class ScalarCegis(Cegis):
    """
    Variant on CEGIS class but learns N networks, one for each dimension of dynamical system (R^n -> R)
    """

    def __init__(
        self, benchmark: Benchmark, error: float, width: List[int], config
    ) -> None:
        super().__init__(benchmark, error, width, config)
        self.learner = [
            ScalarReluNet(
                self.benchmark,
                width,
                self.error,
                padding=config.padding,
                verbose=config.verbose,
            )
            for i in range(benchmark.dimension)
        ]
        self.error = error
        self.verifier.error = self.error

    def synthesise(self) -> Tuple[str, ReconstructedRelu]:
        S = self.benchmark.get_data()
        Sdot = torch.tensor(list(map(self.benchmark.f, S)))
        truef = np.array(self.benchmark.f(self.x)).reshape(-1, 1)

        for i in range(len(self.learner)):
            learner = self.learner[i]
            # optim = torch.optim.Adamax(learner.model.parameters(),)
            optim = torch.optim.Adam(learner.model.parameters(), lr=0.003)
            found = False
            while not found:
                learner.learn(S, Sdot[:, i].reshape(-1, 1), optim)
                candidate = self.translator.translate(learner)
                truef_i = truef[i].reshape(-1, 1) if len(self.learner) > 1 else truef
                found, cex = self.verifier.verify(truef_i, candidate)
                if not found:
                    S, Sdot = self.augment(S, Sdot, cex)
            self.learner[i] = learner
        net = self.reconstruct_net(self.learner)
        return (SUCCESS, net)

    def reconstruct_net(self, learners: Tuple) -> ReconstructedRelu:
        net = ReconstructedRelu(self.benchmark, None, self.error, learners)
        return net

    def synthesise_with_timeout(self, t=float("inf")) -> Tuple[str, ReconstructedRelu]:
        if t == float("inf"):
            return self.synthesise()
        else:
            try:
                with Timeout(seconds=t):
                    return self.synthesise()
            except TimeoutError:
                # FV ==> time out in Verifier; F ==> General timeout
                res = FAILURE_IN_VERIFIER if check_timeout_loc() else FAILURE
                net = self.reconstruct_net(self.learner)
                return (res, net)

    def synthesise_iteratively(
        self, t=20, reduction=0.6
    ) -> Tuple[str, ReconstructedRelu]:
        timedout = False
        lowest_successful_error = float("inf")
        while not timedout:
            res, l = self.synthesise_with_timeout(t=t)
            timedout = True if res != SUCCESS else False
            if not timedout:
                succ_net = copy.deepcopy(l)
                lowest_successful_error = self.error
                self.error = round(reduction * self.error, 6)
                self.verifier.error = self.error
                for learner in self.learner:
                    learner.error = self.error
        if lowest_successful_error == float("inf"):
            res = FAILURE
            succ_net = l
        else:
            succ_net.error = lowest_successful_error
            res = SUCCESS
        return res, succ_net
