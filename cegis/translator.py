# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Callable, List, Tuple

import numpy as np
from torch.nn import Sequential

from cegis.nn import ReluNet


class Translator:
    """Translates a torch neural network representing a dynamical model
    to a symbolic expression.
    """
    def __init__(self, vars: Tuple, relu: Callable) -> None:
        """Initialises the translator.

        Args:
            vars (Tuple): Symbolic variables representing the input.
            relu (Callable): ReLU function for symbolic inputs 
        """
        self.input_vars = np.array(vars).reshape(-1, 1)
        self.relu = relu

    def translate(self, net: ReluNet, dp=8) -> np.ndarray:
        """ Performs the translation of the neural network.

        Args:
            net (ReluNet): net to translate
            dp (int, optional): precision of translated floats. Defaults to 8.

        Returns:
            np.ndarray: vector representing the translated neural network.
        """
        W_in = net.model[0].weight.detach().numpy().round(dp)
        b_in = net.model[0].bias.detach().numpy().reshape(-1, 1).round(dp)
        W_out = net.model[-1].weight.detach().numpy().round(dp)
        b_out = net.model[-1].bias.detach().numpy().reshape(-1, 1).round(dp)

        x = W_in @ self.input_vars + b_in
        x = self.relu(x)
        for i in range(int((len(net.model) - 3) / 2)):
            W = net.model[2 * i + 2].weight.detach().numpy().round(dp)
            b = net.model[2 * i + 2].bias.detach().numpy().reshape(-1, 1).round(dp)
            x = W @ x + b
            x = self.relu(x)
        x = W_out @ x + b_out
        return x
