# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Callable, Dict

import dreal
import torch
import z3

from utils import vprint, Timer, timer

T = Timer()


class Verifier:
    def __init__(
        self,
        vars: tuple,
        dimension: int,
        domain: Callable,
        verbose: bool = True,
    ):
        super().__init__()
        self.vars = vars
        self.n = dimension
        self._vars_bounds = inf_bounds_n(dimension)
        self.domain = domain
        self.fncts = self.solver_fncts()
        self.verbose = verbose

    @staticmethod
    def new_vars(n):
        """Example: return [Real('x%d' % i) for i in range(n_vars)]"""
        raise NotImplementedError("")

    @staticmethod
    def solver_fncts() -> Dict[Callable, str]:
        """Example: return {'And': z3.And}"""
        raise NotImplementedError("")

    def new_solver(self):
        """Example: return z3.Solver()"""
        raise NotImplementedError("")

    def is_sat(self, res) -> bool:
        """Example: return res == sat"""
        raise NotImplementedError("")

    def is_unsat(self, res) -> bool:
        """Example: return res == unsat"""
        raise NotImplementedError("")

    def _solver_solve(self, solver, fml):
        """Example: solver.add(fml); return solver.check()"""
        raise NotImplementedError("")

    def _solver_model(self, solver, res):
        """Example: return solver.model()"""
        raise NotImplementedError("")

    def _model_result(self, solver, model, var, idx):
        """Example: return float(model[var[0, 0]].as_fraction())"""
        raise NotImplementedError("")

    @timer(T)
    def verify(self, f, NN, epsilon=None):
        """
        :param V: z3 expr
        :param Vdot: z3 expr
        :return:
                found_lyap: True if V is valid
                C: a list of ctx
        """
        found = False
        fmls = self.get_constraints(f, NN, epsilon=epsilon)
        results = []
        solvers = []

        for iii, condition in enumerate(fmls):
                s = self.new_solver()
                res = self.solve(s, condition)
                results.append(res)
                solvers.append(s)
        assert(len(results) == len(solvers))

        C = []

        if all(self.is_unsat(res) for res in results):
            vprint(["No counterexamples found!"], self.verbose)
            found = True
        else:
            for iii, res in enumerate(results):
                if self.is_sat(res):
                    original_point = self.compute_model(solvers[iii], res)
                    C = original_point

        # s = self.new_solver()

        # res = self.solve(s, fmls)

        # if self.is_unsat(res):
        #     vprint(["No counterexamples found!"], self.verbose)
        #     found = True
        # else:
        #     original_point = self.compute_model(s, res)
        #     C = original_point
        #     # value_in_ctx = self.replace_point(NN, self.vars, original_point.numpy().T)
        #     # print(['NN(ctx) = ', value_in_ctx])

        return found, C

    def get_constraints(self, f, N, epsilon=None, p=2):
        """Returns approximation error constraint using the selected norm.

        If f is a scalar, then the absolute value of the error is returned.

        If epsilon is a list (and f is a vector), then the absolute value for 
        each separate dimension is returned.

        If f is a vector and epsilon is a scalar, then the norm used is determined by p
        (either 2-norm or inf-norm).


        Args:
            f (np.ndarray): array containing symbolic expression for concrete model.
            N (np.ndarray): array containing symbolic expression for neural network.
            epsilon (_type_, optional): Aprroximation error bound. Defaults to None.
            p (int, optional): p-norm. Defaults to 2.

        Returns:
            constr (List): error constrains.
        """
        if f.shape[0] == 1 and not self.n == 1:
            return self.get_absval_constraint(f, N, epsilon=epsilon)
        elif isinstance(epsilon, list):
            return self.get_dimensional_constraints(f, N, epsilon=epsilon)
        elif p == 2:
            return self.get_2norm_constraint(f, N, epsilon=epsilon)
        elif p == float('inf'):
            return self.get_infnorm_constraint(f, N, epsilon=epsilon)

    def get_dimensional_constraints(self, f, N, epsilon=[]):
        _And = self.fncts["And"]
        error = epsilon if epsilon else self.error
        if len(error) != len(f):
            raise ValueError("Incorrect number of error bounds; must match dimension n.")
        constrs = []
        domain = self.domain(self.vars, _And)
        for f_i, N_i, e_i in zip(f, N, error):
            error_norm = abs((f_i - N_i).item()) >= e_i
            constrs.append(_And(error_norm, domain))
        return constrs

    def get_absval_constraint(self, f, N, epsilon=None):
        """Absolute value constraint.

        Returns the absolute value between the neural network and the concrete model.

        Args:
            f (np.ndarray): 
                symbolic expression for concrete model - 1d array.
            N (np.ndarray): 
                symbolic expression for neural network - 1d array.
            epsilon (float, optional): error. Defaults to None.

        Returns:
            constr list: constraint
        """
        _And = self.fncts["And"]
        error = epsilon if epsilon else self.error
        error_norm = abs((f - N).item()) >= error
        domain = self.domain(self.vars, _And)
        return [_And(error_norm, domain)]

    def get_2norm_constraint(self, f, N, epsilon=None):
        """2-norm value constraint.

        Returns the 2-norm between the neural network and the concrete model.

        Args:
            f (np.ndarray): symbolic expression for concrete model
            N (np.ndarray): symbolic expression for neural network
            epsilon (float, optional): error. Defaults to None.

        Returns:
            constr list: constraint
        """
        _And = self.fncts["And"]
        _Or = self.fncts["Or"]
        error = epsilon if epsilon else self.error
        error_norm = (((f - N)) ** 2).sum() >= error
        # diff = f - N
        # error = _Or(*[diff[i,0]**2  >= self.error for i in range(len(f))])
        domain = self.domain(self.vars, _And)
        return [_And(error_norm, domain)]

    def get_infnorm_constraint(self, f, N, epsilon=None):
        """inf-norm constraint.

        Returns the infinity norm between the neural network and the concrete model.

        Args:
            f (np.ndarray): symbolic expression for concrete model
            N (np.ndarray): symbolic expression for neural network
            epsilon (float, optional): error. Defaults to None.

        Returns:
            constr list: constraint
        """
        _And = self.fncts["And"]
        _Or = self.fncts["Or"]
        _If = self.fncts["If"]
        _Not = self.fncts["Not"]
        error = epsilon if epsilon else self.error
        value = (f - N).squeeze().tolist()
        abs_value = [_If(v > 0, v, -v) for v in value]
        norm_error = [val <= error for val in abs_value]
        constraint = _And(*norm_error)
        domain = self.domain(self.vars, _And)
        return [_Not(_Or(constraint, _Not(domain)))]

    def solve(self, solver, fml):
        """
        :param fml:
        :param solver: z3 solver
        :return:
                res: sat if found ctx
        """
        res = self._solver_solve(solver, fml)

        return res

    def compute_model(self, solver, res):
        """
        :param solver: z3 solver
        :return: tensor containing single ctx
        """
        model = self._solver_model(solver, res)
        vprint("Counterexample Found: {}".format(model), self.verbose)
        temp = []
        for i, x in enumerate(self.vars):
            temp += [self._model_result(solver, model, x, i)]

        original_point = torch.tensor(temp)
        return original_point[None, :]

    def in_bounds(self, var, n):
        left, right = self._vars_bounds[var]
        return left < n < right

    @staticmethod
    def get_timer():
        return T


class Z3Verifier(Verifier):
    @staticmethod
    def new_vars(n):
        return [z3.Real("x%d" % i) for i in range(n)]

    def new_solver(self):
        return z3.Solver()

    @staticmethod
    def relu(x):
        """ReLU function for symbolic z3 variables."""
        y = x.copy()
        _If = z3.If
        for idx in range(len(y)):
            y[idx, 0] = z3.simplify(_If(y[idx, 0] > 0, y[idx, 0], 0))
        return y

    @staticmethod
    def leakyrelu(x):
        alpha = 0.01
        y = x.copy()
        _If = z3.If
        for idx in range(len(y)):
            y[idx, 0] = z3.simplify(_If(y[idx, 0] > 0, y[idx, 0], alpha * y[idx, 0]))

    @staticmethod
    def identity(x):
        """Identity function for symbolic z3 variables."""
        return x

    @staticmethod
    def zero(x):
        """Zero function for symbolic z3 variables."""
        return 0.0

    @staticmethod
    def solver_fncts() -> Dict[Callable, str]:
        return {"And": z3.And, "Or": z3.Or, "If": z3.If, "Not": z3.Not}

    def is_sat(self, res) -> bool:
        return res == z3.sat

    def is_unsat(self, res) -> bool:
        return res == z3.unsat

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return z3_replacements(expr, ver_vars, point)

    def _solver_solve(self, solver, fml):
        solver.add(fml)
        return solver.check()

    def _solver_model(self, solver, res):
        return solver.model()

    def _model_result(self, solver, model, x, i):
        try:
            return float(model[x].as_fraction())
        except AttributeError:
            return float(model[x].approx(10).as_fraction())
        except z3.Z3Exception:
            try:
                return float(model[x[0, 0]].as_fraction())
            except AttributeError:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())

    def __init__(
        self,
        vars: tuple,
        dimension: int,
        domain: Callable,
        verbose: bool = True,
    ):
        super().__init__(vars, dimension, domain, verbose=verbose)


class DRealVerifier(Verifier):
    @staticmethod
    def new_vars(n):
        return [dreal.Variable("x%d" % i) for i in range(n)]

    def new_solver(self):
        return None

    @staticmethod
    def relu(x): 
        """ReLU function for symbolic DReal variables."""
        y = x.copy()
        _max = dreal.Max
        for idx in range(len(y)):
            y[idx, 0] = _max(y[idx, 0], 0)
        return y
    
    @staticmethod
    def leakyrelu(x):
        y = x.copy()
        alpha = 0.01
        _If = dreal.If
        for idx in range(len(y)):
            y[idx, 0] = _If(y[idx, 0] > 0, y[idx, 0], alpha * y[idx, 0])
        return y

    @staticmethod
    def identity(x):
        """Identity function for symbolic DReal variables."""
        return x

    @staticmethod
    def zero(x):
        """Zero function for symbolic DReal variables."""
        return 0.0

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {
            "sin": dreal.sin,
            "cos": dreal.cos,
            "exp": dreal.exp,
            "And": dreal.And,
            "Or": dreal.Or,
            "If": dreal.if_then_else,
            "Not": dreal.Not,
        }

    def is_sat(self, res) -> bool:
        return isinstance(res, dreal.Box)

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return dreal_replacements(expr, ver_vars, point)

    def is_unsat(self, res) -> bool:
        # int(str("x0")) = 0
        bounds_not_ok = not self.within_bounds(res)
        return res is None or bounds_not_ok

    def within_bounds(self, res) -> bool:
        return isinstance(res, dreal.Box) and all(
            self.in_bounds(int(str(x)[1:]), interval.mid())
            for x, interval in res.items()
        )

    def _solver_solve(self, solver, fml):
        # c = dreal.Config()
        res = dreal.CheckSatisfiability(fml, 0.0001)
        if self.is_sat(res) and not self.within_bounds(res):
            new_bound = self.optional_configs.get(
                dreal.VerifierConfig.DREAL_SECOND_CHANCE_BOUND.k,
                dreal.VerifierConfig.DREAL_SECOND_CHANCE_BOUND.v,
            )
            fml = dreal.And(
                fml, *(dreal.And(x < new_bound, x > -new_bound) for x in self.xs)
            )
            res = dreal.CheckSatisfiability(fml, 0.01)
        return res

    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())

    def __init__(
        self,
        vars: tuple,
        dimension: int,
        domain: Callable,
        verbose: bool = True,
    ):
        super().__init__(vars, dimension, domain, verbose=verbose)


def z3_replacements(expr, z3_vars, ctx):
    """
    :param expr: z3 expr
    :param z3_vars: z3 vars, matrix
    :param ctx: matrix of numerical values
    :return: value of V, Vdot in ctx
    """
    replacements = []
    for i in range(len(z3_vars)):
        try:
            replacements += [(z3_vars[i, 0], z3.RealVal(ctx[i, 0]))]
        except TypeError:
            replacements += [(z3_vars[i], z3.RealVal(ctx[i, 0]))]

    replaced = z3.substitute(expr, replacements)

    return z3.simplify(replaced)


def dreal_replacements(expr, dr_vars, ctx):
    """
    :param expr: dreal expr
    :param dr_vars: dreal vars, matrix
    :param ctx: matrix of numerical values
    :return: value of V, Vdot in ctx
    """
    try:
        replacements = {dr_vars[i, 0]: ctx[i, 0] for i in range(len(dr_vars))}
    except TypeError:
        replacements = {dr_vars[i]: ctx[i, 0] for i in range(len(dr_vars))}

    return expr.Substitute(replacements)


inf = 1e300
inf_bounds = [-inf, inf]


def inf_bounds_n(n):
    return [inf_bounds] * n
