# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations
import pickle
import queue
from math import copysign
import itertools
import copy
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
# import gurobipy as gp
import numpy as np
import torch
import z3

from cegis import nn, verifier
from utils import timer, Timer
import polyhedrons
from utils import decimal_from_fraction
import sysplot

# env = gp.Env(empty=True)
# env.setParam('OutputFlag', 0)
# env.start()
T = Timer()


def get_fully_connected_transitions(self):
    """Returns a list of transitions for a fully connected automaton"""
    return itertools.permutations(self.locations.keys(), 2)


def indent(elem: ET, level=0):
    """In-place pretty print of XML element

    Args:
        elem (ElementTree): XML element to pretty print
        level (int, optional): base indent level. Defaults to 0.
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def increment(act: list) -> bool:
    """Increments the activation vector

    Args:
        act (List):
            bitvector representing activation configuration

    Returns:
        bool:
            True if increment reaches list of zeros, else None 
    """
    for i in range(len(act)):
        if act[i] == 0:
            act[i] = 1
            return False
        else:
            act[i] = 0
    return True


def get_domain_constraints(domain, ndim, nvars=None):
    """Returns a list of constraints for a domain

    Generates array corresponding to constraints for box shaped domain
    of form [b -A] where b is the bounds in each support direction 
    given by A


    Args:
        domain (Rectangle): Hyperrectangle domain
        ndim (_type_): number of dimensions
        nvars (_type_, optional): 
        number of unique variables (if different to nvars). Defaults to None.

    Returns:
        np.ndarray: array of constraints
    """
    if nvars == None:
        nvars = ndim
    domain_constr = []
    for i in range(ndim):
        domain_constr.append([abs(domain.lower_bounds[i]), *[copysign(
            1, -domain.lower_bounds[i]) if j == i else 0 for j in range(ndim)]])
        domain_constr.append([abs(domain.upper_bounds[i]), *[copysign(
            1, -domain.upper_bounds[i]) if j == i else 0 for j in range(ndim)]])
    domain_constr = np.array(domain_constr)
    full_array = np.zeros((domain_constr.shape[0], nvars + 1))
    full_array[:, 0:ndim + 1] = domain_constr
    return full_array


def construct_empty_constraint_matrices(W, b):
    """Constructs empty matrices of correct size for contsrains

    Args:
        W (List[torch.Tensor]): List of weight matrices
        b (List[torch.Tensor]): List of bias vectors

    Returns:
        ineq (np.ndarray): inequality constraint matrix
        eq (np.ndarray): equality constraint matrix
    """    
    n_dim = W[0].shape[1]
    n_vars = n_dim + sum([W[i].shape[0] for i in range(len(W) - 2)])
    n_ineq_constr = sum([W[i].shape[0] for i in range(len(W) - 1)])
    n_eq_constr = sum(W[i].shape[0] for i in range(len(W) - 2))
    ineq = np.zeros((n_ineq_constr, 1 + n_vars))
    if n_eq_constr > 0:
        eq = np.zeros((n_eq_constr, 1 + n_vars))
    else:
        eq = None
    return ineq, eq


def get_constraints(ineq, eq, W, b, act):
    """Returns constraint matrices for a given activation vector

    Args:
        ineq (np.ndarray): empty inequality constraint matrix
        eq (np.ndarray): empty equality constraint matrix
        W (List(np.ndarray)): weight matrices
        b (List(np.ndarray)): bias vectors
        act (List): activation configuration

    Returns:
        ineq (np.ndarray): inequality constraint matrix
        eq (np.ndarray): equality constraint matrix
    """
    k = len(W)
    h_tot = 0
    var_tot = 0
    for i in range(k - 1):
        # For all hidden layers (and not output layer)
        Wi, bi = W[i], b[i]
        hi = Wi.shape[0]  # Number of neurons in layer
        ni = Wi.shape[1]  # Number of vars in layer (or neurons in prev layer)
        activation_i = act[h_tot: h_tot + hi]
        diag_a = np.diag([2 * a - 1 for a in activation_i])
        # Net is Wx + b >= 0   (including diags to switch inactive sign)
        # Cdd wants [c -A] from Ax <= 0
        # Therefore c = b, -W = -A
        ineq[h_tot:h_tot + hi, 0] = diag_a @ bi
        ineq[h_tot:h_tot + hi, var_tot +
             1: var_tot + ni + 1] = diag_a @ Wi
        if eq is not None and i != (k-2):
            diag_a = np.diag([a for a in activation_i])
            # Should be able to exclude eq constraints for final layer
            eq[h_tot: h_tot + hi, 0] = -diag_a @ bi
            eq[h_tot:h_tot + hi, var_tot +
               1: var_tot + ni + 1] = -diag_a @ Wi
            eq[h_tot:h_tot + hi, var_tot +
               ni + 1: var_tot + ni + hi + 1] = np.eye(hi)

        h_tot += hi
        var_tot += ni

    return ineq, eq


def check_fixed_hyperplanes(n_dim: int, domain_constr, W1, b1):
    """Checks if hyperplanes are fixed in the domain

    Checks if each neuron is only ever active or inactive. If only
    one then it can be fixed and 'pruned'
    First layer neurons only

    Args:
        n_dim (int): number of dimensions
        domain_constr (np.ndarray): domain constraints
        W1 (np.ndarray): weight matrix for first layer
        b1 (np.ndarray): bias vector for first layer

    Returns:
        dict:
            dict of index: fix of indices of 
            neurons that can be fixed and the corresponding
            mode - active (1) or inactive (0)
    """
    activation_on = [1] * W1.shape[0]
    domain_constr = domain_constr[:, 0:n_dim + 1]
    A = np.zeros((W1.shape[0], W1.shape[1] + 1))
    diag_a = np.diag([2 * a - 1 for a in activation_on])
    A[:, 0] = diag_a @ (b1)
    A[:, 1:] = diag_a @ (W1)

    on_res = []
    for plane in A:
        constr = np.vstack((domain_constr, plane))
        P = polyhedrons.Polyhedron(constr)
        on_res.append(int(P.is_nonempty()))
    activation_off = [0] * W1.shape[0]
    A = np.zeros((W1.shape[0], W1.shape[1] + 1))
    diag_a = np.diag([2 * a - 1 for a in activation_off])
    A[:, 0] = diag_a @ (b1)
    A[:, 1:] = diag_a @ (W1)
    off_res = []
    for plane in A:
        constr = np.vstack((domain_constr, plane))
        P = polyhedrons.Polyhedron(constr)
        off_res.append(-int(P.is_nonempty()))
    # If both on&off are non-empty, we want zero as cannot fix plane.
    # Otherwise want to fix plane to on (1) or off (0)
    combine_res = [a + b for a, b in zip(on_res, off_res)]
    fixed_res = {i: val for i, val in enumerate(combine_res) if val != 0}
    fixed_res = {i: val if val == 1 else 0 for i, val in fixed_res.items()}
    return fixed_res


def check_activation_cdd_1l(ineq_constr, eq_constr, W, b, activation):
    W1 = W[0]
    b1 = b[0]
    n_constr = ineq_constr.shape[0] - 2 * (ineq_constr.shape[1] - 1)
    diag_a = np.diag([2 * a - 1 for a in activation])
    # Net is Wx + b >= 0   (including diags to switch inactive sign)
    # Cdd wants [c -A] from Ax <= 0
    # Therefore c = b, -W = -A
    ineq_constr[:n_constr, 0] = diag_a @ (b1)
    ineq_constr[:n_constr, 1:] = diag_a @ (W1)
    P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
    return P.is_nonempty()


@timer(T)
def check_activation_cdd(ineq_constr, eq_constr, W, b, activation):
    """ Checks if activation is valid using cdd solver

    Using cdd-lib to check if a given activation configuration is 
    active by constructing the convex polyhedron and check if it is
    non-empty.
    Args:
        ineq_constr (np.ndarray): inequality constraint
        eq_constr (np.ndarray): equality constraint
        W (List(np.ndarray)): weight matrices
        b (List(np.ndarray)): bias vectors
        activation (List): activation configuration
    Returns:
        bool: True if nonempty, False if not
    """
    ineq_constr, eq_constr = get_constraints(
        ineq_constr, eq_constr, W, b, activation)
    P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
    return P.is_nonempty()


# @timer(T)
# def check_activation_gb(h, n_dim, W1, b1, activation):
#     """
#     Using cdd-lib to check if a given activation configuration is 
#     active by constructing the convex polyhedron and check if it is
#     non-empty.
#     """
#     LP = gp.Model('x', env=env)
#     x = np.array(
#         [LP.addVar(name=f'x{i}', lb=-float('inf')) for i in range(n_dim)])
#     # Add input set constraints
#     for i in range(n_dim):
#         LP.addConstr(x[i] <= 1)
#         LP.addConstr(x[i] >= -1)
#     # Add activation (polyhedron) constraints
#     for i in range(n_dim):
#         W1i = W1[i]
#         b1i = b1[i]
#         for j in range(h):
#             cons = b1i[j]
#             for k in range(n_dim):
#                 cons += W1i[j, k] * x[k]

#             # print(cons)
#             LP.addConstr((2 * activation[i*h + j] - 1) * cons >= 0)

#     # Check feasibilty
#     LP.optimize()
#     return LP.status == gp.GRB.OPTIMAL


def enumerate_configurations(ineq_constr, eq_constr, W, b, fixed_acts={}):
    """Returns all possible activation configurations by enumeration

    Finds all active configs by enumerating through all possible 
    configurations and checking the corresponding LP is feasible.

    Args:
        ineq_constr (np.ndarray): inequality constraint matrix
        eq_constr (np.ndarray): equality constraint matrix
        W (List(np.ndarray)): list of weight matrices
        b (List(np.ndarray)): list of bias vectors
        fixed_acts (dict, optional): 
            dict of fixed activations. k-v pairs neuron index
            and mode (1 on and 0 off). Defaults to {}.

    Returns:
        activations (list): list of activation configurations
    """
    n_acts = sum([W[i].shape[0] for i in range(len(W) - 1)])
    activation = [0] * (n_acts - len(fixed_acts))
    feasible_acts = []
    # NOTE: There's a slight bug with this. If all activations are fixed (Not sure if this should be possible,
    # then this function returns not legit activations, when there should be a single one (the full fixed one)).
    # The below code acts as an edge case check for this, but this might not be correct behaviour
    if activation == []:
        return [list(fixed_acts.values())]
    while True:

        full_activation = copy.deepcopy(activation)
        _ = [full_activation.insert(i, val)
             for i, val in fixed_acts.items()]
        if check_activation_cdd(ineq_constr, eq_constr, W, b, full_activation):
            feasible_acts.append(copy.deepcopy(full_activation))
        if increment(activation):
            break
    # print(T)
    return feasible_acts


def get_active_configurations(domain, n_dim, W, b, mode='tree'):
    """Find all activation configurations using a given technique

    Args:
        domain (domain.Rectangle):  
            Rectangle domain with lower & upper bounds
        n_dim (int): number of dimensions of f(x)
        W (List(np.ndarray)): list of weight matrices
        b (_type_): list of bias vectors
        mode (str, optional): mode to use. Defaults to 'tree'.

    Returns:
        List: list of activation configurations
    """    
    ineq_constr, eq_constr = construct_empty_constraint_matrices(W, b)
    domain_constr = get_domain_constraints(
        domain, W[0].shape[1], nvars=(ineq_constr.shape[1] - 1))
    # fixed_acts = check_fixed_hyperplanes(n_dim, domain_constr, W[0], b[0])
    # print(len(fixed_acts))
    fixed_acts = {}
    ineq_constr = np.concatenate([ineq_constr, domain_constr])
    if mode == 'tree':
        tree = NeuronTree(W, b, domain)
        configs = tree.explore_tree()
    elif mode == 'allsat':
        configs = all_sat(W, b, domain)
    elif mode == 'enum':
        configs = enumerate_configurations(
            ineq_constr, eq_constr, W, b, fixed_acts=fixed_acts)

    return configs


def get_Wb(nets):
    """ Returns combined weight matrices and bias vectors for ReluNets"""
    W = []
    b = []
    for net in nets:
        Wi = []
        bi = []
        for layer in net.model:
            if isinstance(layer, torch.nn.Linear):
                Wi.append(layer.weight)
                bi.append(layer.bias)
        W.append(Wi)
        b.append(bi)
    W, b = list(map(list, zip(*W))), list(map(list, zip(*b)))
    W = [torch.block_diag(*Wi) if i != 0 else torch.cat(Wi)
         for i, Wi in enumerate(W)]
    b = [torch.cat(bi) for bi in b]

    for w1, w2 in zip(W, W[1:]):
        assert(w1.shape[0] == w2.shape[1])
    for w1, b1 in zip(W, b):
        assert(w1.shape[0] == b1.shape[0])
    W = [Wi.detach().numpy().round(7) for Wi in W]
    b = [bi.detach().numpy().round(7) for bi in b]
    return W, b

def get_mode_flow(W, b, act):
    k = len(W)
    A = np.eye(W[0].shape[1])
    c = 0
    h_tot = 0
    layer_acts = []
    for i in range(k-1):
        Wi, bi = W[i], b[i]
        hi = Wi.shape[0]  # Number of neurons in layer
        activation_i = act[h_tot: h_tot + hi]
        layer_acts.append(activation_i)
        h_tot += hi

    for i in range(k-1):
        Wi, bi = W[i], b[i]
        activation_i = layer_acts[i]
        diag_a = np.diag(activation_i)
        A = diag_a @ Wi @ A
        Wj = np.eye(W[i].shape[0])
        for j in range(i + 1, k - 1):
            diag_a = np.diag(layer_acts[j])
            Wj = diag_a @  W[j] @ Wj
        Wj = W[-1] @ Wj
        c += Wj @ np.diag(layer_acts[i])@ bi
    A = W[-1] @ A
    c = b[-1] + c
    return A,c

class NeuralAbstraction:
    """Class to represent a neural abstraction and determine the required
    objects to cast a ReluNet & error bound as a hybrid automaton."""
    def __init__(self, net, error, benchmark) -> None:
        """Initialise the NeuralAbstraction class for a given dynamical model

        Args:
            net (ReLUNet): trained neural network
            error (List(float)): error bound (for each dimension)
            benchmark (benchmarks.Benchmark): benchmark object
        """
        self.nets = net
        self.reconstructed_net = nn.ReconstructedRelu(self.nets)
        self.dim = self.nets[0].model[0].weight.shape[1]
        self.error = error
        # print(self.error)
        self.benchmark = benchmark
        self.locations = self.get_activations()
        self.invariants = self.get_invariants()
        self.flows = self.get_flows()
        self.transitions = self.get_transitions()
        self.modes = self.get_modes()

    @timer(T)
    def get_activations(self) -> dict:
        """Get all valid activation configurations"""
        W, b = get_Wb(self.nets)
        acts = get_active_configurations(
            self.benchmark.domain, self.dim, W, b)
        return {str(i): acts[i] for i in range(len(acts))}
    
    @staticmethod
    def get_timer():
        return T

    def get_invariants(self) -> dict:
        """Get invariant polyhedrons for each activation configuration"""
        invariants = {}
        W, b = get_Wb(self.nets)
        ineq, eq = construct_empty_constraint_matrices(W, b)

        for loc, activation in self.locations.items():
            ineq_constr, eq_constr = get_constraints(
                ineq, eq, W, b, activation)
            domain_constr = get_domain_constraints(
                self.benchmark.domain, self.dim, nvars=(ineq_constr.shape[1] - 1))
            ineq_constr = np.vstack([domain_constr, ineq_constr])
            P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
            P.reduce_to_nd(self.dim)
            invariants.update({loc: P})
        return invariants

    def get_transitions(self) -> dict:
        """Detect all transitions between activation configurations"""
        transitions = []
        locations = list(self.locations.keys())
        for l0 in locations:
            P0 = self.invariants[l0]
            for l1 in locations:
                if l0 == l1:
                    # Self transition
                    continue
                P1 = self.invariants[l1]
                V = get_shared_vertices(P0, P1, self.dim)
                if len(V) > 0:
                    # Transition may exists from l0 to l1
                    if len(V) == 1:
                        # A, c = self.flows[l0]
                        # x = np.array(
                        #     next(iter(V))[1: 1 + self.dim]).reshape(-1, 1)
                        # flow_V = A @ x + c.reshape(-1, 1)
                        # # Get a hyperplane from P1 that x is part of
                        # for i, hyperplane in enumerate(P0.H):
                        #     if i not in P0.H.lin_set:
                        #         hp = hyperplane[0: 1 + self.dim]
                        #         a = np.array(hp[1:]).reshape(-1, 1)
                        #         b = np.array(hp[0])
                        #         # print(float(abs(a.T @ x - b)))
                        #         if a.T @ x == b:
                        #             threshold = 0.1
                        #             if a @ flow_V < threshold:
                        #                 pass
                        transitions.append((l0, l1))
                    else:
                        transitions.append((l0, l1))
        return transitions

    def get_flows(self):
        """Get affine flow for each activation configuration"""
        flows = {}
        W, b = get_Wb(self.nets)
        for loc, act in self.locations.items():
            A, c = get_mode_flow(W, b, act)
            flows.update({loc: (copy.deepcopy(A), copy.deepcopy(c))})
        return flows

    def get_modes(self):
        """Construct modes of abstraction"""
        modes = {}
        for loc in self.locations.keys():
            M = Mode(self.flows[loc], self.invariants[loc], self.error)
            modes.update({loc: M})
        return modes

    def find_mode(self, x: list):
        """Finds which mode a point lies in"""
        for loc, mode in self.modes.items():
            if mode.contains(x):
                return loc
        return None

    def plot(self, label=False, show=True):
        """Plot the neural abstraction and its partitions"""
        net = nn.ReconstructedRelu(self.nets)
        domain = self.benchmark.domain
        xb = [domain.lower_bounds[0], domain.upper_bounds[0]]
        yb = [domain.lower_bounds[1], domain.upper_bounds[1]]
        sysplot.plot_nn_vector_field(net, xb, yb)
        for lab, inv in self.invariants.items():
            inv.plot(color='k')
            if label:
                c = np.array(inv.V)[:, 1:].mean(axis=0)
                plt.text(c[0], c[1], r"$P_{{{}}}$".format(
                    lab), fontsize='large')
        plt.xticks([-1, 0, 1])
        plt.yticks([-1, 0, 1])
        plt.xlim(xb)
        plt.ylim(yb)
        plt.gca().set_aspect('equal')
        if show:
            plt.show()
        else:
            return plt.gca()

    def to_xml(self, filename: str, bounded_time=False, T=1.5, initial_state=None):
        """Method for saving the abstraction to an XML file

        Args:
            filename (str):
                name of xml file to save to (no extension)
            bounded_time (bool, optional): 
                If True, it bounds the time horizon for the
                hybrid automaton. Defaults to False.
            T (float, optional): 
                Time horizon if bounded time is True . Defaults to 1.5.
            initial_state (Polyhedron, optional): adds extra mode 
                corresponding to the initial state. Defaults to None.
        """
        if initial_state:
            filename += '_init'
        SEP = " &\n"
        vx = ['x' + str(i) for i in range(self.dim)]
        vu = ['u' + str(i) for i in range(len(self.error))]
        var_attrib = {'name': None, 'type': 'real', 'd1': '1', 'd2': '1',
                      'local': 'false', 'dynamics': 'any', 'controlled': 'true'}
        root = ET.Element('xml')
        root = ET.Element('sspaceex', {
                          "xmlns": "http://www-verimag.imag.fr/xml-namespaces/sspaceex", 'version': '0.2', 'math': 'SpaceEx'})
        component = ET.SubElement(root, 'component', {'id': 'NA'})
        note = ET.SubElement(component, "note")
        # self.error = [0 for i in self.error]
        note.text = "Model error = {}".format(self.error)
        # Add variables  to component
        for var in vx:
            var_attrib.update({'name': var})
            ET.SubElement(component, 'param', var_attrib)
        for var in vu:
            var_attrib.update({'name': var, 'controlled': 'false'})
            ET.SubElement(component, 'param', var_attrib)
        if bounded_time:
            var_attrib.update({'name':'t', 'controlled': 'true'})
            ET.SubElement(component, 'param', var_attrib)

        # Add each location (inv & flow)
        for loc_id in self.locations.keys():
            location = ET.SubElement(component, 'location', {
                                     'id': loc_id, 'name': "P"+loc_id})


            ### Flows
            mode = self.modes[loc_id]
            flow_element = ET.SubElement(location, 'flow')
            flow = self.flows[loc_id]
            flow_element.text = mode.flow_str(sep=" &\n")

            if bounded_time:
                flow_element.text += "&\nt'==1"

            ### Invariants
            inv = ET.SubElement(location, 'invariant')
            P = self.invariants[loc_id]
            inv.text = mode.inv_str(sep=SEP)

            if bounded_time:
                inv.text += "&\nt <={}".format(T)

        if bounded_time:
            # Final location for bounded time
            location = ET.SubElement(component, 'location', {
                                     'id': str(len(self.locations)), 'name': "End"})
            flow_element = ET.SubElement(location, 'flow')
            flow_element.text = ""
            for i, var in enumerate(vx):
                flow_element.text +=  var + "'== 0" + SEP
            flow_element.text += "t'==0"
            inv = ET.SubElement(location, 'invariant')
            inv.text = "t >={}\n".format(T)

        if initial_state:
            # Add extra location for initial state
            location = ET.SubElement(component, 'location', {
                                     'id': str(len(self.locations)+1), 'name': "Init"})
            flow_element = ET.SubElement(location, 'flow')
            flow_element.text = ""
            for i, var in enumerate(vx):
                flow_element.text +=  var + "'== 0" + SEP
            flow_element.text = flow_element.text[:-2]
            if bounded_time:
                flow_element.text += SEP + "t'==1"
                # inv = ET.SubElement(location, 'invariant')
                # inv.text = "t == 0"

        # Add transitions between locations (guard, l0 l1)
        for transition in self.transitions:
            t0 = transition[0]
            t1 = transition[1]
            transition = ET.SubElement(component, 'transition', {
                                       'source': str(t0), 'target': str(t1)})
            # ET.SubElement(transition, 'label').text = 'switch_mode'
            guard_element = ET.SubElement(transition, 'guard')
            guard = self.invariants[t1]
            guard_element.text = guard.to_str(sep=SEP) + SEP 
            if bounded_time:
                guard_element.text += "t <= {}&\n".format(T)
            guard_element.text = guard_element.text[:-2]
        
        if bounded_time:
            for loc_id in self.locations.keys():
                transition = ET.SubElement(component, 'transition', {
                                       'source': loc_id, 'target': str(len(self.locations))})
                guard_element = ET.SubElement(transition, 'guard')
                guard_element.text = "t >= {}".format(T)

        if initial_state:
            for loc_id in self.locations.keys():
                mode = self.modes[loc_id]
                if mode.P.intersection(mode.P, initial_state).is_nonempty():
                    transition = ET.SubElement(component, 'transition', {
                                       'source': str(len(self.locations) +1), 'target': loc_id})
                                       
                    # guard_element = ET.SubElement(transition, 'guard')
                    # if bounded_time:
                        # guard_element.text +=  "t == 0"

        tree = ET.ElementTree(root)
        indent(root)
        tree.write('{}.xml'.format(filename),
                   encoding="", xml_declaration=True)

    def to_pkl(self, filename: str):
        filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, filename: str) -> NeuralAbstraction:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def as_smt(self, x, And):
        raise NotImplementedError
    
def share_vertex(P1, P2):
    """Checks is two Polyhedra share a vertex.

    Args:
        P1 (polyhedrons.Polyhedron): first polyhedron.
        P2 (polyhedrons.Polyhedron): second polyhedron.

    Returns:
        bool: True if P1 and P2 share a vertex
    """
    V1 = np.array(P1.V)  # .round(7)
    V2 = np.array(P2.V)  # .round(7)
    V1 = tuple(map(tuple, V1))
    V2 = tuple(map(tuple, V2))
    return not set(V1).isdisjoint(V2)
    # return shared_vertices(P1, P2) > 1

def count_shared_vertices(P1, P2):
    """Takes two Polyhedra and returns the number of shared vertices.

    Args:
        P1 (polyhedrons.Polyhedron): first polyhedron.
        P2 (polyhedrons.Polyhedron): second polyhedron.

    Returns:
        int: number of shared vertices.
    """
    V1 = np.array(P1.V)  # .round(7)
    V2 = np.array(P2.V)  # .round(7)
    V1 = tuple(map(tuple, V1))
    V2 = tuple(map(tuple, V2))
    return len(set(V1).intersection(V2))


def get_shared_vertices(P1, P2, ndim):
    """Takes two Polyhedra and returns the shared vertices.

    Args:
        P1 (polyhedrons.Polyhedron): first polyhedron.
        P2 (polyhedrons.Polyhedron): second polyhedron.
        ndim (int): number of dimensions of the polyhedra.

    Returns:
        V (set): shared vertices.
    """
    V1 = np.array(P1.V)[:, :ndim+1]  # .round(7)
    V2 = np.array(P2.V)[:, :ndim+1]  # .round(7)
    V1 = tuple(map(tuple, V1))
    V2 = tuple(map(tuple, V2))
    return set(V1).intersection(V2)


UNKNOWN = None
OFF = 0
ON = 1
# BOTH_ON = "both_on"
# BOTH_OFF ="both_off"


class NeuronTree:
    """Represents all possible activation configurations as a tree,
        and contains a depth-first search algorithm to find the valid 
        ones."""

    def __init__(self, W, b, domain) -> None:
        """Initializes the tree.

        Args:
            W (List(np.ndarray)): list of weight matrices.
            b (List(np.ndarray)): list of bias vectors.
            domain (domains.Rectangle): domain of the abstraction
        """
        self.W = W
        self.b = b
        self.domain = domain
        self.layer_struct = [W[i].shape[0] for i in range(len(W) - 1)]
        self.current_path = []
        self.current_ineq_constr, self.current_eq_constr = construct_empty_constraint_matrices(
            self.W, self.b)
        self.queue = queue.Queue()
        self.configs = []
        self.stack = queue.LifoQueue()

    def check_mode(self, neuron, domain_constr):
        """Checks if a neuron is ON or OFF or both at the current location in the tree.

        Args:
            neuron (dict): neuron to check
            domain_constr (np.ndarray): current domain constraints
        """
        # Check off halfspace
        off_mode = None
        neuron_count = neuron['neuron_count']
        self.current_path.append(0)
        a = self.current_path + [1] * \
            (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a)
        ineq_constr = ineq_constr[:len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            index = min(len(self.current_path), eq_constr.shape[0])
            if index > 0:
                eq_constr = eq_constr[:index, :]
        P_off = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        off_mode = P_off.is_nonempty()
        self.current_path.pop()

        # Check on halfpsace
        on_mode = None
        self.current_path.append(1)
        a = self.current_path + [1] * \
            (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a)
        ineq_constr = ineq_constr[:len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            index = min(len(self.current_path), eq_constr.shape[0])
            if index > 0:
                eq_constr = eq_constr[:index, :]
        P_on = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        on_mode = P_on.is_nonempty()
        self.current_path.pop()

        if off_mode and on_mode:
            neuron['mode'] = 0
            # Add neuron to stack with opposite mode to return to later
            self.stack.put(neuron)
            self.current_path.append(1)
        elif on_mode:
            self.current_path.append(1)
        else:
            assert(off_mode == True)
            self.current_path.append(0)

    def explore_tree(self):
        """Explores the tree using a depth first approach.
        
        Finds all valid activation configurations."""
        neuron_count = 0
        for hi, layer in enumerate(self.layer_struct):
            for ni in range(layer):
                neuron_count += 1
                self.queue.put({'layer': hi, 'neuron_index': ni,
                               'mode': None, 'neuron_count': copy.deepcopy(neuron_count)})

        domain_constr = get_domain_constraints(
            self.domain, self.W[0].shape[1], nvars=(self.current_ineq_constr.shape[1] - 1))
        while not (self.queue.empty()) or not self.stack.empty():
            if not self.queue.empty():
                # Still going down tree (towards leaf)
                neuron = self.queue.get()
                self.check_mode(neuron, domain_constr)
            else:
                # End of path. Save config and go back to last branching
                assert(not self.stack.empty())
                assert(len(self.current_path) == sum(self.layer_struct))
                self.configs.append(copy.deepcopy(self.current_path))
                neuron = self.stack.get()
                neuron_count = neuron['neuron_count']
                self.current_path = self.current_path[:neuron_count-1]
                self.current_path.append(0)
                for hi, layer in enumerate(self.layer_struct):
                    for ni in range(layer):
                        if hi == neuron['layer']:
                            if ni > neuron['neuron_index']:
                                neuron_count += 1
                                self.queue.put({'layer': hi, 'neuron_index': ni,
                                                'mode': None, 'neuron_count': copy.deepcopy(neuron_count)})
                        if hi > neuron['layer']:
                            neuron_count += 1
                            self.queue.put({'layer': hi, 'neuron_index': ni,
                                            'mode': None, 'neuron_count': copy.deepcopy(neuron_count)})
        a = self.current_path + [1] * \
            (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a)
        ineq_constr = ineq_constr[:len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            eq_constr = eq_constr[:len(self.current_path), :]
        # print(self.current_path)
        P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        mode = P.is_nonempty()
        if self.current_path not in self.configs:
            if mode:
                self.configs.append(copy.deepcopy(self.current_path))
        else:
            print('It happened')
        return self.configs


def all_sat(W, b, domain):
    """SMT-based algorithm for finding all active neuron configurations.

    Finds all active neuron configurations of the ReLUNet described by 
    W and b in the domain.

    Args:
        W (List[np.ndarray]): list of weight matrices.
        b (List[np.ndarray]): list of bias vectors.
        domain (domains.Rectangle): domain of abstraction.

    Returns:
        configs (List): list of active neuron configurations.
    """
    ndim = W[0].shape[1]
    configs = []
    x = np.array([z3.Real('x' + str(i)) for i in range(ndim)]).reshape(-1, 1)
    Nx = x
    neurons_enabled = []
    for Wi, bi in zip(W[:-1], b[:-1]):
        Nx = Wi @ Nx + bi.reshape(-1, 1)
        neurons = [neuron > 0 for neuron in Nx.squeeze(axis=1)]
        neurons_enabled.extend(neurons)
        Nx = verifier.Z3Verifier.relu(Nx)
    Nx = W[-1] @ Nx + b[-1].reshape(-1, 1)
    XD = domain.generate_domain(x.squeeze().tolist(), z3.And)
    solver = z3.Solver()
    solver.add(XD)
    while solver.check() == z3.sat:
        witness = solver.model()
        config = [int(z3.is_true(witness.eval(ni))) for ni in neurons_enabled]
        configs.append(config)
        symbolic_config = [z3.simplify(
            z3.Not(ni)) if c == 0 else ni for ni, c in zip(neurons_enabled, config)]
        solver.add(z3.Not(z3.And(symbolic_config)))
    return configs

# perform a depth first tree search on a ReLUNet and return all valid activation configurations as a list of lists of integers

def dfs(P_mat, ineq, eq):
    """Depth first search on a ReLUNet.
    
    Args:
        P_mat (np.ndarray): polyhedron.
        ineq (np.ndarray): inequality constraints.
        eq (np.ndarray): equality constraints.
    
    Returns:
        configs (List): list of valid activation configurations.
    """
    ## Add next positive neuron constraint to P
    P = polyhedrons.Polyhedron(P_mat, L=eq)
    if P.is_nonempty():
        return [1, *dfs(P_mat, ineq, eq)]
    
    
class Mode:
    def __init__(self, flow, invariant, disturbance=None) -> None:
        """Represents a single mode in a linear hybrid automaton"""
        self.A, self.c = flow
        self.ndim = len(self.c)
        self.P = invariant
        if disturbance is None:
            self.disturbance = np.zeros((self.ndim))
        else:
            self.disturbance = np.array(disturbance)
        self.rand = np.random.default_rng()
    
    def flow(self, x):
        """Returns the mode's output for a given input"""
        d = self.rand.uniform(0, self.disturbance.max(), self.disturbance.shape)
        return (self.A @ np.array(x) + self.c + d).tolist()

    def flow_smt(self, x):
        """Returns the flow without disturbance in SMT format"""
        return (self.A @ np.array(x) + self.c).tolist()

    def contains(self, x):
        """Checks if a given input is in the mode's domain"""
        return self.P.contains(x)

    def as_smt(self, x, And):
        """Represent the mode as SMT formula for the invariant and flow"""
        return self.P.as_smt(x, And), self.flow_smt(x)

    def inv_str(self, vx=[], vu=[], sep="\n"):
        """Returns the invariant as a string"""
        s = self.P.to_str(vx=vx, sep=sep) + sep
        vu = vu if vu else ["u" + str(i) for i in range(self.ndim)]
        for i, ui in enumerate(vu):
            s += ui + " <={}".format(self.disturbance[i])
            s += sep + ui + " >= -{}".format(self.disturbance[i])
            s += sep
        s = s[:-2]  # remove final &
        return s

    def flow_str(self, vx=[], vu=[], sep="\n"):
        """Get string representation of flow"""
        s = ""
        vx = vx if vx else ["x" + str(i) for i in range(self.ndim)]
        vu = vu if vu else ["u" + str(i) for i in range(self.ndim)]
        if len(vx) != self.ndim:
            raise ValueError("Number of variables does not match")
        if len(vu) != self.ndim:
            raise ValueError("Number of inputs does not match")
        if not all(isinstance(item, str) for item in vx):
            raise ValueError("vx must be a list of strings")
        s = ""
        for i, var in enumerate(vx[:-1]):
            A_row = self.A[i, :]
            b = self.c[i]
            s += var + "'=="
            fi = " + ".join([str(A_row[j]) + " * " + vx[j]
                            for j in range(self.ndim)])
            s += fi + "+ " + str(b) + " + " + vu[i]
            s += sep

        A_row = self.A[-1, :]
        b = self.c[-1]
        t = "+ ".join([str(A_row[j]) + " * " + vx[j]
                        for j in range(self.ndim)])

        s += vx[-1] + "'=="
        s += t + "+ " + str(b) + ' + ' + vu[-1]
        return s


def prune_transitions(P1, P2, f1, f2):
    V = get_shared_vertices(P1, P2)
    if len(V) == 1:
        pass
    elif len(V) == 2:
        pass
    else:
        return True
        
def check_transition(hyperplane, flow, p1, p2):
    """Check if a transition is valid"""
    A, c = flow
    w = - np.array(hyperplane[1:]) # Polyhedron stores [b -W] for Wx <= b
    x = [z3.Real('x' + str(i)) for i in range(A.shape[1])]
    v1 = w.T @ (A @ np.array(p1) + c)
    v2 = w.T @ (A @ np.array(p2) + c)
    return v1 > 0 or v2 > 0
    