# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np
from math import cos, sin, pi
from matplotlib import pyplot as plt
import cdd

import domains
from utils import decimal_from_fraction

# env = gp.Env(empty=True)
# env.setParam('OutputFlag', 0)
# env.start()


def get_slope(p1, p2):
    """Get the slope of the line between p1 and p2"""
    if p1[0] == p2[0]:
        return float("inf")
    else:
        return 1.0 * (p1[1] - p2[1]) / (p1[0] - p2[0])


def get_cross_product(p1, p2, p3):
    """Get the cross product of the vectors p1-p2 and p1-p3"""
    return ((p2[0] - p1[0]) * (p3[1] - p1[1])) - ((p2[1] - p1[1]) * (p3[0] - p1[0]))


def graham_scan(points):
    """Graham's scan algorithm for convex hull of set of points"""
    # TO do in n-D, need to do this as normal, then project all points into the
    #  dimensions you want to plot on - eg [x1, x2, x3, x4] -> [x2, x4]
    hull = []
    points.sort(key=lambda x: [x[0], x[1]])
    for point in points:
        if not all([(p < 1 and p > -1) for p in point]):
            pass  # raise RuntimeError('Point outside domain')
    start = points.pop(0)
    hull.append(start)
    points.sort(key=lambda p: (get_slope(p, start), -p[1], p[0]))
    for pt in points:
        hull.append(pt)
        while len(hull) > 2 and get_cross_product(hull[-3], hull[-2], hull[-1]) < 0:
            hull.pop(-2)
    return hull


def project_points(points, dimensions=[1, 1]):
    """Project points onto the given dimensions"""
    assert len(points[0]) == len(dimensions)
    assert sum(dimensions) == 2
    projected_points = []
    for point in points:
        p = [point[i] for i in range(len(point)) if dimensions[i] == 1]
        projected_points.append(p)
    return projected_points


# def check_collisions(n, XB, P):
#     LP = gp.Model('x', env=env)
#     x = np.array(
#         [LP.addVar(name=f'x{i}', lb=-float('inf')) for i in range(n)])
#     # Add input set constraints
#     for direction, bound in zip(XB.template, XB.bound):
#         LP.addConstr(sum([direction[i] * x[i]
#                           for i in range(n)]) <= bound)
#     for direction, bound in zip(P.template, P.bound):
#         LP.addConstr(sum([direction[i] * x[i]
#                           for i in range(n)]) <= bound)
#     LP.optimize()
#     if LP.status == gp.GRB.OPTIMAL:
#         raise RuntimeError('Collision Detected')


class CDDPolyHedron(cdd.Matrix):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def __reduce__(self):
        return None


class Polyhedron:
    """Wrapper for cdd.Polyhedron"""

    def __init__(self, A, L=None) -> None:
        """Initialize a polyhedron from a list of inequalities.

        Generates a polyhedron from a list of inequality constraints
        and equality constraints. Constraint matrices are of the form
        [b -A] for Ax <= b and [b -A] for Ax = b.


        Args:
            A (np.ndarray):
                list of inequality constraints
            L (np.ndarray, optional):
                list of equality constraints. Defaults to None.
        """
        M = cdd.Matrix(A, number_type="fraction")
        if L is not None:
            M.extend(L, linear=True)
        M.rep_type = cdd.RepType.INEQUALITY
        # M.canonicalize()
        P = cdd.Polyhedron(M)
        self.H = tuple(P.get_inequalities())
        self.V = tuple(P.get_generators())

    @classmethod
    def intersection(cls, P1, P2):
        """Intersection of two polyhedra"""
        H = (*P1.H, *P2.H)
        return cls(H)

    def get_vertices(self, dim=2, project=False):
        """Get verticies of polyhedron

        Args:
            dim (int, optional):
                number of dimensions of polyhedron. Defaults to 2.
            project (bool, optional): project points to 2D. Defaults to False.

        Returns:
            _type_: coordinates of vertices in R^n.
        """
        gen = self.V
        l = []
        for row in gen:
            assert row[0] == 1
            l.append(row[1 : 1 + dim])
        l = graham_scan(l)
        if project:
            l = project_points(l)
            l = graham_scan(l)
        return l

    def plot(self, color="tab:red"):
        """Plots polyhedron

        Args:
            color (str, optional):
            Colour of polyhedron. Defaults to 'tab:red'.
        """
        coord = self.get_vertices()
        # repeat the first point to create a 'closed loop'
        coord.append(coord[0])
        xs, ys = zip(*coord)  # create lists of x and y values
        plt.plot(xs, ys, color=color)

    def is_nonempty(self) -> bool:
        """Check if polyhedron is nonempty"""
        # return len(self.V) > 0
        M = cdd.Matrix(self.H)
        M.obj_type = cdd.LPObjType.MAX
        M.obj_func = [0 for i in range(M.col_size)]
        LP = cdd.LinProg(M)
        LP.solve()
        return (LP.status == cdd.LPStatusType.OPTIMAL) and self.V[:] != ()

    def reduce_to_nd(self, ndim):
        """Simplifies polyhedron to n-dimensional space

        Args:
            ndim (int): _description_
        """
        if ndim == 1:
            # Not sure what to do here
            M = cdd.Matrix(self.H)
            M.rep_type = cdd.RepType.INEQUALITY
            M.canonicalize()
            P = cdd.Polyhedron(M)
            self.H = tuple(P.get_inequalities())
            self.V = tuple(P.get_generators())
            return
        v = self.get_vertices(dim=ndim)
        v = [[1, *vi] for vi in v]
        M = cdd.Matrix(v)
        M.rep_type = cdd.RepType.GENERATOR
        M.canonicalize()
        P = cdd.Polyhedron(M)
        self.H = tuple(P.get_inequalities())
        self.V = tuple(P.get_generators())

    def contains(self, x):
        """Check if x lies in polyhedron"""
        x = np.array(x).reshape(-1, 1)
        b = np.array(self.H)[:, 0].reshape(-1, 1)
        A = -np.array(self.H)[:, 1:]
        return np.all(np.dot(A, x) <= b)

    def as_smt(self, x, And):
        """Express in SMT"""
        x = np.array(x).reshape(-1, 1)
        b = np.array(self.H)[:, 0]
        A = -np.array(self.H)[:, 1:]
        Ax = (A @ x).squeeze()
        return And([ax <= bi for ax, bi in zip(Ax, b)])

    def is_subset(self, P):
        """Check if polyhedron is a subset of P

        Very inefficient. Might be better to use LPs or SMT"""
        return all([P.contains(vi[1:]) for vi in self.V])

    def box_sample(self, N: int):
        """Sample points from within polyhedron"""
        box_template = get_2d_template(4)
        P_box = convert2template(self, box_template)
        ub = [float(P_box.bound[0]), float(P_box.bound[1])]
        lb = [-float(P_box.bound[2]), -float(P_box.bound[3])]
        R = domains.Rectangle(lb, ub)
        D = R.generate_data(N).numpy().astype(float)
        D = [d for d in D if self.contains(d)]
        return D

    def max_1_norm(self):
        """Get maximum 1-norm of polyhedron"""
        M = cdd.Matrix(self.H)
        M.obj_type = cdd.LPObjType.MAX
        M.obj_func = [0, *[1 for i in range(M.col_size - 1)]]
        LP = cdd.LinProg(M)
        LP.solve()
        return LP.obj_value

    def max_inf_norm(self):
        """Get maximum inf-norm of polyhedron"""
        max_val = 0
        for i in range(len(self.H[0]) - 1):
            M = cdd.Matrix(self.H)
            M.obj_type = cdd.LPObjType.MAX
            M.obj_func = [0, *[0 if j != i else 1 for j in range(M.col_size - 1)]]
            LP = cdd.LinProg(M)
            LP.solve()
            max_val = max(max_val, LP.obj_value)
        return max_val

    def to_str(self, vx=[], sep="\n"):
        """Get string representation of polyhedron"""
        s = ""
        vx = vx if vx else ["x" + str(i) for i in range(len(self.H[0]) - 1)]
        if len(vx) != len(self.H[0]) - 1:
            raise ValueError("Number of variables does not match")
        if not all(isinstance(item, str) for item in vx):
            raise ValueError("vx must be a list of strings")
        for row in self.H:
            const = " + ".join(
                [
                    str(decimal_from_fraction(row[0])),
                    *[
                        str(decimal_from_fraction(row[i + 1])) + " * " + vx[i]
                        for i in range(len(self.H[0]) - 1)
                    ],
                ]
            )
            s += const + " >= 0" + sep

        s = s[: -len(sep)]  # remove final &
        return s


class TemplatePolyhedron:
    """Wrapper for cdd.Polhedron with template constraints"""

    def __init__(self, template, bounds) -> None:
        """Initialize a polyhedron from a list of inequalities.

        Generates a polyhedron from a list of inequality constraints
        corresponding to a given template. Constraint matrices are
        of the form [b -A] for Ax <=b.

        Args:
            template (list): list of template vectors
            bounds (list):
                list of floats representing bounds in each template
            vector direction.
        """
        self.template = template
        self.bound = bounds

    def get_vertices(self, project=False):
        """Get verticies of polyhedron

        Args:
            project (bool, optional): project points to 2D. Defaults to False.

        Returns:
            _type_: coordinates of vertices in R^n.
        """
        P = convert2polyhedron(self)
        return P.get_vertices(project=project)

    def plot(self, color="tab:red"):
        """Plots polyhedron

        Args:
            color (str, optional):
            Colour of polyhedron. Defaults to 'tab:red'.
        """
        coord = self.get_vertices()
        # repeat the first point to create a 'closed loop'
        coord.append(coord[0])
        xs, ys = zip(*coord)  # create lists of x and y values
        plt.plot(xs, ys, color=color)

    def is_subset(self, P):
        """Check if template polyhedron is a subset of another"""
        if self.template != P.template:
            raise RuntimeError(
                "Template Polyhedron and other Polyhedron have different template vectors"
            )
        return all(self.bound[i] <= P.bound[i] for i in range(len(self.bound)))

    def contains(self, x):
        """Check if x lies in polyhedron"""
        x = np.array(x).reshape(-1, 1)
        L = np.array(self.template)
        b = np.array(self.bound).reshape(-1, 1)
        return np.all(np.dot(L, x) <= b)

    def as_smt(self, x, And):
        """Express in SMT"""
        x = np.array(x).reshape(-1, 1)
        L = np.array(self.template)
        b = np.array(self.bound)
        Lx = (L @ x).squeeze()
        return And([lx <= bi for lx, bi in zip(Lx, b)])

    def is_nonempty(self):
        """Check if polyhedron is nonempty"""
        P = convert2polyhedron(self)
        return P.is_nonempty()

    def hitandrunsample(self, N):
        """Sample points from within polyhedron"""
        rng = np.random.default_rng()
        np.random.seed(0)
        x = np.array([0.1, 0.1]).reshape(-1, 1)  # Need initial point in polyhedron
        data = []
        A = np.array(self.template)
        b = np.array([float(b) for b in self.bound]).reshape(-1, 1)
        data.append(x)
        while len(data) < N:
            u = rng.uniform(0, 1, (2, 1))
            u = u / np.linalg.norm(u)
            z = A @ u
            c = (b - A @ x) / z
            tmin = c[z < 0].max()
            tmax = c[z > 0].min()
            x = x + (tmin + (tmax - tmin) * rng.uniform(0, 1)) * u
            data.append(x)
        return data

    def box_sample(self, N: int):
        """Sample points from within polyhedron"""
        P = convert2polyhedron(self)
        box_template = get_2d_template(4)
        P_box = convert2template(P, box_template)
        ub = [float(P_box.bound[0]), float(P_box.bound[1])]
        lb = [-float(P_box.bound[2]), -float(P_box.bound[3])]
        R = domains.Rectangle(lb, ub)
        D = R.generate_data(N).numpy().astype(float)
        D = [d for d in D if self.contains(d)]
        return D

    def support(self, direction):
        """Get support of polyhedron in direction"""
        H = [
            [bound, *[-d for d in direction]]
            for direction, bound in zip(self.template, self.bound)
        ]
        M = cdd.Matrix(H)
        M.obj_type = cdd.LPObjType.MAX
        M.obj_func = [0, *direction]
        LP = cdd.LinProg(M)
        LP.solve()
        if LP.status != cdd.LPStatusType.OPTIMAL:
            raise RuntimeError("Template conversion failed")
        return LP.obj_value


def convert2template(P, template) -> TemplatePolyhedron:
    """Convert polyhedron to template polyhedron

    Args:
        P (Polyhedron): polyhedron to convert
        template (list): list of template vectors
    """
    bounds = []
    M = cdd.Matrix(P.H)
    M.obj_type = cdd.LPObjType.MAX
    for direction in template:
        M.obj_func = [0, *direction]
        LP = cdd.LinProg(M)
        LP.solve()
        if LP.status != cdd.LPStatusType.OPTIMAL:
            raise RuntimeError("Template conversion failed")
        bounds.append(LP.obj_value)
    return TemplatePolyhedron(template, bounds)


def convert2polyhedron(P) -> Polyhedron:
    """Convert template polyhedron to polyhedron

    Args:
        P (TemplatePolyhedron): template polyhedron to convert
        l (list): list of template vectors

    Raises:
        TypeError: if P is not a TemplatePolyhedron

    Returns:
        Polyhedron: general polyhedron

    """
    # n_constr = len(P.template)
    # n_vars = len(P.template[0])
    # constr = np.zeros((n_constr, n_vars + 1))
    # constr[:, 0] = np.array(P.bound)
    # constr[:, 1:] = -np.array(P.template)
    if isinstance(P, Polyhedron):
        return P
    elif isinstance(P, TemplatePolyhedron):
        l = [
            [bound, *[-d for d in direction]]
            for direction, bound in zip(P.template, P.bound)
        ]
        return Polyhedron(np.array(l))
    else:
        raise TypeError("P must be either Polyhedron or TemplatePolyhedron")


def get_2d_template(N) -> list:
    if N < 4:
        raise ValueError("N must be >= 4")
    template_2d = [
        [round(cos(theta), 1), round(sin(theta), 1)]
        for theta in np.linspace(0, 2 * pi, N, endpoint=False)
    ]
    return template_2d


def vertices2polyhedron(vertices) -> Polyhedron:
    """Convert vertices to polyhedron"""
    vertices = graham_scan([[1, *V] for V in vertices])

    M = cdd.Matrix(vertices)
    M.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(M)
    return Polyhedron(P.get_inequalities())


NL1_XI = vertices2polyhedron([[0, 0], [0, 0.1], [0.05, 0.1], [0.05, 0]])
NL2_XI = vertices2polyhedron(
    [[-0.025, -0.9], [-0.025, -0.85], [0.025, -0.9], [0.025, -0.85]]
)
watertank_XI = vertices2polyhedron([[0], [0.01]])
jet_XI = vertices2polyhedron([[0.45, -0.6], [0.45, -0.55], [0.5, -0.6], [0.5, -0.55]])
exp_XI = vertices2polyhedron([[0.45, 0.86], [0.45, 0.91], [0.5, 0.86], [0.5, 0.91]])
steam_XI = vertices2polyhedron(
    [
        [0.7, -0.05, 0.7],
        [0.7, -0.05, 0.75],
        [0.7, 0.05, 0.7],
        [0.7, 0.05, 0.75],
        [0.75, -0.05, 0.7],
        [0.75, -0.05, 0.75],
        [0.75, 0.05, 0.7],
        [0.75, 0.05, 0.75],
    ]
)


def get_XI(benchmark: str):
    benchmark = benchmark.lower()
    if benchmark == "nl1":
        return NL1_XI
    elif benchmark == "nl2":
        return NL2_XI
    elif benchmark == "exp" or benchmark == "exponential":
        return exp_XI
    elif benchmark == "tank":
        return watertank_XI
    elif benchmark == "steam":
        return steam_XI
    elif benchmark == "jet":
        return jet_XI
    else:
        return None


if __name__ == "__main__":
    steam_XI.plot()
    plt.show()
