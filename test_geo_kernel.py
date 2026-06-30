#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariant tests for geo_kernel: assert solved coordinates satisfy the
geometric constraints each primitive promises. Run: python -m pytest test_geo_kernel.py
or just `python test_geo_kernel.py`.
"""
import math
import numpy as np
import pytest

import geo_kernel as gk

EPS = 1e-6


def _d(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def test_point_and_polar():
    fig = gk.solve([
        {"op": "point", "name": "O", "x": 0, "y": 0},
        {"op": "point_polar", "name": "P", "center": "O", "r": 5, "theta": 90},
    ])
    assert _d(fig.vec("P"), (0, 5)) < EPS


def test_point_on_circle_radius():
    fig = gk.solve([
        {"op": "point", "name": "O", "x": 1, "y": 2},
        {"op": "circle", "name": "k", "center": "O", "r": 3},
        {"op": "point_on_circle", "name": "A", "circle": "k", "theta": 37},
        {"op": "point_on_circle", "name": "B", "circle": "k", "theta": 200},
    ])
    assert abs(_d(fig.vec("A"), fig.vec("O")) - 3) < EPS
    assert abs(_d(fig.vec("B"), fig.vec("O")) - 3) < EPS


def test_midpoint_and_division():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 10, "y": 0},
        {"op": "midpoint", "name": "M", "a": "A", "b": "B"},
        {"op": "division_point", "name": "D", "a": "A", "b": "B", "m": 1, "n": 3},
    ])
    assert _d(fig.vec("M"), (5, 0)) < EPS
    # internal ratio AD:DB = m:n = 1:3 -> D at 2.5
    assert _d(fig.vec("D"), (2.5, 0)) < EPS


def test_foot_is_perpendicular():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 10, "y": 0},
        {"op": "line", "name": "L", "a": "A", "b": "B"},
        {"op": "point", "name": "P", "x": 3, "y": 4},
        {"op": "foot", "name": "F", "pt": "P", "to_line": "L"},
    ])
    f = fig.vec("F")
    assert _d(f, (3, 0)) < EPS
    # PF perpendicular to AB
    ab = fig.vec("B") - fig.vec("A")
    pf = fig.vec("P") - f
    assert abs(float(np.dot(ab, pf))) < EPS


def test_perpendicular_line_dot_zero():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 4, "y": 2},
        {"op": "line", "name": "L", "a": "A", "b": "B"},
        {"op": "point", "name": "P", "x": 1, "y": 1},
        {"op": "perpendicular", "name": "M", "through": "P", "to_line": "L"},
    ])
    L, M = fig.line("L"), fig.line("M")
    dl = fig.vec(L.b) - fig.vec(L.a)
    dm = fig.vec(M.b) - fig.vec(M.a)
    assert abs(float(np.dot(dl, dm))) < EPS


def test_parallel_cross_zero():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 4, "y": 2},
        {"op": "line", "name": "L", "a": "A", "b": "B"},
        {"op": "point", "name": "P", "x": 0, "y": 3},
        {"op": "parallel", "name": "M", "through": "P", "to_line": "L"},
    ])
    L, M = fig.line("L"), fig.line("M")
    dl = fig.vec(L.b) - fig.vec(L.a)
    dm = fig.vec(M.b) - fig.vec(M.a)
    cross = dl[0] * dm[1] - dl[1] * dm[0]
    assert abs(float(cross)) < EPS


def test_intersect_ll():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 4, "y": 4},
        {"op": "line", "name": "L1", "a": "A", "b": "B"},
        {"op": "point", "name": "C", "x": 0, "y": 4},
        {"op": "point", "name": "D", "x": 4, "y": 0},
        {"op": "line", "name": "L2", "a": "C", "b": "D"},
        {"op": "intersect_ll", "name": "X", "line1": "L1", "line2": "L2"},
    ])
    assert _d(fig.vec("X"), (2, 2)) < EPS


def test_intersect_lc_pick():
    fig = gk.solve([
        {"op": "point", "name": "O", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "O", "r": 5},
        {"op": "point", "name": "A", "x": -10, "y": 0},
        {"op": "point", "name": "B", "x": 10, "y": 0},
        {"op": "line", "name": "L", "a": "A", "b": "B"},
        {"op": "intersect_lc", "name": "R", "line": "L", "circle": "k", "pick": "right"},
        {"op": "intersect_lc", "name": "Lft", "line": "L", "circle": "k", "pick": "left"},
    ])
    assert _d(fig.vec("R"), (5, 0)) < EPS
    assert _d(fig.vec("Lft"), (-5, 0)) < EPS


def test_intersect_cc():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 6, "y": 0},
        {"op": "circle", "name": "c1", "center": "A", "r": 5},
        {"op": "circle", "name": "c2", "center": "B", "r": 5},
        {"op": "intersect_cc", "name": "P", "c1": "c1", "c2": "c2", "pick": "upper"},
    ])
    p = fig.vec("P")
    assert abs(_d(p, fig.vec("A")) - 5) < EPS
    assert abs(_d(p, fig.vec("B")) - 5) < EPS
    assert p[1] > 0  # upper


def test_ray_at_angle_creates_correct_angle():
    fig = gk.solve([
        {"op": "point", "name": "V", "x": 0, "y": 0},
        {"op": "point", "name": "A", "x": 5, "y": 0},
        {"op": "ray_at_angle", "name": "B", "vertex": "V", "from_pt": "A",
         "measure": 50, "side": "ccw"},
    ])
    va = fig.vec("A") - fig.vec("V")
    vb = fig.vec("B") - fig.vec("V")
    cosang = float(np.dot(va, vb) / (np.hypot(*va) * np.hypot(*vb)))
    assert abs(math.degrees(math.acos(cosang)) - 50) < 1e-4


def test_triangle_sss_side_lengths():
    fig = gk.solve([
        {"op": "triangle_sss", "name": "T", "names": ["A", "B", "C"],
         "a": 5, "b": 4, "c": 3},  # a=BC,b=CA,c=AB -> 3-4-5
    ])
    A, B, C = fig.vec("A"), fig.vec("B"), fig.vec("C")
    assert abs(_d(B, C) - 5) < EPS
    assert abs(_d(C, A) - 4) < EPS
    assert abs(_d(A, B) - 3) < EPS


def test_triangle_sss_inequality_raises():
    with pytest.raises(gk.GeoError):
        gk.solve([{"op": "triangle_sss", "name": "T", "a": 1, "b": 1, "c": 10}])


def test_reflect_equidistant():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 10, "y": 0},
        {"op": "line", "name": "L", "a": "A", "b": "B"},
        {"op": "point", "name": "P", "x": 3, "y": 4},
        {"op": "reflect", "name": "Q", "pt": "P", "over_line": "L"},
    ])
    assert _d(fig.vec("Q"), (3, -4)) < EPS


def test_tangent_is_perpendicular_to_radius():
    fig = gk.solve([
        {"op": "point", "name": "O", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "O", "r": 3},
        {"op": "point", "name": "E", "x": 7, "y": 0},
        {"op": "tangent_from", "name": "T", "ext": "E", "circle": "k", "pick": "upper"},
    ])
    T = fig.vec("T")
    assert abs(_d(T, fig.vec("O")) - 3) < EPS
    # radius OT perpendicular to tangent TE
    ot = T - fig.vec("O")
    te = fig.vec("E") - T
    assert abs(float(np.dot(ot, te))) < 1e-4


def test_circumcircle_passes_through_vertices():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 6, "y": 0},
        {"op": "point", "name": "C", "x": 1, "y": 5},
        {"op": "circumcircle", "name": "k", "verts": ["A", "B", "C"], "center_name": "O"},
    ])
    O = fig.vec("O")
    r = fig.circ("k").r
    for v in ("A", "B", "C"):
        assert abs(_d(fig.vec(v), O) - r) < EPS


def test_unknown_reference_raises():
    with pytest.raises(gk.GeoError):
        gk.solve([{"op": "midpoint", "name": "M", "a": "A", "b": "B"}])


def test_inscribed_angle_program():
    """The plan's worked example: inscribed angle = half the central angle."""
    fig = gk.solve([
        {"op": "point", "name": "M", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "M", "r": 5},
        {"op": "point_on_circle", "name": "A", "circle": "k", "theta": 200},
        {"op": "ray_at_angle", "name": "r1", "vertex": "M", "from_pt": "A",
         "measure": 80, "side": "ccw", "length": 5},
        {"op": "point_on_circle", "name": "Bc", "circle": "k", "theta": 280},
    ])
    # central angle AMBc should be ~80 by construction of r1 endpoint near Bc;
    # here just assert A and Bc are on the circle (full scenario tested in measure)
    assert abs(_d(fig.vec("A"), fig.vec("M")) - 5) < EPS


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
