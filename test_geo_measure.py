#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifier tests: each query cross-checked two independent ways, plus the
inscribed=half-central theorem on a real construction."""
import math
import pytest

import geo_kernel as gk
import geo_measure as gm

EPS = 1e-6


def test_length_and_pythagoras():
    fig = gk.solve([
        {"op": "triangle_sss", "name": "T", "names": ["A", "B", "C"],
         "a": 5, "b": 4, "c": 3},
    ])
    # 3-4-5 right triangle: legs 3,4 hypotenuse 5
    assert abs(gm.length(fig, "B", "C") - 5) < EPS
    assert gm.is_right_triangle(fig, "T")


def test_area_two_ways():
    # right triangle legs 6 and 8 -> area 24 via shoelace AND via base*height/2
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 6, "y": 0},
        {"op": "point", "name": "C", "x": 0, "y": 8},
        {"op": "polygon", "name": "T", "verts": ["A", "B", "C"]},
    ])
    shoelace = gm.area_polygon(fig, "T")
    bh = gm.area_triangle_base_height(6, 8)
    assert abs(shoelace - 24) < EPS
    assert abs(shoelace - bh) < EPS


def test_angle_sum_triangle():
    fig = gk.solve([
        {"op": "triangle_sss", "name": "T", "names": ["A", "B", "C"],
         "a": 7, "b": 6, "c": 5},
    ])
    sa = gm.angle(fig, "A", "B", "C")
    sb = gm.angle(fig, "B", "A", "C")
    sc = gm.angle(fig, "C", "A", "B")
    assert abs(sa + sb + sc - 180) < 1e-4


def test_sector_and_arc():
    # r=6, 60 deg: arc = 6 * pi/3 = 2pi ; sector = 0.5*36*pi/3 = 6pi
    assert abs(gm.arc_length(6, 60) - 2 * math.pi) < EPS
    assert abs(gm.sector_area(6, 60) - 6 * math.pi) < EPS


def test_circle_area_circumference():
    assert abs(gm.circle_area(5) - math.pi * 25) < EPS
    assert abs(gm.circumference(5) - 10 * math.pi) < EPS


def test_solid_volumes():
    assert abs(gm.solid_volume("cube", {"edge": 4}) - 64) < EPS
    assert abs(gm.solid_volume("cylinder", {"r": 3, "h": 10}) - math.pi * 9 * 10) < EPS
    assert abs(gm.solid_volume("sphere", {"r": 3}) - 4 / 3 * math.pi * 27) < EPS
    assert abs(gm.solid_surface("cube", {"edge": 5}) - 150) < EPS


def test_polygon_formulas():
    assert gm.regular_polygon_interior_sum(7) == 900
    assert abs(gm.regular_polygon_each_interior(6) - 120) < EPS
    assert gm.polygon_diagonals(5) == 5


def test_inscribed_is_half_central():
    """Construct central angle ∠AMB = 80° on a circle, an inscribed point C on
    the major arc, and assert the kernel measures ∠ACB = 40°."""
    central = 80.0
    fig = gk.solve([
        {"op": "point", "name": "M", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "M", "r": 5},
        {"op": "point_on_circle", "name": "A", "circle": "k", "theta": 0},
        {"op": "point_on_circle", "name": "B", "circle": "k", "theta": central},
        # C on the major arc (opposite side)
        {"op": "point_on_circle", "name": "C", "circle": "k", "theta": 220},
        {"op": "polygon", "name": "T", "verts": ["A", "C", "B"]},
    ])
    measured_central = gm.angle(fig, "M", "A", "B")
    inscribed = gm.angle(fig, "C", "A", "B")
    assert abs(measured_central - central) < 1e-4
    assert abs(inscribed - central / 2) < 1e-4


def test_query_dispatch_ratio():
    fig = gk.solve([
        {"op": "point", "name": "A", "x": 0, "y": 0},
        {"op": "point", "name": "B", "x": 8, "y": 0},
        {"op": "point", "name": "C", "x": 0, "y": 0},
        {"op": "point", "name": "D", "x": 4, "y": 0},
    ])
    r = gm.query(fig, {"op": "ratio",
                       "of1": {"op": "length", "a": "A", "b": "B"},
                       "of2": {"op": "length", "a": "C", "b": "D"}})
    assert abs(r - 2.0) < EPS


def _triangle_fig():
    # أ(0,0) ب(15,0) ج(near top): a 9,12,15-ish triangle placed by coordinates.
    return gk.solve([
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": 15, "y": 0},
        {"op": "point", "name": "ج", "x": 15, "y": 9},  # side ب-ج = 9
    ])


def test_verify_figure_accepts_matching_labels():
    import geo_scenarios as gs
    fig = _triangle_fig()
    spec = {
        "answer": "36",  # not cross-checked here (perimeter would be ~44); use a side op
        "measure": {"op": "length", "a": "ب", "b": "ج"},
        "annotations": [
            {"type": "side", "a": "ب", "b": "ج", "text": "٩"},   # matches figure (9)
        ],
    }
    measured = gm.query(fig, spec["measure"])
    spec["answer"] = "9"
    ok, reason = gs._verify_figure(fig, spec, measured)
    assert ok, reason


def test_verify_figure_rejects_label_contradicting_figure():
    import geo_scenarios as gs
    fig = _triangle_fig()
    # The stem/label claims side ب-ج = 13, but the figure draws it as 9.
    spec = {
        "answer": "9",
        "measure": {"op": "length", "a": "ب", "b": "ج"},
        "annotations": [
            {"type": "side", "a": "ب", "b": "ج", "text": "١٣"},  # WRONG vs figure
        ],
    }
    measured = gm.query(fig, spec["measure"])
    ok, reason = gs._verify_figure(fig, spec, measured)
    assert not ok and "label" in reason, reason


def test_verify_figure_rejects_answer_not_matching_measured():
    import geo_scenarios as gs
    fig = _triangle_fig()
    spec = {"answer": "99", "measure": {"op": "length", "a": "ب", "b": "ج"},
            "annotations": []}
    measured = gm.query(fig, spec["measure"])  # == 9
    ok, reason = gs._verify_figure(fig, spec, measured)
    assert not ok and "answer" in reason, reason


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
