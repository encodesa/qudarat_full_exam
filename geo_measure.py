#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement / verifier layer for the geometry kernel.

`query(figure, spec) -> float` computes a derived quantity from the SOLVED
coordinates. This is the source of the correct answer — closed-form, no LLM. The
scenario declares what to ask; the value returned here is the verified answer.

Solid quantities (volume/surface) are computed from explicit solid params passed
in the spec, since a 3D solid is carried as parameters rather than 2D points.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from geo_kernel import Figure, GeoError

EPS = 1e-9


def _v(fig: Figure, name: str) -> np.ndarray:
    return fig.vec(name)


def length(fig: Figure, a: str, b: str) -> float:
    return float(np.hypot(*(_v(fig, b) - _v(fig, a))))


def angle(fig: Figure, vertex: str, a: str, b: str) -> float:
    """Interior angle at `vertex` between rays to a and b, in degrees."""
    va = _v(fig, a) - _v(fig, vertex)
    vb = _v(fig, b) - _v(fig, vertex)
    na, nb = np.hypot(*va), np.hypot(*vb)
    if na < EPS or nb < EPS:
        raise GeoError("angle at coincident points")
    cosang = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _poly_pts(fig: Figure, verts):
    if isinstance(verts, str):           # a named polygon
        verts = fig.polygons[verts].verts if verts in fig.polygons else [verts]
    return [_v(fig, v) for v in verts]


def area_polygon(fig: Figure, verts) -> float:
    pts = _poly_pts(fig, verts)
    n = len(pts)
    if n < 3:
        raise GeoError("polygon area needs >=3 vertices")
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def perimeter(fig: Figure, verts) -> float:
    pts = _poly_pts(fig, verts)
    n = len(pts)
    return float(sum(np.hypot(*(pts[(i + 1) % n] - pts[i])) for i in range(n)))


def area_triangle_base_height(base: float, height: float) -> float:
    return 0.5 * float(base) * float(height)


def circle_area(r: float) -> float:
    return math.pi * float(r) ** 2


def circumference(r: float) -> float:
    return 2 * math.pi * float(r)


def arc_length(r: float, deg: float) -> float:
    return float(r) * math.radians(float(deg))


def sector_area(r: float, deg: float) -> float:
    return 0.5 * float(r) ** 2 * math.radians(float(deg))


def is_right_triangle(fig: Figure, verts) -> bool:
    pts = _poly_pts(fig, verts)
    if len(pts) != 3:
        return False
    for i in range(3):
        a = pts[(i - 1) % 3] - pts[i]
        b = pts[(i + 1) % 3] - pts[i]
        na, nb = np.hypot(*a), np.hypot(*b)
        if na < EPS or nb < EPS:
            continue
        if abs(float(np.dot(a, b)) / (na * nb)) < 1e-4:
            return True
    return False


# ---- solids: params in the spec, not points ----
def solid_volume(kind: str, p: Dict[str, Any]) -> float:
    kind = (kind or "").lower()
    g = lambda k: float(p[k])
    if kind == "cube":
        return g("edge") ** 3
    if kind in ("box", "cuboid", "rectangular_prism"):
        return g("width") * g("depth") * g("height")
    if kind == "cylinder":
        return math.pi * g("r") ** 2 * g("h")
    if kind == "cone":
        return math.pi * g("r") ** 2 * g("h") / 3.0
    if kind == "sphere":
        return 4.0 / 3.0 * math.pi * g("r") ** 3
    raise GeoError(f"unknown solid kind for volume: {kind}")


def solid_surface(kind: str, p: Dict[str, Any]) -> float:
    kind = (kind or "").lower()
    g = lambda k: float(p[k])
    if kind == "cube":
        return 6 * g("edge") ** 2
    if kind in ("box", "cuboid", "rectangular_prism"):
        w, d, h = g("width"), g("depth"), g("height")
        return 2 * (w * d + w * h + d * h)
    if kind == "cylinder":
        r, h = g("r"), g("h")
        return 2 * math.pi * r * (r + h)
    if kind == "cone":
        r, h = g("r"), g("h")
        sl = math.hypot(r, h)
        return math.pi * r * (r + sl)
    if kind == "sphere":
        return 4 * math.pi * g("r") ** 2
    raise GeoError(f"unknown solid kind for surface: {kind}")


def regular_polygon_interior_sum(n: int) -> float:
    return (int(n) - 2) * 180.0


def regular_polygon_each_interior(n: int) -> float:
    return (int(n) - 2) * 180.0 / int(n)


def polygon_diagonals(n: int) -> int:
    n = int(n)
    return n * (n - 3) // 2


# ============================================================
# dispatch — spec = {"op": ..., ...}
# ============================================================
def query(fig: Figure, spec: Dict[str, Any]) -> float:
    op = (spec.get("op") or "").lower()
    if op == "length":
        return length(fig, spec["a"], spec["b"])
    if op == "angle":
        return angle(fig, spec["vertex"], spec["a"], spec["b"])
    if op == "area_polygon":
        return area_polygon(fig, spec["verts"])
    if op == "perimeter":
        return perimeter(fig, spec["verts"])
    if op == "area_triangle_bh":
        return area_triangle_base_height(spec["base"], spec["height"])
    if op == "circle_area":
        return circle_area(spec["r"])
    if op == "circumference":
        return circumference(spec["r"])
    if op == "arc_length":
        return arc_length(spec["r"], spec["deg"])
    if op == "sector_area":
        return sector_area(spec["r"], spec["deg"])
    if op == "ratio":
        denom = query(fig, spec["of2"])
        if abs(denom) < EPS:
            raise GeoError("ratio division by zero")
        return query(fig, spec["of1"]) / denom
    if op == "solid_volume":
        return solid_volume(spec["kind"], spec.get("params", spec))
    if op == "solid_surface":
        return solid_surface(spec["kind"], spec.get("params", spec))
    if op == "interior_sum":
        return regular_polygon_interior_sum(spec["n"])
    if op == "each_interior":
        return regular_polygon_each_interior(spec["n"])
    if op == "diagonals":
        return float(polygon_diagonals(spec["n"]))
    if op == "const":                       # pass-through computed scalar
        return float(spec["value"])
    raise GeoError(f"unknown measure op '{op}'")
