#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry construction kernel.

A small constructive (compass-straightedge style) interpreter. A *program* is an
ordered list of steps; each step deterministically computes exact coordinates for
a newly named object from objects solved by earlier steps. The solved Figure is
the single source of truth: `geo_measure` derives the answer from it and
`geo_render` draws it.

Design choices (see plan):
  * Pure constructive, not a numeric constraint solver. Every primitive is a
    closed-form computation, so the same program always yields identical
    coordinates — essential because the answer is measured from them.
  * Branch ambiguity (line/circle, circle/circle, tangents) is resolved by an
    explicit `pick` in the step, never by solver luck.
  * A reference to an unsolved name, or a geometrically impossible request, raises
    GeoError; the generator catches it and resamples (never emits a wrong figure).

No LLM, no matplotlib, no I/O here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-9


class GeoError(Exception):
    """Construction failed (bad reference, impossible intersection, degenerate)."""


# ============================================================
# Object kinds — all coordinates are plain floats (numpy under the hood)
# ============================================================
@dataclass
class Point:
    x: float
    y: float

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def np(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class LineLike:
    """A line, ray or segment defined by two endpoint names + a kind."""
    a: str               # endpoint name
    b: str               # endpoint name
    kind: str            # "line" | "ray" | "segment"


@dataclass
class Circle:
    center: str          # point name
    r: float


@dataclass
class Polygon:
    verts: List[str]     # ordered point names


@dataclass
class Figure:
    points: Dict[str, Point] = field(default_factory=dict)
    lines: Dict[str, LineLike] = field(default_factory=dict)
    circles: Dict[str, Circle] = field(default_factory=dict)
    polygons: Dict[str, Polygon] = field(default_factory=dict)
    program: List[dict] = field(default_factory=list)

    # ---- lookups (raise GeoError on a missing/wrong-typed reference) ----
    def pt(self, name: str) -> Point:
        if name not in self.points:
            raise GeoError(f"unknown point '{name}'")
        return self.points[name]

    def vec(self, name: str) -> np.ndarray:
        return self.pt(name).np

    def line(self, name: str) -> LineLike:
        if name not in self.lines:
            raise GeoError(f"unknown line '{name}'")
        return self.lines[name]

    def circ(self, name: str) -> Circle:
        if name not in self.circles:
            raise GeoError(f"unknown circle '{name}'")
        return self.circles[name]

    def all_point_names(self) -> List[str]:
        return list(self.points.keys())


# ============================================================
# small vector helpers
# ============================================================
def _np(p) -> np.ndarray:
    return np.array([float(p[0]), float(p[1])], dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n < EPS:
        raise GeoError("zero-length direction")
    return v / n


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


def _rotate(v: np.ndarray, deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _line_dir(fig: Figure, ln: LineLike) -> np.ndarray:
    return _unit(fig.vec(ln.b) - fig.vec(ln.a))


def _line_point(fig: Figure, ln: LineLike) -> np.ndarray:
    return fig.vec(ln.a)


# ============================================================
# intersection math (closed form; returns 0/1/2 points)
# ============================================================
def _intersect_line_line(p1, d1, p2, d2) -> np.ndarray:
    # solve p1 + t d1 = p2 + s d2
    denom = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(denom) < EPS:
        raise GeoError("parallel lines do not intersect")
    rhs = p2 - p1
    t = (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / denom
    return p1 + t * d1


def _intersect_line_circle(p, d, c, r) -> List[np.ndarray]:
    # points on line p + t d at distance r from c
    d = _unit(d)
    f = p - c
    b = 2 * np.dot(f, d)
    cc = np.dot(f, f) - r * r
    disc = b * b - 4 * cc
    if disc < -1e-7:
        return []
    disc = max(disc, 0.0)
    sq = math.sqrt(disc)
    t1 = (-b + sq) / 2
    t2 = (-b - sq) / 2
    if abs(t1 - t2) < EPS:
        return [p + t1 * d]
    return [p + t1 * d, p + t2 * d]


def _intersect_circle_circle(c1, r1, c2, r2) -> List[np.ndarray]:
    d = float(np.hypot(*(c2 - c1)))
    if d < EPS:
        raise GeoError("concentric circles")
    if d > r1 + r2 + 1e-7 or d < abs(r1 - r2) - 1e-7:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    h = math.sqrt(max(h2, 0.0))
    mid = c1 + a * (c2 - c1) / d
    perp = _rot90(_unit(c2 - c1))
    if h < EPS:
        return [mid]
    return [mid + h * perp, mid - h * perp]


def _pick(points: List[np.ndarray], pick, ref_center: Optional[np.ndarray] = None,
          ref_from: Optional[np.ndarray] = None) -> np.ndarray:
    """Choose one solution from 0/1/2 candidates by an explicit rule."""
    if not points:
        raise GeoError("no intersection")
    if len(points) == 1:
        return points[0]
    pk = (pick or "").lower() if isinstance(pick, str) else pick
    if isinstance(pk, int):
        return points[pk % len(points)]
    if pk in ("upper", "top"):
        return max(points, key=lambda q: q[1])
    if pk in ("lower", "bottom"):
        return min(points, key=lambda q: q[1])
    if pk in ("left",):
        return min(points, key=lambda q: q[0])
    if pk in ("right",):
        return max(points, key=lambda q: q[0])
    if pk in ("far", "farther") and ref_from is not None:
        return max(points, key=lambda q: np.hypot(*(q - ref_from)))
    if pk in ("near", "nearer") and ref_from is not None:
        return min(points, key=lambda q: np.hypot(*(q - ref_from)))
    # default: first
    return points[0]


# ============================================================
# primitive handlers — each: (fig, step) -> None (mutates fig)
# ============================================================
def _need(step: dict, *keys) -> tuple:
    out = []
    for k in keys:
        if k not in step:
            raise GeoError(f"op '{step.get('op')}' missing '{k}'")
        out.append(step[k])
    return tuple(out)


def _add_point(fig: Figure, name: str, xy: np.ndarray) -> None:
    if not name:
        raise GeoError("point step missing 'name'")
    fig.points[name] = Point(float(xy[0]), float(xy[1]))


# -- placement --
def _op_point(fig, s):
    name, x, y = _need(s, "name", "x", "y")
    _add_point(fig, name, _np((x, y)))


def _op_point_polar(fig, s):
    name, center, r, theta = _need(s, "name", "center", "r", "theta")
    c = fig.vec(center)
    t = math.radians(float(theta))
    _add_point(fig, name, c + float(r) * np.array([math.cos(t), math.sin(t)]))


# -- linear objects --
def _op_linelike(fig, s, kind):
    name, a, b = _need(s, "name", "a", "b")
    fig.pt(a); fig.pt(b)  # validate refs
    fig.lines[s["name"]] = LineLike(a=a, b=b, kind=kind)


def _op_point_on_line(fig, s, clamp):
    name, a, b, t = _need(s, "name", "a", "b", "t")
    pa, pb = fig.vec(a), fig.vec(b)
    tt = float(t)
    if clamp:
        tt = min(1.0, max(0.0, tt))
    _add_point(fig, name, pa + tt * (pb - pa))


def _op_midpoint(fig, s):
    name, a, b = _need(s, "name", "a", "b")
    _add_point(fig, name, (fig.vec(a) + fig.vec(b)) / 2.0)


def _op_division_point(fig, s):
    name, a, b, m, n = _need(s, "name", "a", "b", "m", "n")
    m, n = float(m), float(n)
    if abs(m + n) < EPS:
        raise GeoError("division ratio sums to zero")
    _add_point(fig, name, (n * fig.vec(a) + m * fig.vec(b)) / (m + n))


# -- circles --
def _op_circle(fig, s):
    name, center, r = _need(s, "name", "center", "r")
    fig.pt(center)
    if float(r) <= 0:
        raise GeoError("circle radius must be positive")
    fig.circles[name] = Circle(center=center, r=float(r))


def _op_circle_through(fig, s):
    name, center, through = _need(s, "name", "center", "through")
    r = float(np.hypot(*(fig.vec(through) - fig.vec(center))))
    if r <= EPS:
        raise GeoError("circle_through has zero radius")
    fig.circles[name] = Circle(center=center, r=r)


def _op_point_on_circle(fig, s):
    name, circle, theta = _need(s, "name", "circle", "theta")
    c = fig.circ(circle)
    cc = fig.vec(c.center)
    t = math.radians(float(theta))
    _add_point(fig, name, cc + c.r * np.array([math.cos(t), math.sin(t)]))


# -- perpendicular / parallel / foot --
def _op_perpendicular(fig, s):
    name, through, to_line = _need(s, "name", "through", "to_line")
    ln = fig.line(to_line)
    d = _rot90(_line_dir(fig, ln))
    p = fig.vec(through)
    # represent as a line via two points: through and through+d
    helper = f"__{name}_d"
    _add_point(fig, helper, p + d)
    fig.lines[name] = LineLike(a=through, b=helper, kind="line")


def _op_parallel(fig, s):
    name, through, to_line = _need(s, "name", "through", "to_line")
    ln = fig.line(to_line)
    d = _line_dir(fig, ln)
    p = fig.vec(through)
    helper = f"__{name}_d"
    _add_point(fig, helper, p + d)
    fig.lines[name] = LineLike(a=through, b=helper, kind="line")


def _op_foot(fig, s):
    name, pt, to_line = _need(s, "name", "pt", "to_line")
    ln = fig.line(to_line)
    a = fig.vec(ln.a)
    d = _line_dir(fig, ln)
    p = fig.vec(pt)
    f = a + np.dot(p - a, d) * d
    _add_point(fig, name, f)


def _op_perp_bisector(fig, s):
    name, a, b = _need(s, "name", "a", "b")
    pa, pb = fig.vec(a), fig.vec(b)
    mid = (pa + pb) / 2.0
    d = _rot90(_unit(pb - pa))
    m_name = f"__{name}_m"
    h_name = f"__{name}_h"
    _add_point(fig, m_name, mid)
    _add_point(fig, h_name, mid + d)
    fig.lines[name] = LineLike(a=m_name, b=h_name, kind="line")


# -- intersections --
def _op_intersect_ll(fig, s):
    name, l1, l2 = _need(s, "name", "line1", "line2")
    a, b = fig.line(l1), fig.line(l2)
    p = _intersect_line_line(_line_point(fig, a), _line_dir(fig, a),
                             _line_point(fig, b), _line_dir(fig, b))
    _add_point(fig, name, p)


def _op_intersect_lc(fig, s):
    name, line, circle = _need(s, "name", "line", "circle")
    ln = fig.line(line)
    c = fig.circ(circle)
    pts = _intersect_line_circle(_line_point(fig, ln), _line_dir(fig, ln),
                                 fig.vec(c.center), c.r)
    chosen = _pick(pts, s.get("pick"), ref_center=fig.vec(c.center),
                   ref_from=fig.vec(ln.a))
    _add_point(fig, name, chosen)


def _op_intersect_cc(fig, s):
    name, c1, c2 = _need(s, "name", "c1", "c2")
    a, b = fig.circ(c1), fig.circ(c2)
    pts = _intersect_circle_circle(fig.vec(a.center), a.r, fig.vec(b.center), b.r)
    chosen = _pick(pts, s.get("pick"))
    _add_point(fig, name, chosen)


# -- polygons --
def _op_polygon(fig, s):
    name, verts = _need(s, "name", "verts")
    for v in verts:
        fig.pt(v)
    if len(verts) < 3:
        raise GeoError("polygon needs >=3 vertices")
    fig.polygons[name] = Polygon(verts=list(verts))


def _op_regular_polygon(fig, s):
    name, n, center, r = _need(s, "name", "n", "center", "r")
    names = s.get("names") or [f"{name}_{i}" for i in range(int(n))]
    start = math.radians(float(s.get("start_theta", 90)))
    cc = fig.vec(center)
    rr = float(r)
    vnames = []
    for i in range(int(n)):
        ang = start + i * 2 * math.pi / int(n)
        pn = names[i]
        _add_point(fig, pn, cc + rr * np.array([math.cos(ang), math.sin(ang)]))
        vnames.append(pn)
    fig.polygons[name] = Polygon(verts=vnames)


def _op_triangle_sss(fig, s):
    """Place a triangle from three side lengths. names=[A,B,C], sides a=BC,b=CA,c=AB."""
    names = s.get("names") or ["A", "B", "C"]
    a = float(s["a"]); b = float(s["b"]); c = float(s["c"])  # a=BC, b=CA, c=AB
    # triangle inequality
    if a + b <= c + 1e-9 or b + c <= a + 1e-9 or a + c <= b + 1e-9:
        raise GeoError("triangle inequality violated")
    A = np.array([0.0, 0.0])
    B = np.array([c, 0.0])
    # C: |C-A|=b, |C-B|=a
    pts = _intersect_circle_circle(A, b, B, a)
    if not pts:
        raise GeoError("triangle_sss: circles miss")
    C = max(pts, key=lambda q: q[1])  # upper solution
    for nm, p in zip(names, (A, B, C)):
        _add_point(fig, nm, p)
    fig.polygons[s["name"]] = Polygon(verts=list(names))


# -- angle / transforms --
def _op_ray_at_angle(fig, s):
    """New point: rotate the ray (vertex->from_pt) by ±measure deg, at distance
    `length` (default = |vertex-from_pt|). Used to build 'given angle' figures."""
    name, vertex, from_pt, measure = _need(s, "name", "vertex", "from_pt", "measure")
    v = fig.vec(vertex)
    base = fig.vec(from_pt) - v
    length = float(s.get("length") or np.hypot(*base))
    side = (s.get("side") or "ccw").lower()
    deg = float(measure) * (1 if side in ("ccw", "left", "+") else -1)
    d = _rotate(_unit(base), deg)
    _add_point(fig, name, v + length * d)


def _op_reflect(fig, s):
    name, pt, over_line = _need(s, "name", "pt", "over_line")
    ln = fig.line(over_line)
    a = fig.vec(ln.a)
    d = _line_dir(fig, ln)
    p = fig.vec(pt)
    foot = a + np.dot(p - a, d) * d
    _add_point(fig, name, 2 * foot - p)


def _op_rotate(fig, s):
    name, pt, center, deg = _need(s, "name", "pt", "center", "deg")
    c = fig.vec(center)
    _add_point(fig, name, c + _rotate(fig.vec(pt) - c, float(deg)))


def _op_translate(fig, s):
    name, pt, dx, dy = _need(s, "name", "pt", "dx", "dy")
    _add_point(fig, name, fig.vec(pt) + _np((dx, dy)))


# -- circle specials --
def _op_tangent_from(fig, s):
    """Tangent point on `circle` from external point `ext`. Two solutions; pick."""
    name, ext, circle = _need(s, "name", "ext", "circle")
    c = fig.circ(circle)
    cc = fig.vec(c.center)
    e = fig.vec(ext)
    d = float(np.hypot(*(e - cc)))
    if d < c.r - 1e-7:
        raise GeoError("tangent_from: point inside circle")
    # tangent points lie on circle AND on the Thales circle of diameter (e, center)
    mid = (e + cc) / 2.0
    rmid = d / 2.0
    pts = _intersect_circle_circle(cc, c.r, mid, rmid)
    chosen = _pick(pts, s.get("pick"))
    _add_point(fig, name, chosen)


def _op_circumcircle(fig, s):
    name, verts = _need(s, "name", "verts")
    if len(verts) < 3:
        raise GeoError("circumcircle needs 3 points")
    p = [fig.vec(v) for v in verts[:3]]
    cen, r = _circumcenter(p[0], p[1], p[2])
    cname = s.get("center_name") or f"__{name}_c"
    _add_point(fig, cname, cen)
    fig.circles[name] = Circle(center=cname, r=r)


def _op_incircle(fig, s):
    name, verts = _need(s, "name", "verts")
    A, B, C = (fig.vec(v) for v in verts[:3])
    a = float(np.hypot(*(B - C)))
    b = float(np.hypot(*(C - A)))
    c = float(np.hypot(*(A - B)))
    per = a + b + c
    if per < EPS:
        raise GeoError("degenerate triangle")
    inc = (a * A + b * B + c * C) / per
    s_ = per / 2.0
    area = abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / 2.0
    r = area / s_ if s_ > EPS else 0.0
    cname = s.get("center_name") or f"__{name}_c"
    _add_point(fig, cname, inc)
    fig.circles[name] = Circle(center=cname, r=r)


def _circumcenter(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < EPS:
        raise GeoError("collinear points have no circumcircle")
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / d
    cen = np.array([ux, uy])
    r = float(np.hypot(*(cen - a)))
    return cen, r


# ============================================================
# dispatch table
# ============================================================
_OPS = {
    "point": _op_point,
    "point_polar": _op_point_polar,
    "segment": lambda f, s: _op_linelike(f, s, "segment"),
    "line": lambda f, s: _op_linelike(f, s, "line"),
    "ray": lambda f, s: _op_linelike(f, s, "ray"),
    "point_on_segment": lambda f, s: _op_point_on_line(f, s, clamp=True),
    "point_on_line": lambda f, s: _op_point_on_line(f, s, clamp=False),
    "midpoint": _op_midpoint,
    "division_point": _op_division_point,
    "circle": _op_circle,
    "circle_through": _op_circle_through,
    "point_on_circle": _op_point_on_circle,
    "perpendicular": _op_perpendicular,
    "parallel": _op_parallel,
    "foot": _op_foot,
    "perp_bisector": _op_perp_bisector,
    "intersect_ll": _op_intersect_ll,
    "intersect_lc": _op_intersect_lc,
    "intersect_cc": _op_intersect_cc,
    "polygon": _op_polygon,
    "regular_polygon": _op_regular_polygon,
    "triangle_sss": _op_triangle_sss,
    "ray_at_angle": _op_ray_at_angle,
    "reflect": _op_reflect,
    "rotate": _op_rotate,
    "translate": _op_translate,
    "tangent_from": _op_tangent_from,
    "circumcircle": _op_circumcircle,
    "incircle": _op_incircle,
}


def solve(program) -> Figure:
    """Interpret a construction program (list of steps OR {'program': [...]})."""
    if isinstance(program, dict):
        steps = program.get("program") or []
    else:
        steps = program
    fig = Figure(program=list(steps))
    for i, step in enumerate(steps):
        op = step.get("op")
        handler = _OPS.get(op)
        if handler is None:
            raise GeoError(f"step {i}: unknown op '{op}'")
        try:
            handler(fig, step)
        except GeoError:
            raise
        except Exception as e:  # numeric / key errors -> uniform GeoError
            raise GeoError(f"step {i} ('{op}'): {e}") from e
    return fig


def sanity_check(fig: Figure, min_edge: float = 0.05, min_angle_deg: float = 8.0) -> None:
    """Reject degenerate figures (tiny edges, near-collinear polygons). Raises GeoError."""
    for name, poly in fig.polygons.items():
        pts = [fig.vec(v) for v in poly.verts]
        n = len(pts)
        for k in range(n):
            e = float(np.hypot(*(pts[(k + 1) % n] - pts[k])))
            if e < min_edge:
                raise GeoError(f"polygon '{name}' has a near-zero edge")
        for k in range(n):
            a = pts[(k - 1) % n] - pts[k]
            b = pts[(k + 1) % n] - pts[k]
            na, nb = np.hypot(*a), np.hypot(*b)
            if na < EPS or nb < EPS:
                raise GeoError(f"polygon '{name}' degenerate vertex")
            cosang = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
            ang = math.degrees(math.acos(cosang))
            if ang < min_angle_deg or ang > 180 - min_angle_deg + 1e-6:
                raise GeoError(f"polygon '{name}' near-degenerate angle {ang:.1f}")
