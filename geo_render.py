#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Universal renderer: draw ANY solved kernel Figure in Qudrat past-exam style.

The renderer NEVER solves geometry (the kernel did) and NEVER annotates the
asked quantity (the answer). The scenario passes an explicit `annotations` list
describing only what is GIVEN in the figure.

annotations entries (all optional):
  {"type":"side","a":"أ","b":"ب","text":"٥"}            length label on an edge
  {"type":"angle","vertex":"ج","a":"أ","b":"ب","text":"٣٠°"}   angle arc + value
  {"type":"right_angle","vertex":"أ","a":"ب","b":"ج"}    filled-square marker
  {"type":"segment","a":"أ","b":"ج","dashed":false}      extra line not in figure
  {"type":"label","point":"أ"}                            force-label a point
  {"type":"shade","outer":[pts or circle],"holes":[...]}  shaded compound region

`render_construction(figure, annotations, out_path, hide=set())` draws every
object in the figure, auto-labels real points (helper names starting "__" and
those in `hide` are skipped), then applies annotations.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from matplotlib import patches as mpatches

import geo_style as S
from geo_kernel import Figure


def _all_drawn_points(fig: Figure) -> List[np.ndarray]:
    return [p.np for p in fig.points.values()]


def _bbox(fig: Figure, pad: float = 0.25):
    pts = _all_drawn_points(fig)
    if not pts:
        return (-1, -1, 1, 1)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs) or 1.0
    h = max(ys) - min(ys) or 1.0
    m = pad * max(w, h)
    return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)


def _clip_line_to_bbox(p, d, bbox):
    """Return two endpoints where the infinite line p+t*d crosses the bbox."""
    x0, y0, x1, y1 = bbox
    ts = []
    for (axis, lo, hi) in ((0, x0, x1), (1, y0, y1)):
        if abs(d[axis]) > 1e-12:
            ts.append((lo - p[axis]) / d[axis])
            ts.append((hi - p[axis]) / d[axis])
    if not ts:
        return None
    pts = []
    for t in ts:
        q = p + t * d
        if x0 - 1e-6 <= q[0] <= x1 + 1e-6 and y0 - 1e-6 <= q[1] <= y1 + 1e-6:
            pts.append(q)
    if len(pts) < 2:
        return None
    pts.sort(key=lambda q: (q[0], q[1]))
    return pts[0], pts[-1]


def _scene_span(fig: Figure) -> float:
    x0, y0, x1, y1 = _bbox(fig, pad=0.0)
    return max(x1 - x0, y1 - y0, 1.0)


def render_construction(fig: Figure, annotations: Optional[List[dict]] = None,
                        out_path: str = "out.png", hide: Optional[Iterable[str]] = None,
                        solids: Optional[List[dict]] = None) -> bool:
    annotations = annotations or []
    hide = set(hide or [])
    try:
        figu, ax = S.new_axes()
        bbox = _bbox(fig)
        span = _scene_span(fig)

        # --- shaded regions first (under the strokes) ---
        for an in annotations:
            if an.get("type") == "shade":
                _draw_shade(ax, fig, an)

        # --- polygons ---
        for poly in fig.polygons.values():
            pts = [fig.pt(v).xy for v in poly.verts]
            S.stroke_polygon(ax, pts)

        # --- circles ---
        for c in fig.circles.values():
            S.stroke_circle(ax, fig.pt(c.center).xy, c.r)

        # --- lines / rays / segments ---
        for lname, ln in fig.lines.items():
            if str(lname).startswith("__"):
                continue  # construction-only helper line (not drawn)
            a = fig.vec(ln.a)
            b = fig.vec(ln.b)
            if ln.kind == "segment":
                S.stroke_segment(ax, a, b)
            elif ln.kind == "ray":
                d = b - a
                n = np.hypot(*d) or 1.0
                far = a + d / n * span * 2.0
                clipped = _clip_line_to_bbox(a, d / n, bbox)
                end = clipped[1] if clipped else far
                S.stroke_segment(ax, a, end)
            else:  # infinite line
                d = b - a
                n = np.hypot(*d) or 1.0
                seg = _clip_line_to_bbox(a, d / n, bbox)
                if seg:
                    S.stroke_segment(ax, seg[0], seg[1])

        # --- 3D solids (pre-projected points passed as edge list) ---
        for sol in (solids or []):
            _draw_solid_edges(ax, sol)

        # --- auto-label real points (collision-aware so labels never overlap) ---
        forced = {a["point"] for a in annotations if a.get("type") == "label"}
        # centroid from VISIBLE labeled points only, so labels push truly outward
        cen = _label_centroid(fig, hide - forced)
        label_items = []
        for name, p in fig.points.items():
            if name.startswith("__"):
                continue
            if name in hide and name not in forced:
                continue
            label_items.append((p.xy, name))
        # Collect every drawn segment so labels can be pushed off lines/edges too
        # (a vertex label sitting on a cube edge was a common 'ugly figure' case).
        segs = []
        for poly in fig.polygons.values():
            pv = [fig.pt(v).xy for v in poly.verts]
            for i in range(len(pv)):
                segs.append((pv[i], pv[(i + 1) % len(pv)]))
        for lname, ln in fig.lines.items():
            if str(lname).startswith("__"):
                continue
            segs.append((fig.pt(ln.a).xy, fig.pt(ln.b).xy))
        for an in annotations:
            if an.get("type") == "segment":
                segs.append((fig.pt(an["a"]).xy, fig.pt(an["b"]).xy))
        S.vertex_labels(ax, label_items, cen, span, segments=segs)

        # --- annotations (given measurements only) ---
        for an in annotations:
            t = an.get("type")
            if t == "side":
                S.side_label(ax, fig.pt(an["a"]).xy, fig.pt(an["b"]).xy,
                             an.get("text"), cen)
            elif t == "angle":
                _ann_angle(ax, fig, an, span, right=False)
            elif t == "right_angle":
                _ann_angle(ax, fig, an, span, right=True)
            elif t == "segment":
                S.stroke_segment(ax, fig.pt(an["a"]).xy, fig.pt(an["b"]).xy,
                                 lw=S.GEO_ARC_LW,
                                 ls="--" if an.get("dashed") else "-")

        return S.finalize(figu, ax, out_path)
    except Exception as e:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        # surface the error to the caller's log if present
        print(f"[geo_render] failed: {e}")
        return False


def _label_centroid(fig: Figure, hide=None):
    hide = set(hide or [])
    pts = [p.np for n, p in fig.points.items()
           if not n.startswith("__") and n not in hide]
    if not pts:
        return (0.0, 0.0)
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))


def _ann_angle(ax, fig: Figure, an: dict, span: float, right: bool):
    v = fig.pt(an["vertex"]).xy
    a = fig.pt(an["a"]).xy
    b = fig.pt(an["b"]).xy
    if right:
        S.right_angle_marker(ax, v, a, b, size=0.10 * span)
    else:
        S.angle_arc(ax, v, a, b, an.get("text"), radius=0.13 * span)


# ---- shaded regions via shapely (graceful fallback if unavailable) ----
def _region_geom(fig: Figure, spec):
    """Build a shapely geometry from a region spec: a circle dict or a vertex list."""
    from shapely.geometry import Polygon as ShPoly, Point as ShPoint
    if isinstance(spec, dict) and spec.get("circle"):
        c = fig.circ(spec["circle"])
        return ShPoint(*fig.pt(c.center).xy).buffer(c.r, resolution=64)
    if isinstance(spec, dict) and spec.get("verts"):
        return ShPoly([fig.pt(v).xy for v in spec["verts"]])
    if isinstance(spec, list):
        return ShPoly([fig.pt(v).xy for v in spec])
    raise ValueError("bad region spec")


def _draw_shade(ax, fig: Figure, an: dict):
    try:
        from shapely.geometry import MultiPolygon
        region = _region_geom(fig, an["outer"])
        for h in (an.get("holes") or []):
            region = region.difference(_region_geom(fig, h))
        geoms = region.geoms if isinstance(region, MultiPolygon) else [region]
        for g in geoms:
            if g.is_empty:
                continue
            xs, ys = g.exterior.xy
            ax.fill(list(xs), list(ys), facecolor="none", hatch="///",
                    edgecolor=S.GEO_COLOR, linewidth=0)
    except Exception as e:
        print(f"[geo_render] shade skipped: {e}")


# ---- 3D solids: caller passes projected edges ----
def _draw_solid_edges(ax, sol: dict):
    """sol = {"edges":[((x,y),(x,y),dashed_bool), ...], "labels":[(x,y,text), ...]}"""
    for e in sol.get("edges", []):
        p1, p2 = e[0], e[1]
        dashed = e[2] if len(e) > 2 else False
        S.stroke_segment(ax, p1, p2, lw=S.GEO_ARC_LW if dashed else S.GEO_LW,
                         ls="--" if dashed else "-")
    for (x, y, text) in sol.get("labels", []):
        ax.text(x, y, S.ar(str(text)), ha="center", va="center",
                fontsize=S.GEO_FONT - 1, color=S.GEO_COLOR)
