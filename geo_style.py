#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Qudrat-style matplotlib drawing primitives.

Extracted so both geo_render.py and generate_visual.py can draw figures in the
same past-exam style WITHOUT pulling in the LLM pipeline (app6_final / genai).
No API key needed to import this module.

Style target (from real Qudrat past-exam figures): solid black lines lw~2 on
white, filled-square right-angle marker, thin angle arcs with values, Arabic
vertex labels (أ ب ج د، center م) rendered RTL via arabic_reshaper + bidi.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches as mpatches
import arabic_reshaper
from bidi.algorithm import get_display

# ---- style constants ----
GEO_LW = 2.0
GEO_ARC_LW = 1.2
GEO_COLOR = "black"
GEO_FONT = 16

ARABIC_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",
]

_AR_FONT = None
for _p in ARABIC_FONT_CANDIDATES:
    if Path(_p).exists():
        try:
            font_manager.fontManager.addfont(_p)
            _AR_FONT = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            continue
if _AR_FONT:
    plt.rcParams["font.family"] = _AR_FONT


_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")


def ar(text) -> str:
    """Reshape + bidi so Arabic renders correctly in matplotlib.

    Only reshape when an Arabic LETTER is present. Pure digit/number labels
    (Arabic-Indic digits live in the Arabic Unicode block but must stay
    left-to-right) are returned verbatim — otherwise bidi reverses multi-digit
    numbers (١٤ -> ٤١)."""
    if text is None:
        return ""
    s = str(text)
    if _ARABIC_LETTER.search(s):
        return get_display(arabic_reshaper.reshape(s))
    return s


def num(label):
    """Parse a label to float if plainly numeric (Western or Arabic-Indic)."""
    if label is None:
        return None
    s = str(label).strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    try:
        return float(s)
    except ValueError:
        return None


# ---- vector helpers ----
def unit(dx, dy):
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# ---- low-level drawers (all take ax + coordinate tuples) ----
def stroke_polygon(ax, pts):
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.plot(xs, ys, color=GEO_COLOR, lw=GEO_LW, solid_joinstyle="miter")


def stroke_segment(ax, p1, p2, lw=None, ls="-"):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GEO_COLOR,
            lw=lw or GEO_LW, linestyle=ls)


def stroke_circle(ax, center, radius, lw=None):
    ax.add_patch(mpatches.Circle(center, radius, fill=False,
                                 edgecolor=GEO_COLOR, lw=lw or GEO_LW))


def dot_label(ax, p, name, outward=(0.0, 0.0), gap=0.13, span=1.0):
    """Draw a small vertex dot and its label, offset along `outward`."""
    ax.plot([p[0]], [p[1]], "o", color=GEO_COLOR, ms=3)
    if name:
        ux, uy = unit(*outward) if (outward[0] or outward[1]) else (0, 1)
        ax.text(p[0] + ux * gap * span, p[1] + uy * gap * span, ar(str(name)),
                ha="center", va="center", fontsize=GEO_FONT, color=GEO_COLOR)


def vertex_label(ax, p, name, cen, span, gap=0.12):
    """Label a polygon vertex, pushed outward from the centroid."""
    if not name:
        return
    ux, uy = unit(p[0] - cen[0], p[1] - cen[1])
    ax.text(p[0] + ux * gap * span, p[1] + uy * gap * span, ar(str(name)),
            ha="center", va="center", fontsize=GEO_FONT, color=GEO_COLOR)


def _pt_seg_dist(px, py, a, b):
    """Distance from point (px,py) to segment a-b."""
    ax_, ay_ = a
    bx_, by_ = b
    dx, dy = bx_ - ax_, by_ - ay_
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax_, py - ay_)
    t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay_) * dy) / L2))
    cx, cy = ax_ + t * dx, ay_ + t * dy
    return math.hypot(px - cx, py - cy)


def _seg_push_dir(px, py, a, b):
    """Unit vector from the nearest point of segment a-b toward (px,py)
    (i.e. the direction that moves the point away from the segment)."""
    ax_, ay_ = a
    bx_, by_ = b
    dx, dy = bx_ - ax_, by_ - ay_
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return unit(px - ax_, py - ay_)
    t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay_) * dy) / L2))
    cx, cy = ax_ + t * dx, ay_ + t * dy
    vx, vy = px - cx, py - cy
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return (0.0, 0.0)
    return unit(vx, vy)


def vertex_labels(ax, items, cen, span, gap=0.13, segments=None, circles=None):
    """Place several vertex labels at once, nudging apart any that collide AND
    pushing them off any drawn segment they land on.

    `items` is a list of (xy, name). Each label starts pushed radially outward
    from the centroid; pairs closer than a minimum separation are spread further
    out, and labels sitting on a line/edge are pushed further along their outward
    ray until clear (fixes vertex letters interlocking with figure lines).

    `circles` is an optional list of (center_xy, r). A vertex sitting ON a circle
    gets its label pushed radially OUTSIDE that circle, so labels never land on
    the arc (a radius endpoint would otherwise be pushed tangentially along it).
    """
    segments = segments or []
    circles = circles or []

    def _centroid_dir(px, py):
        ux, uy = unit(px - cen[0], py - cen[1])
        return (ux, uy) if (ux or uy) else (0.0, 1.0)

    def _on_circle(px, py):
        """Return (center, r) of a circle this point lies on, else None."""
        for cc, r in circles:
            if abs(math.hypot(px - cc[0], py - cc[1]) - r) < 0.06 * max(r, 1.0):
                return cc, r
        return None

    def _outward_dir(px, py):
        """Pick where to put a vertex label:
        - point ON a circle -> radially outward from the circle center (beyond arc);
        - polygon vertex (edges span an angle) -> into the angular gap, away from edges;
        - point ON a line (edges collinear, e.g. a transversal or an extended base)
          -> PERPENDICULAR to the line, on the side away from the centroid, so the
          label never sits on top of the line;
        - isolated point -> radially outward from the centroid.
        """
        oc = _on_circle(px, py)
        if oc is not None:
            cc, _r = oc
            return unit(px - cc[0], py - cc[1]) or _centroid_dir(px, py)
        dirs = []
        for a, b in segments:
            for end, other in ((a, b), (b, a)):
                if math.hypot(px - end[0], py - end[1]) < 1e-6:
                    dirs.append(unit(other[0] - px, other[1] - py))
        if not dirs:
            return _centroid_dir(px, py)
        ax_, ay_ = dirs[0]
        # all incident edges parallel to the first (cross product ~ 0) => on a line
        collinear = all(abs(dx * ay_ - dy * ax_) < 0.26 for dx, dy in dirs)
        if collinear:
            nx, ny = -ay_, ax_                       # perpendicular to the line
            cx, cy = _centroid_dir(px, py)
            if nx * cx + ny * cy < 0:                 # face away from the centroid
                nx, ny = -nx, -ny
            return unit(nx, ny)
        sx = sum(d[0] for d in dirs)
        sy = sum(d[1] for d in dirs)
        if math.hypot(sx, sy) > 1e-6:
            return unit(-sx, -sy)                     # into the angular gap
        return _centroid_dir(px, py)

    placed = []  # [x, y, ux, uy, px, py, name]
    for p, name in items:
        if not name:
            continue
        ux, uy = _outward_dir(p[0], p[1])
        placed.append([p[0] + ux * gap * span, p[1] + uy * gap * span,
                       ux, uy, p[0], p[1], str(name)])

    min_sep = 0.10 * span
    for _ in range(12):  # relaxation: spread overlapping labels
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                d = math.hypot(placed[i][0] - placed[j][0], placed[i][1] - placed[j][1])
                if d < min_sep:
                    step = 0.04 * span
                    placed[i][0] += placed[i][2] * step
                    placed[i][1] += placed[i][3] * step
                    placed[j][0] += placed[j][2] * step
                    placed[j][1] += placed[j][3] * step
                    moved = True
        if not moved:
            break

    # Push each label off any segment it sits on, moving along that segment's
    # normal (away from the line). Works even for the label's own edges — fixes
    # the 3D-corner case where a cube/box vertex label lands on a depth edge.
    clear = 0.07 * span
    for lab in placed:
        for _ in range(10):
            x, y = lab[0], lab[1]
            worst = None
            worst_d = clear
            for a, b in segments:
                d = _pt_seg_dist(x, y, a, b)
                if d < worst_d:
                    worst_d = d
                    worst = (a, b)
            if worst is None:
                break
            a, b = worst
            nx, ny = _seg_push_dir(x, y, a, b)
            if nx == 0 and ny == 0:                 # label exactly on the line
                nx, ny = lab[2], lab[3]             # fall back to outward ray
            lab[0] += nx * 0.05 * span
            lab[1] += ny * 0.05 * span

    for x, y, ux, uy, px, py, name in placed:
        ax.text(x, y, ar(name), ha="center", va="center",
                fontsize=GEO_FONT, color=GEO_COLOR)


def side_label(ax, p1, p2, label, cen, gap=0.14):
    if not label:
        return
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    span = math.hypot(dx, dy) or 1.0
    ux, uy = dx / span, dy / span
    # Offset PERPENDICULAR to the segment (not along it), on the side away from the
    # centroid. The old outward-from-centroid offset put a radius/edge label ON the
    # line (a radius points away from the centre, so it slid along itself).
    nx, ny = -uy, ux
    cx, cy = mx - cen[0], my - cen[1]
    if nx * cx + ny * cy < 0:                 # face away from the figure's centre
        nx, ny = -nx, -ny
    ax.text(mx + nx * gap * span, my + ny * gap * span, ar(str(label)),
            ha="center", va="center", fontsize=GEO_FONT - 1, color=GEO_COLOR)


def angle_arc(ax, vertex, p_prev, p_next, value, radius=0.5):
    a1 = math.degrees(math.atan2(p_prev[1] - vertex[1], p_prev[0] - vertex[0]))
    a2 = math.degrees(math.atan2(p_next[1] - vertex[1], p_next[0] - vertex[0]))
    start, end = sorted([a1, a2])
    if end - start > 180:
        start, end = end, start + 360
    ax.add_patch(mpatches.Arc(vertex, 2 * radius, 2 * radius, angle=0,
                 theta1=start, theta2=end, color=GEO_COLOR, lw=GEO_ARC_LW))
    if value is not None and str(value) != "":
        mid = math.radians((start + end) / 2)
        r = radius * 1.55
        txt = f"{value}°" if str(value).strip().isdigit() else str(value)
        ax.text(vertex[0] + r * math.cos(mid), vertex[1] + r * math.sin(mid),
                ar(txt), ha="center", va="center",
                fontsize=GEO_FONT - 2, color=GEO_COLOR)


def right_angle_marker(ax, vertex, p_prev, p_next, size=0.3):
    u1 = unit(p_prev[0] - vertex[0], p_prev[1] - vertex[1])
    u2 = unit(p_next[0] - vertex[0], p_next[1] - vertex[1])
    pa = (vertex[0] + u1[0] * size, vertex[1] + u1[1] * size)
    pc = (vertex[0] + u2[0] * size, vertex[1] + u2[1] * size)
    pb = (vertex[0] + (u1[0] + u2[0]) * size, vertex[1] + (u1[1] + u2[1]) * size)
    ax.add_patch(mpatches.Polygon([vertex, pa, pb, pc], closed=True,
                 facecolor=GEO_COLOR, edgecolor=GEO_COLOR))


def new_axes(figsize=(6, 6), dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def finalize(fig, ax, out_path, margin=0.22):
    ax.margins(margin)
    ax.relim()
    ax.autoscale_view()
    # autoscale only sees data artists (lines/patches), NOT text — a vertex label
    # pushed outward (e.g. straight up from a point at the circle's top) can sit
    # beyond the limits and get clipped at the image edge. Expand the limits to
    # include every label position, then pad, so no label is ever cut off.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for t in ax.texts:
        tx, ty = t.get_position()
        x0, x1 = min(x0, tx), max(x1, tx)
        y0, y1 = min(y0, ty), max(y1, ty)
    pad = 0.08 * max(x1 - x0, y1 - y0, 1e-6)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    fig.tight_layout()
    # pad_inches keeps outward labels from being clipped at the image edge.
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return Path(out_path).exists()
