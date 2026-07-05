#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-capable Qudrat question generator (separate version).

Generates questions via the existing app6_final.py pipeline, then runs an
extra IMAGE stage: for each question a Gemini call decides whether a figure is
needed and how to draw it. Charts/bars/tables are rendered precisely with
matplotlib from a structured spec; geometry / science / free figures are drawn
by a Gemini image-generation model.

Output goes to questins_visual/<category>.xlsx (same schema as questins/*.xlsx
plus image columns) with PNGs under questins_visual/images/<category>/.

The existing pipeline (app6_final.py, questins/, main.py) is untouched.

Usage:
  python generate_visual.py --section Quantitative \
      --category "Data Interpretation & Reasoning" \
      --subcat "Tables/Charts/Graphs" --lang Arabic --num 5
"""
import os
import io
import re
import json
import math
import uuid
import argparse
from pathlib import Path

from dotenv import load_dotenv

# --- env: app6_final requires GOOGLE_API_KEY at import time; .env holds GEMINI_API_KEY ---
load_dotenv()
# Prefer the .env GEMINI_API_KEY over any stale GOOGLE_API_KEY in the shell env.
_k = os.getenv("GEMINI_API_KEY")
if _k:
    os.environ["GOOGLE_API_KEY"] = _k

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display

# Reuse the existing generation pipeline.
import app6_final as gen
from google import genai

# Paid API key — raise app6's conservative 3-concurrent-call cap so generation
# (oversample + multi-model validation) runs much faster.
import threading as _threading
gen._MAX_CONCURRENT_API_CALLS = 12
gen._api_semaphore = _threading.Semaphore(gen._MAX_CONCURRENT_API_CALLS)

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "questins_visual"
IMG_DIR = OUT_DIR / "images"
IMAGE_MODEL = "gemini-2.5-flash-image"   # Gemini image-gen for non-chart figures
CLASSIFIER_MODEL = "gemini-2.5-flash"    # decides needs_image + spec

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


_AR_LETTER = re.compile(r"[ء-يٱ-ۓ]")


def _ar(text: str) -> str:
    """Reshape + bidi so Arabic renders correctly in matplotlib.

    Only reshape when an Arabic LETTER is present. Digit-only labels (Arabic-Indic
    digits live in the Arabic block but must stay left-to-right) are returned
    verbatim — otherwise bidi reverses multi-digit numbers (١٢ -> ٢١)."""
    if text is None:
        return ""
    s = str(text)
    if _AR_LETTER.search(s):
        return get_display(arabic_reshaper.reshape(s))
    return s


# ============================================================
# IMAGE COLUMNS
# ============================================================
IMG_COLUMNS = ["NeedsImage", "RenderMethod", "ImagePath", "ImageSpec"]


# ============================================================
# 1. CLASSIFIER — does this question need a figure, and how to draw it
# ============================================================
CLASSIFIER_SYSTEM = (
    "You analyze a single test question and decide whether it REQUIRES an "
    "accompanying figure to be solvable or to match its wording. Many questions "
    "reference a figure explicitly (e.g. 'in the figure', 'from the chart', "
    "Arabic 'في الشكل', 'الرسم البياني', 'الجدول', 'المخطط الدائري'). "
    "Return clean JSON ONLY, no markdown."
)


def _classifier_prompt(question: str, answer: str, category: str, subcat: str, section: str = "") -> str:
    return f"""
Question (verbatim): {question}
Correct answer: {answer}
Section: {section}
Category: {category}
Sub-Category: {subcat}

Decide if a figure is needed. Output JSON with this exact shape:
{{
  "needs_image": true|false,
  "render": "matplotlib" | "geometry" | "gemini",
  "reason": "short",
  "chart": {{
     "type": "pie" | "bar" | "line" | "table",
     "title": "",
     "labels": ["..."],
     "values": [num, ...],
     "columns": ["..."],
     "rows": [["..."], ...]
  }},
  "geometry": {{
     "shapes": [
        {{
          "kind": "triangle" | "rectangle" | "square" | "circle" | "regular_polygon" | "quadrilateral" | "parallel_lines" | "composite" | "solid",
          "id": "t1",
          "classification": "right" | "acute" | "obtuse",
          "vertices": [{{"name": "أ", "angle": 90, "angle_label": ""}}, {{"name": "ب", "angle": 30}}, {{"name": "ج"}}],
          "sides": [{{"from": "ب", "to": "ج", "label": "10"}}],
          "right_angle_at": "أ",
          "show_angle_arcs": ["ب"],
          "width_label": "", "height_label": "", "side_label": "",
          "n": 5, "names": ["..."],
          "center_name": "م", "radius_label": "", "diameter_label": "",
          "subtype": "parallelogram|trapezoid|rhombus  (for quadrilateral) OR cube|box|cylinder|cone|sphere (for solid)",
          "base_label": "", "top_label": "", "bottom_label": "", "angle": 60,
          "edge_label": "", "depth_label": "",
          "points": [{{"name": "أ", "at_deg": 90}}],
          "sector": {{"start_deg": 0, "end_deg": 80, "label": "٨٠°"}},
          "tangent_at": "أ",
          "transversal_angle": 60,
          "angles": [{{"line": "top|bottom", "pos": "TL|TR|BL|BR", "label": "٧٥°"}}],
          "parts": [{{"kind": "rect|circle|triangle", "x": 0, "y": 0, "w": 4, "h": 4, "r": 1, "points": [[0,0],[1,0],[0,1]], "fill": "shade|white|hatch|none", "label": ""}}]
        }}
     ],
     "segments": [{{"from": "أ", "to": "ج", "label": ""}}],
     "relation": {{"inner": "t1", "outer": "c1", "mode": "inscribed"}}
  }},
  "image_prompt": ""
}}

Rules:
- needs_image=false for pure text questions (synonyms, word problems with no
  figure reference, etc). When false, render="", chart can be omitted/empty.
- DATA VISUALS: if the Sub-Category is "Tables/Charts/Graphs", OR the question
  mentions a table/chart/graph (جدول، رسم بياني، مخطط، أعمدة، دائري) OR presents
  multi-row / multi-category numeric data (e.g. production per factory per year,
  sales per day), set needs_image=true and render="matplotlib". Pick the right
  type: a comparison across categories/time → "bar" or "line"; parts of a whole
  / percentages → "pie"; rows of values per entity → "table". Extract the EXACT
  numbers and labels from the question, even if it is solvable from the text.
- GEOMETRY (2D shapes — triangle, square, rectangle, circle, regular polygon):
  if the question is about such a shape/figure OR Section is "Geometry", set
  needs_image=true and render="geometry", and fill the "geometry" block. This is
  drawn by a precise deterministic renderer, NOT an image model.
  * Fill "shapes" with one entry per shape. For triangles give all 3 "vertices"
    with their "name" and any given "angle"; set "right_angle_at" to the vertex
    name of a right angle; set "classification" (right/acute/obtuse). Put side
    lengths in "sides" as {{from,to,label}} using the EXACT given value or
    algebraic expression (e.g. "10", "س", "2س", "٥"). List vertices whose angle
    value should be shown in "show_angle_arcs".
  * For square/rectangle use "vertices" (4 names) + "side_label" or
    "width_label"/"height_label". For circle use "center_name", and exactly one
    of "radius_label"/"diameter_label". For regular_polygon set "n" + "names".
  * INTERNAL LINES — if the question references a diagonal, chord, radius line,
    height, or any segment between two named points, add it to "segments" as
    {{from,to,label}} (e.g. a square's diagonal أ→ج, a circle radius م→أ). Both
    endpoints must be names that exist (vertices or a circle's center_name).
  * NESTED figures (e.g. triangle inscribed in a circle, circle inside a square):
    give BOTH shapes their own "id" and set "relation":{{inner,outer,mode:"inscribed"}}.
  * LABELS: keep the question's OWN script. For Arabic questions use Arabic
    letters (أ ب ج د for vertices, م for a center) and Arabic-Indic digits (٠-٩)
    exactly as the question does; use Latin letters/Western digits only when the
    question itself is in Latin/coordinate form. The renderer handles Arabic.
  * VARIABLE / UNKNOWN ANGLES — if the question gives angles as algebraic
    expressions (س، ٢س، ٣س، x، 2x) or leaves an angle unknown, put the displayed
    text in "angle_label" (e.g. "٢س") and a numeric "angle" only as a drawing
    hint for the shape. Do NOT resolve variables to numbers in the labels.
  * RIGHT-ANGLE MARKER — set "right_angle_at" ONLY when the question explicitly
    states a right angle (قائم الزاوية / a 90° given in the text). NEVER add it
    just because you computed an angle to be 90°, as that would reveal the answer.
  * Every vertex/center "name" MUST be UNIQUE across all shapes. NEVER label or
    reveal the answer value (no side length, angle, or area that IS the answer).
  * QUADRILATERALS: kind="quadrilateral" with subtype "parallelogram"/"trapezoid"
    /"rhombus"; give base_label/top_label/bottom_label, height_label, side_label,
    and "angle" (degrees) as the question provides. 4 vertex names in order.
  * CIRCLE PARTS: on a circle add "points" (named points on the circumference at
    "at_deg"), "sector" {{start_deg,end_deg,label}} for a central angle / sector,
    and "tangent_at" a named point for a tangent line. Use "segments" to draw a
    chord between two named circumference points or a radius from center_name.
    For an inscribed/central angle, place the points and connect them via segments.
  * PARALLEL LINES & TRANSVERSAL (very common): kind="parallel_lines" with
    "transversal_angle" and "angles":[{{line:"top"/"bottom", pos:"TL/TR/BL/BR",
    label:"٧٥°" or "س"}}] for each labeled angle at the two intersection points.
  * COMPOSITE / SHADED regions: kind="composite" with "parts" — a list of
    primitives (rect/circle/triangle) at given x,y and size, drawn in order. Use
    "fill":"shade" for the shaded region and "fill":"white" for a hole punched out
    of it (e.g. big shaded square then a white circle = shaded ring/leftover).
    Use this for compound figures, L-shapes, grids, and "المنطقة المظللة" questions.
  * 3D SOLIDS: kind="solid" with subtype "cube"/"box"/"cylinder"/"cone"/"sphere";
    give edge_label (cube), width_label/height_label/depth_label (box),
    radius_label/height_label (cylinder, cone), radius_label (sphere). The renderer
    draws a clean 2D projection with hidden edges dashed.
- Use render="matplotlib" ONLY for data-driven visuals: pie chart, bar chart,
  line chart, or a data table. Extract the EXACT numbers/labels from the
  question so the chart is consistent with the correct answer. For pie/bar/line
  fill labels+values; for table fill columns+rows; leave the unused ones empty.
- Use render="gemini" ONLY for drawings the geometry/chart renderers cannot do:
  science illustrations, real-world scenes, free visual patterns, or unusual
  solids (pyramids, irregular 3D). NOTE: standard shapes — triangles, quads,
  circles + parts, parallel lines, composite/shaded regions, and the common 3D
  solids (cube, box, cylinder, cone, sphere) — now use render="geometry", NOT
  gemini. In image_prompt, write a precise, literal
  description restating every measurement, label, and relationship from the
  question so the drawing matches it exactly. Specify a clean black-and-white
  textbook/diagram style, no answer revealed, no extra text beyond labels.
  IMPORTANT for gemini figures: all labels MUST use Latin letters (A, B, C, x)
  and Western digits (0-9) ONLY — convert Arabic-Indic digits to Western and use
  Latin letters for variables. The image model cannot render Arabic text.
- ALWAYS also fill a short "image_prompt" even when render="geometry", as a
  safety-net description in case the deterministic renderer fails.
- For matplotlib charts, keep Arabic labels/Arabic-Indic digits as given.
- REALISTIC DATA (past-paper style): the figure must look like a real exam chart.
  NEVER output degenerate/placeholder data — no all-equal bars, no round
  duplicated values, no obvious filler. The question is written past-paper style
  and usually does NOT list the values — the FIGURE must supply them.
- CONSISTENCY WITH THE ANSWER (critical): build the chart/table data so that the
  given Correct answer is the UNIQUE correct answer to the question.
  * "highest/lowest/which is largest" → make exactly that category the max/min.
  * "difference between A and B" → choose values whose difference equals the answer.
  * "sum/total" or "average" → choose VARIED values that sum to the stated total
    (e.g. total 9000 over 6 months → six different realistic numbers summing to
    9000, never 1500×6).
  * "value of X" → the bar/slice/cell for X must equal that value.
  Double-check the numbers produce the given answer before returning.
- For pie charts, prefer naming the real sectors over a generic "أخرى/المتبقي"
  remainder slice unless the question genuinely leaves the rest unspecified.
"""


def _classify_call(prompt: str, max_retries: int = 5) -> str:
    """Classifier call with thinking DISABLED — gemini-2.5-flash's thinking
    tokens otherwise eat the output budget and truncate the JSON."""
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=CLASSIFIER_SYSTEM,
        temperature=0.1,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    import time
    for attempt in range(1, max_retries + 1):
        try:
            with gen._api_semaphore:
                resp = _img_client().models.generate_content(
                    model=CLASSIFIER_MODEL, contents=prompt, config=cfg
                )
            return resp.text or "{}"
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(30 * attempt)
            else:
                gen.log(f"[classify] {e}", level="warning")
                return "{}"
    return "{}"


# Latin -> Arabic vertex-letter map (Qudrat convention: أ ب ج د ...; center م).
# NOTE: distinct name from _AR_LETTER (the compiled regex above) — reusing that
# name here silently overwrote the regex, breaking _ar() on all Arabic text.
_AR_VERTEX = {
    "A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "ه", "F": "و",
    "G": "ز", "H": "ح", "K": "ك", "L": "ل", "N": "ن", "O": "م", "M": "م",
}


def _arabize_label(s):
    """Map a single Latin vertex letter to its Arabic equivalent. Leaves
    multi-char labels (س، ٢س، 10) and already-Arabic labels untouched."""
    if s is None:
        return s
    t = str(s).strip()
    if len(t) == 1 and t.upper() in _AR_VERTEX:
        return _AR_VERTEX[t.upper()]
    return s


def _arabize_geometry(geo: dict) -> dict:
    """Force Arabic vertex/center letters on an Arabic question so labels match
    the question's script (fixes the classifier emitting A/B/C/D)."""
    for sh in (geo.get("shapes") or []):
        if sh.get("vertices"):
            sh["vertices"] = [
                ({**v, "name": _arabize_label(v.get("name"))} if isinstance(v, dict)
                 else _arabize_label(v))
                for v in sh["vertices"]
            ]
        if sh.get("names"):
            sh["names"] = [_arabize_label(n) for n in sh["names"]]
        if sh.get("center_name"):
            sh["center_name"] = _arabize_label(sh["center_name"])
        if sh.get("right_angle_at"):
            sh["right_angle_at"] = _arabize_label(sh["right_angle_at"])
        if sh.get("show_angle_arcs"):
            sh["show_angle_arcs"] = [_arabize_label(n) for n in sh["show_angle_arcs"]]
        for s in (sh.get("sides") or []):
            s["from"] = _arabize_label(s.get("from"))
            s["to"] = _arabize_label(s.get("to"))
    for seg in (geo.get("segments") or []):
        seg["from"] = _arabize_label(seg.get("from"))
        seg["to"] = _arabize_label(seg.get("to"))
    return geo


def _validate_geometry(spec: dict) -> dict:
    """Sanity-check a geometry spec. On duplicate labels or malformed shapes,
    downgrade render to 'gemini' (using the image_prompt fallback) so a bad spec
    never produces a mislabeled figure."""
    geo = spec.get("geometry") or {}
    shapes = geo.get("shapes") or []
    if not shapes:
        spec["render"] = "gemini" if spec.get("image_prompt") else ""
        return spec

    names = []
    for sh in shapes:
        kind = (sh.get("kind") or "").lower()
        if kind == "triangle" and len(sh.get("vertices") or []) != 3:
            gen.log("[geometry] triangle without 3 vertices; downgrading", level="warning")
            spec["render"] = "gemini" if spec.get("image_prompt") else ""
            return spec
        for v in (sh.get("vertices") or []):
            if v.get("name"):
                names.append(str(v["name"]).strip())
        if sh.get("center_name"):
            names.append(str(sh["center_name"]).strip())

    # Garbled / composite spec guard: a real vertex label is a single letter (or
    # a short expression like ٢س). Long multi-char names mean the classifier
    # hallucinated (often on composite figures the renderer can't draw) -> gemini.
    if any(len(n) > 2 for n in names):
        gen.log(f"[geometry] suspicious labels {names}; downgrading to gemini",
                level="warning")
        spec["render"] = "gemini" if spec.get("image_prompt") else ""
        return spec

    if len(names) != len(set(names)):
        gen.log(f"[geometry] duplicate labels {names}; downgrading to gemini",
                level="warning")
        spec["render"] = "gemini" if spec.get("image_prompt") else ""
    return spec


def classify(q: dict) -> dict:
    question = gen.clean_text(q.get("Question", ""))
    answer = gen.clean_text(q.get("Answer", q.get("CorrectAnswer", "")))
    category = gen.clean_text(q.get("Category", ""))
    subcat = gen.clean_text(q.get("Sub-Category", ""))
    section = gen.clean_text(q.get("Section", ""))
    raw = _classify_call(_classifier_prompt(question, answer, category, subcat, section))
    spec = gen.safe_json_load(raw) or {}
    if not isinstance(spec, dict) or spec.get("error"):
        return {"needs_image": False, "render": "", "reason": "classifier_failed"}
    if (spec.get("render") or "").lower() == "geometry":
        # On Arabic questions, force Arabic vertex/center letters before validating.
        if re.search(r"[؀-ۿ]", question) and isinstance(spec.get("geometry"), dict):
            spec["geometry"] = _arabize_geometry(spec["geometry"])
        spec = _validate_geometry(spec)
    return spec


# ============================================================
# 2a. MATPLOTLIB RENDERER (charts / bars / tables)
# ============================================================
def render_chart(spec: dict, out_path: Path) -> bool:
    chart = spec.get("chart") or {}
    ctype = (chart.get("type") or "").lower()
    title = _ar(chart.get("title", ""))
    labels = [_ar(x) for x in (chart.get("labels") or [])]
    values = chart.get("values") or []

    # --- layout hygiene: keep labels readable and non-overlapping -------------
    def _longest(seq):
        return max((len(str(s)) for s in seq), default=0)

    try:
        n = len(values)
        if ctype == "pie" and values:
            many = n > 5 or _longest(labels) > 10
            # wide canvas when labels go to a side legend so nothing overlaps
            fig, ax = plt.subplots(figsize=(7.2 if many else 6, 4.8), dpi=150)
            mx = max(values) if values else 0
            explode = [0.06 if (mx and v / mx < 0.08) else 0 for v in values]
            wedges, _texts, autotexts = ax.pie(
                values,
                labels=None if many else (labels or None),
                autopct="%1.0f%%", startangle=90, counterclock=False,
                explode=explode, pctdistance=0.78, labeldistance=1.08,
                wedgeprops={"edgecolor": "white", "linewidth": 1},
            )
            for t in autotexts:
                t.set_fontsize(9)
            if many and labels:
                ax.legend(wedges, labels, loc="center left",
                          bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)
            ax.axis("equal")
        elif ctype == "bar" and values:
            xs = labels if labels else [str(i + 1) for i in range(n)]
            width = max(6, min(12, 0.9 * n + 2))
            fig, ax = plt.subplots(figsize=(width, 4.8), dpi=150)
            ax.bar(xs, values, color="#4C72B0", width=0.6, edgecolor="white")
            mx = max(values)
            ax.set_ylim(0, mx * 1.18 if mx > 0 else 1)  # headroom for value labels
            for i, v in enumerate(values):
                ax.text(i, v + mx * 0.015, str(v), ha="center", va="bottom",
                        fontsize=10)
            # rotate long x labels so they never overlap each other
            if _longest(xs) > 6 or n > 6:
                ax.set_xticks(range(n))
                ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=10)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif ctype == "line" and values:
            xs = labels if labels else list(range(1, n + 1))
            width = max(6, min(12, 0.8 * n + 2))
            fig, ax = plt.subplots(figsize=(width, 4.8), dpi=150)
            ax.plot(xs, values, marker="o", color="#4C72B0")
            lo, hi = min(values), max(values)
            pad = (hi - lo) * 0.18 or (abs(hi) * 0.18 or 1)
            ax.set_ylim(lo - pad * 0.4, hi + pad)  # clearance for annotations
            for x, v in zip(xs, values):
                ax.annotate(str(v), (x, v), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=9)
            if _longest([str(x) for x in xs]) > 6 or n > 8:
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=10)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif ctype == "table":
            cols = [_ar(c) for c in (chart.get("columns") or [])]
            rows = [[_ar(c) for c in r] for r in (chart.get("rows") or [])]
            ncol = max((len(r) for r in rows), default=len(cols))
            fig, ax = plt.subplots(
                figsize=(max(5, 1.7 * max(ncol, 1)), max(2.2, 0.6 * (len(rows) + 1))),
                dpi=150)
            ax.axis("off")
            if rows:
                tbl = ax.table(
                    cellText=rows,
                    colLabels=cols if cols else None,
                    loc="center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(11)
                tbl.scale(1, 1.6)
                # size each column to its longest cell so columns never collide
                try:
                    tbl.auto_set_column_width(col=list(range(ncol)))
                except Exception:
                    pass
        else:
            return False

        if title:
            ax.set_title(title, fontsize=13, pad=10)
        fig.tight_layout(pad=1.2)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path.exists()
    except Exception as e:
        gen.log(f"[chart] render failed: {e}", level="warning")
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ============================================================
# 2c. DETERMINISTIC GEOMETRY RENDERER (triangles / quads / circles / nested)
#     Styled to match the past-exam figures embedded in the sample sheet:
#     solid black lines, white bg, filled-square right-angle marker, angle
#     arcs + values, Arabic labels (أ ب ج، م) rendered via _ar().
# ============================================================
from matplotlib import patches as _mpatches

_GEO_LW = 2.0          # main edge line weight
_GEO_ARC_LW = 1.2      # angle-arc line weight
_GEO_COLOR = "black"
_GEO_FONT = 14


def _num(label):
    """Parse a side/length label to float if it is plainly numeric (Western or
    Arabic-Indic digits). Returns None for algebraic labels like 'س' or '2س'."""
    if label is None:
        return None
    s = str(label).strip()
    # normalize Arabic-Indic digits to Western for parsing
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s2 = s.translate(trans)
    try:
        return float(s2)
    except ValueError:
        return None


def _angles_from_verts(verts, right_at, classification):
    """Return three interior angles (deg) in vertex order."""
    given = [v.get("angle") for v in verts]
    # force right angle if requested
    if right_at:
        for i, v in enumerate(verts):
            if v.get("name") == right_at:
                given[i] = 90.0
    known = [(i, float(a)) for i, a in enumerate(given) if a is not None]
    if len(known) >= 2:
        # fill the missing one so all three sum to 180
        s = sum(a for _, a in known)
        idxs = {i for i, _ in known}
        for i in range(3):
            if i not in idxs:
                given[i] = max(1.0, 180.0 - s)
        angs = [float(a) for a in given]
    elif len(known) == 1 and right_at:
        # right + one other -> third known
        i0, a0 = known[0]
        # place the remaining (non-right) angle; if only the 90 is known, default
        others = [i for i in range(3) if given[i] is None]
        if len(others) == 2:
            given[others[0]] = 60.0 if a0 == 90 else (90.0 - a0 if a0 < 90 else 45.0)
            given[others[1]] = 180.0 - given[others[0]] - a0
        angs = [float(a) for a in given]
    else:
        # under-determined: default by classification
        c = (classification or "").lower()
        if c == "right" or right_at:
            angs = [90.0, 60.0, 30.0]
        elif c == "obtuse":
            angs = [110.0, 40.0, 30.0]
        else:
            angs = [60.0, 60.0, 60.0]
        # respect any single given angle by slotting it first
        if len(known) == 1:
            i0, a0 = known[0]
            angs[i0] = a0
    # normalize to sum 180
    total = sum(angs)
    if total <= 0:
        angs = [60.0, 60.0, 60.0]
    elif abs(total - 180.0) > 1.0:
        angs = [a * 180.0 / total for a in angs]
    return angs


def _solve_triangle(verts, sides, right_at, classification):
    """Compute three (x,y) points. Right angle (if any) is exact by construction.
    Uses law of sines; scales by numeric side labels when available."""
    angs = _angles_from_verts(verts, right_at, classification)
    names = [v.get("name") for v in verts]

    if right_at and right_at in names:
        # Build perpendicular legs from the right-angle vertex -> exact 90°.
        r = names.index(right_at)
        a, b = [i for i in range(3) if i != r]
        # legs along +x and +y from the right vertex
        # leg lengths: from numeric side labels if present, else from angles
        leg_ra = _side_len(sides, names[r], names[a])
        leg_rb = _side_len(sides, names[r], names[b])
        hyp = _side_len(sides, names[a], names[b])  # side opposite the right angle
        # One leg + hypotenuse given -> other leg by Pythagoras (not angle guess).
        if hyp and leg_ra is not None and leg_rb is None and hyp > leg_ra:
            leg_rb = math.sqrt(hyp**2 - leg_ra**2)
        elif hyp and leg_rb is not None and leg_ra is None and hyp > leg_rb:
            leg_ra = math.sqrt(hyp**2 - leg_rb**2)
        if leg_ra is None or leg_rb is None:
            # derive from angles via law of sines (hyp opposite the 90°)
            leg_ra = leg_ra or math.cos(math.radians(angs[a]))
            leg_rb = leg_rb or math.sin(math.radians(angs[a]))
        pts = [None, None, None]
        pts[r] = (0.0, 0.0)
        pts[a] = (float(leg_ra), 0.0)
        pts[b] = (0.0, float(leg_rb))
        return pts, names

    # General: place vertex 0 at origin, side 0-1 along +x, vertex 2 by angle at 0.
    # side lengths via law of sines: side opposite angle i.
    sin = [math.sin(math.radians(a)) for a in angs]
    # opposite-side ratios
    a_len = sin[0] or 0.5  # opposite v0 => edge v1-v2
    b_len = sin[1] or 0.5  # opposite v1 => edge v0-v2
    c_len = sin[2] or 0.5  # opposite v2 => edge v0-v1
    # try to honor a numeric side label for scaling
    scale = 1.0
    for s in (sides or []):
        ln = _num(s.get("label"))
        if ln:
            fr, to = s.get("from"), s.get("to")
            if {fr, to} == {names[0], names[1]}:
                scale = ln / c_len if c_len else 1.0
            elif {fr, to} == {names[1], names[2]}:
                scale = ln / a_len if a_len else 1.0
            elif {fr, to} == {names[0], names[2]}:
                scale = ln / b_len if b_len else 1.0
            break
    c_len *= scale
    b_len *= scale
    p0 = (0.0, 0.0)
    p1 = (c_len, 0.0)
    ang0 = math.radians(angs[0])
    p2 = (b_len * math.cos(ang0), b_len * math.sin(ang0))
    return [p0, p1, p2], names


def _side_len(sides, n1, n2):
    for s in (sides or []):
        if {s.get("from"), s.get("to")} == {n1, n2}:
            return _num(s.get("label"))
    return None


def _vname(v):
    """A vertex may arrive as a plain string or as {'name': ...}. Normalize."""
    if isinstance(v, dict):
        return v.get("name", "")
    return v


def _centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _unit(dx, dy):
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def _draw_polygon_edges(ax, pts):
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.plot(xs, ys, color=_GEO_COLOR, lw=_GEO_LW, solid_joinstyle="miter")


def _draw_vertex_labels(ax, pts, names, gap=0.12):
    c = _centroid(pts)
    span = max(p[0] for p in pts) - min(p[0] for p in pts)
    span = max(span, max(p[1] for p in pts) - min(p[1] for p in pts), 1.0)
    for (x, y), name in zip(pts, names):
        name = _vname(name)
        if not name:
            continue
        ux, uy = _unit(x - c[0], y - c[1])
        ax.text(x + ux * gap * span, y + uy * gap * span, _ar(str(name)),
                ha="center", va="center", fontsize=_GEO_FONT, color=_GEO_COLOR)


def _draw_side_label(ax, p1, p2, label, centroid, gap=0.08):
    if not label:
        return
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    # outward normal (away from centroid)
    nx, ny = _unit(mx - centroid[0], my - centroid[1])
    span = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) or 1.0
    ax.text(mx + nx * gap * span, my + ny * gap * span, _ar(str(label)),
            ha="center", va="center", fontsize=_GEO_FONT - 1, color=_GEO_COLOR)


def _draw_angle_arc(ax, vertex, p_prev, p_next, value, radius=0.5):
    a1 = math.degrees(math.atan2(p_prev[1] - vertex[1], p_prev[0] - vertex[0]))
    a2 = math.degrees(math.atan2(p_next[1] - vertex[1], p_next[0] - vertex[0]))
    # draw the smaller arc
    start, end = sorted([a1, a2])
    if end - start > 180:
        start, end = end, start + 360
    arc = _mpatches.Arc(vertex, 2 * radius, 2 * radius, angle=0, theta1=start, theta2=end,
                        color=_GEO_COLOR, lw=_GEO_ARC_LW)
    ax.add_patch(arc)
    if value is not None and str(value) != "":
        mid = math.radians((start + end) / 2)
        # place the value just beyond the arc, snug in the corner along the bisector
        r = radius * 1.55
        ax.text(vertex[0] + r * math.cos(mid), vertex[1] + r * math.sin(mid),
                _ar(f"{value}°" if str(value).strip().isdigit() else str(value)),
                ha="center", va="center", fontsize=_GEO_FONT - 2, color=_GEO_COLOR)


def _draw_right_angle(ax, vertex, p_prev, p_next, size=0.5):
    u1 = _unit(p_prev[0] - vertex[0], p_prev[1] - vertex[1])
    u2 = _unit(p_next[0] - vertex[0], p_next[1] - vertex[1])
    s = size
    p_a = (vertex[0] + u1[0] * s, vertex[1] + u1[1] * s)
    p_c = (vertex[0] + u2[0] * s, vertex[1] + u2[1] * s)
    p_b = (vertex[0] + (u1[0] + u2[0]) * s, vertex[1] + (u1[1] + u2[1]) * s)
    sq = _mpatches.Polygon([vertex, p_a, p_b, p_c], closed=True,
                           facecolor=_GEO_COLOR, edgecolor=_GEO_COLOR)
    ax.add_patch(sq)


def _register(registry, names, pts):
    if registry is None:
        return
    for n, p in zip(names, pts):
        nm = _vname(n)
        if nm:
            registry[str(nm)] = p


def _draw_triangle(ax, sh, registry=None):
    verts = sh.get("vertices") or []
    if len(verts) != 3:
        raise ValueError("triangle needs 3 vertices")
    sides = sh.get("sides") or []
    right_at = sh.get("right_angle_at")
    pts, names = _solve_triangle(verts, sides, right_at, sh.get("classification"))
    _draw_polygon_edges(ax, pts)
    cen = _centroid(pts)
    _draw_vertex_labels(ax, pts, names)
    _register(registry, names, pts)
    # side labels
    for s in sides:
        try:
            i = names.index(s.get("from"))
            j = names.index(s.get("to"))
        except ValueError:
            continue
        _draw_side_label(ax, pts[i], pts[j], s.get("label"), cen)
    # angle markers — scale arc/box to the triangle so labels sit in the corners,
    # not clustered at the centroid. unit = shortest edge length.
    edges = [math.hypot(pts[(k + 1) % 3][0] - pts[k][0], pts[(k + 1) % 3][1] - pts[k][1])
             for k in range(3)]
    unit = min(e for e in edges if e > 0) if any(edges) else 1.0
    arc_r = 0.16 * unit
    box_s = 0.13 * unit
    arc_set = set(sh.get("show_angle_arcs") or [])
    for k, v in enumerate(verts):
        nm = v.get("name")
        prev_p = pts[(k - 1) % 3]
        next_p = pts[(k + 1) % 3]
        disp = v.get("angle_label") if v.get("angle_label") not in (None, "") else v.get("angle")
        if nm == right_at:
            _draw_right_angle(ax, pts[k], prev_p, next_p, size=box_s)
        elif disp is not None and (not arc_set or nm in arc_set):
            _draw_angle_arc(ax, pts[k], prev_p, next_p, disp, radius=arc_r)
    return pts


def _draw_rect(ax, sh, registry=None):
    names = [_vname(v) for v in (sh.get("vertices") or ["A", "B", "C", "D"])]
    w = _num(sh.get("width_label")) or _num(sh.get("side_label")) or 1.6
    h = _num(sh.get("height_label")) or _num(sh.get("side_label")) or (
        w if sh.get("kind") == "square" else 1.0)
    if sh.get("kind") == "square":
        h = w
    pts = [(0, 0), (w, 0), (w, h), (0, h)]
    _draw_polygon_edges(ax, pts)
    _draw_vertex_labels(ax, pts, names[:4])
    _register(registry, names[:4], pts)
    cen = _centroid(pts)
    if sh.get("width_label"):
        _draw_side_label(ax, pts[0], pts[1], sh.get("width_label"), cen)
    if sh.get("height_label"):
        _draw_side_label(ax, pts[1], pts[2], sh.get("height_label"), cen)
    if sh.get("side_label") and sh.get("kind") == "square":
        _draw_side_label(ax, pts[0], pts[1], sh.get("side_label"), cen)
    return pts


def _draw_circle(ax, sh, center=(0, 0), radius=1.0, registry=None):
    ax.add_patch(_mpatches.Circle(center, radius, fill=False,
                                  edgecolor=_GEO_COLOR, lw=_GEO_LW))
    cn = sh.get("center_name")
    if cn:
        _register(registry, [cn], [center])
        ax.text(center[0], center[1] - 0.06 * radius, _ar(str(cn)),
                ha="center", va="top", fontsize=_GEO_FONT, color=_GEO_COLOR)
        ax.plot([center[0]], [center[1]], "o", color=_GEO_COLOR, ms=3)
    rlab = sh.get("radius_label")
    dlab = sh.get("diameter_label")
    if rlab:
        end = (center[0] + radius, center[1])
        ax.plot([center[0], end[0]], [center[1], end[1]], color=_GEO_COLOR, lw=_GEO_ARC_LW)
        ax.text((center[0] + end[0]) / 2, center[1] + 0.06 * radius, _ar(str(rlab)),
                ha="center", va="bottom", fontsize=_GEO_FONT - 1, color=_GEO_COLOR)
    elif dlab:
        ax.plot([center[0] - radius, center[0] + radius], [center[1], center[1]],
                color=_GEO_COLOR, lw=_GEO_ARC_LW)
        ax.text(center[0], center[1] + 0.06 * radius, _ar(str(dlab)),
                ha="center", va="bottom", fontsize=_GEO_FONT - 1, color=_GEO_COLOR)

    # ---- circle parts: named points on the circumference, sector, tangent ----
    pts_on = {}
    for p in (sh.get("points") or []):
        a = math.radians(_num(p.get("at_deg")) or 0)
        coord = (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
        nm = p.get("name")
        pts_on[str(nm)] = (coord, a)
        if nm:
            _register(registry, [nm], [coord])
        ax.plot([coord[0]], [coord[1]], "o", color=_GEO_COLOR, ms=3)
        if nm:
            ax.text(coord[0] + 0.13 * radius * math.cos(a),
                    coord[1] + 0.13 * radius * math.sin(a),
                    _ar(str(nm)), ha="center", va="center",
                    fontsize=_GEO_FONT, color=_GEO_COLOR)
    sec = sh.get("sector")
    if sec:
        s1 = math.radians(_num(sec.get("start_deg")) or 0)
        s2 = math.radians(_num(sec.get("end_deg")) or 90)
        for a in (s1, s2):
            ax.plot([center[0], center[0] + radius * math.cos(a)],
                    [center[1], center[1] + radius * math.sin(a)],
                    color=_GEO_COLOR, lw=_GEO_LW)
        # small arc marking the central angle
        ax.add_patch(_mpatches.Arc(center, radius * 0.5, radius * 0.5,
                     theta1=math.degrees(min(s1, s2)), theta2=math.degrees(max(s1, s2)),
                     color=_GEO_COLOR, lw=_GEO_ARC_LW))
        if sec.get("label"):
            mid = (s1 + s2) / 2
            rr = radius * 0.34
            ax.text(center[0] + rr * math.cos(mid), center[1] + rr * math.sin(mid),
                    _ar(str(sec["label"])), ha="center", va="center",
                    fontsize=_GEO_FONT - 1, color=_GEO_COLOR)
    tan = sh.get("tangent_at")
    if tan and str(tan) in pts_on:
        (px, py), a = pts_on[str(tan)]
        tx, ty = -math.sin(a), math.cos(a)
        L = radius * 1.15
        ax.plot([px - tx * L, px + tx * L], [py - ty * L, py + ty * L],
                color=_GEO_COLOR, lw=_GEO_LW)
    return [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)]


def _draw_regular_polygon(ax, sh, center=(0, 0), radius=1.0, registry=None):
    n = int(sh.get("n") or 5)
    names = [_vname(v) for v in (sh.get("names") or [])]
    pts = []
    for k in range(n):
        a = math.radians(90 + k * 360.0 / n)
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    _draw_polygon_edges(ax, pts)
    if names:
        _draw_vertex_labels(ax, pts, names[:n])
        _register(registry, names[:n], pts)
    if sh.get("side_label"):
        _draw_side_label(ax, pts[0], pts[1], sh.get("side_label"), _centroid(pts))
    return pts


def _draw_segments(ax, segments, registry):
    """Draw internal lines (diagonals, chords, radii) between named points."""
    for seg in segments or []:
        p1 = registry.get(str(seg.get("from")))
        p2 = registry.get(str(seg.get("to")))
        if not p1 or not p2:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=_GEO_COLOR, lw=_GEO_ARC_LW)
        lab = seg.get("label")
        if lab:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx, my, _ar(str(lab)), ha="center", va="bottom",
                    fontsize=_GEO_FONT - 1, color=_GEO_COLOR,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))


def _circumcircle(pts):
    """Circumcenter + radius of a triangle (3 points)."""
    (ax_, ay), (bx, by), (cx, cy) = pts[0], pts[1], pts[2]
    d = 2 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        c = _centroid(pts)
        r = max(math.hypot(p[0] - c[0], p[1] - c[1]) for p in pts)
        return c, r
    ux = ((ax_**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax_**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax_ - cx) +
          (cx**2 + cy**2) * (bx - ax_)) / d
    r = math.hypot(ux - ax_, uy - ay)
    return (ux, uy), r


def _circumcircle_pts(pts):
    """Circumcircle of any polygon point set. Exact for triangles; for squares/
    regular polygons centroid + max-vertex-distance is the true circumcircle."""
    if len(pts) == 3:
        return _circumcircle(pts)
    c = _centroid(pts)
    r = max(math.hypot(p[0] - c[0], p[1] - c[1]) for p in pts)
    return c, r


# ---- Quadrilaterals (parallelogram / trapezoid / rhombus / generic) ----
def _quad_points(sh):
    sub = (sh.get("subtype") or "").lower()
    ang = _num(sh.get("angle")) or 65.0
    if sub == "parallelogram":
        w = _num(sh.get("base_label")) or 2.2
        h = _num(sh.get("height_label")) or _num(sh.get("side_label")) or 1.3
        off = h / math.tan(math.radians(ang)) if ang % 180 else 0.7
        return [(0, 0), (w, 0), (w + off, h), (off, h)]
    if sub == "trapezoid":
        w = _num(sh.get("bottom_label")) or _num(sh.get("base_label")) or 2.6
        top = _num(sh.get("top_label")) or w * 0.55
        h = _num(sh.get("height_label")) or 1.3
        off = (w - top) / 2.0
        return [(0, 0), (w, 0), (w - off, h), (off, h)]
    if sub == "rhombus":
        s = _num(sh.get("side_label")) or 1.7
        off = s * math.cos(math.radians(ang))
        hh = s * math.sin(math.radians(ang))
        return [(0, 0), (s, 0), (s + off, hh), (off, hh)]
    # generic quadrilateral -> mild trapezoid so it doesn't look like a rectangle
    return [(0, 0), (2.4, 0), (2.0, 1.4), (0.3, 1.4)]


def _draw_quad(ax, sh, registry=None):
    pts = _quad_points(sh)
    names = [_vname(v) for v in (sh.get("vertices") or ["A", "B", "C", "D"])]
    _draw_polygon_edges(ax, pts)
    _draw_vertex_labels(ax, pts, names[:4])
    _register(registry, names[:4], pts)
    cen = _centroid(pts)
    for s in (sh.get("sides") or []):
        try:
            i = names.index(s.get("from"))
            j = names.index(s.get("to"))
        except ValueError:
            continue
        _draw_side_label(ax, pts[i], pts[j], s.get("label"), cen)
    if sh.get("base_label"):
        _draw_side_label(ax, pts[0], pts[1], sh.get("base_label"), cen)
    if sh.get("bottom_label"):
        _draw_side_label(ax, pts[0], pts[1], sh.get("bottom_label"), cen)
    if sh.get("top_label"):
        _draw_side_label(ax, pts[3], pts[2], sh.get("top_label"), cen)
    if sh.get("side_label") and (sh.get("subtype") or "").lower() == "rhombus":
        _draw_side_label(ax, pts[1], pts[2], sh.get("side_label"), cen)
    # height as a dashed perpendicular from top-left down to the base
    if sh.get("height_label"):
        foot = (pts[3][0], pts[0][1])
        ax.plot([pts[3][0], foot[0]], [pts[3][1], foot[1]],
                color=_GEO_COLOR, lw=_GEO_ARC_LW, ls="--")
        mx, my = pts[3][0], (pts[3][1] + foot[1]) / 2
        ax.text(mx - 0.12, my, _ar(str(sh.get("height_label"))),
                ha="right", va="center", fontsize=_GEO_FONT - 1, color=_GEO_COLOR)
    return pts


# ---- Transversal cutting two parallel lines ----
def _draw_parallel_lines(ax, sh, registry=None):
    ang = _num(sh.get("transversal_angle")) or 60.0
    if ang % 180 == 0:
        ang = 60.0
    slope = math.tan(math.radians(ang))
    y0, y1 = 0.0, 1.4
    xl, xr = -1.8, 1.8
    ax.plot([xl, xr], [y0, y0], color=_GEO_COLOR, lw=_GEO_LW)
    ax.plot([xl, xr], [y1, y1], color=_GEO_COLOR, lw=_GEO_LW)
    # parallel arrow ticks
    for y in (y0, y1):
        ax.annotate("", xy=(xr - 0.05, y), xytext=(xr - 0.35, y),
                    arrowprops=dict(arrowstyle="->", color=_GEO_COLOR, lw=1))
    yc, xc = (y0 + y1) / 2, 0.0
    ytop, ybot = y1 + 0.7, y0 - 0.7
    xtop = xc + (ytop - yc) / slope
    xbot = xc + (ybot - yc) / slope
    ax.plot([xbot, xtop], [ybot, ytop], color=_GEO_COLOR, lw=_GEO_LW)
    ix_top = (xc + (y1 - yc) / slope, y1)
    ix_bot = (xc + (y0 - yc) / slope, y0)
    for an in (sh.get("angles") or []):
        base = ix_top if (an.get("line") == "top") else ix_bot
        quad = (an.get("pos") or "TR").upper()
        dx = 0.30 * (1 if "R" in quad else -1)
        dy = 0.22 * (1 if "T" in quad else -1)
        ax.text(base[0] + dx, base[1] + dy, _ar(str(an.get("label", ""))),
                ha="center", va="center", fontsize=_GEO_FONT - 2, color=_GEO_COLOR)
    return [(xl, ybot), (xr, ytop)]


# ---- Composite / shaded compound figures ----
def _draw_composite(ax, sh, registry=None):
    fcmap = {"shade": "#c4c4c4", "white": "white", "hatch": "none"}
    for p in (sh.get("parts") or []):
        k = (p.get("kind") or "").lower()
        fill = (p.get("fill") or "").lower()
        fc = fcmap.get(fill, "none")
        hatch = "///" if fill == "hatch" else None
        if k in ("rect", "rectangle", "square"):
            x, y = _num(p.get("x")) or 0, _num(p.get("y")) or 0
            w = _num(p.get("w")) or _num(p.get("size")) or 1.0
            h = _num(p.get("h")) or (w if k == "square" else 1.0)
            ax.add_patch(_mpatches.Rectangle((x, y), w, h, facecolor=fc, hatch=hatch,
                         fill=(fc != "none"), edgecolor=_GEO_COLOR, lw=_GEO_LW))
            if p.get("label"):
                ax.text(x + w / 2, y + h / 2, _ar(str(p["label"])), ha="center",
                        va="center", fontsize=_GEO_FONT - 1, color=_GEO_COLOR)
        elif k == "circle":
            cx, cy = _num(p.get("x")) or 0, _num(p.get("y")) or 0
            r = _num(p.get("r")) or 1.0
            ax.add_patch(_mpatches.Circle((cx, cy), r, facecolor=fc, hatch=hatch,
                         fill=(fc != "none"), edgecolor=_GEO_COLOR, lw=_GEO_LW))
        elif k == "triangle":
            tp = [(_num(a[0]) or 0, _num(a[1]) or 0) for a in (p.get("points") or [])]
            if len(tp) == 3:
                ax.add_patch(_mpatches.Polygon(tp, closed=True, facecolor=fc, hatch=hatch,
                             fill=(fc != "none"), edgecolor=_GEO_COLOR, lw=_GEO_LW))
    return None


# ---- 3D solids drawn as clean 2D projections ----
def _draw_solid(ax, sh, registry=None):
    sub = (sh.get("subtype") or "cube").lower()
    edge = _num(sh.get("edge_label"))
    SOLID = _GEO_COLOR

    def lbl(x, y, t, ha="center", va="center", dx=0, dy=0):
        if t:
            ax.text(x + dx, y + dy, _ar(str(t)), ha=ha, va=va,
                    fontsize=_GEO_FONT - 1, color=SOLID)

    if sub in ("cube", "box", "cuboid", "rectangular_prism", "prism"):
        w = _num(sh.get("width_label")) or edge or 1.6
        h = _num(sh.get("height_label")) or edge or 1.6
        if sub == "cube":
            h = w
        dep = 0.42 * min(w, h)
        f = [(0, 0), (w, 0), (w, h), (0, h)]
        b = [(x + dep, y + dep) for (x, y) in f]
        _draw_polygon_edges(ax, f)                       # front face (solid)
        # top + right visible faces
        ax.plot([f[3][0], b[3][0]], [f[3][1], b[3][1]], color=SOLID, lw=_GEO_LW)
        ax.plot([f[2][0], b[2][0]], [f[2][1], b[2][1]], color=SOLID, lw=_GEO_LW)
        ax.plot([f[1][0], b[1][0]], [f[1][1], b[1][1]], color=SOLID, lw=_GEO_LW)
        ax.plot([b[3][0], b[2][0]], [b[3][1], b[2][1]], color=SOLID, lw=_GEO_LW)
        ax.plot([b[2][0], b[1][0]], [b[2][1], b[1][1]], color=SOLID, lw=_GEO_LW)
        # hidden back edges dashed
        ax.plot([f[0][0], b[0][0]], [f[0][1], b[0][1]], color=SOLID, lw=_GEO_ARC_LW, ls="--")
        ax.plot([b[0][0], b[3][0]], [b[0][1], b[3][1]], color=SOLID, lw=_GEO_ARC_LW, ls="--")
        ax.plot([b[0][0], b[1][0]], [b[0][1], b[1][1]], color=SOLID, lw=_GEO_ARC_LW, ls="--")
        lbl(w / 2, -0.12 * h, sh.get("width_label") or (sh.get("edge_label") if sub == "cube" else None), va="top")
        lbl(w + 0.12, h / 2, sh.get("height_label"), ha="left")
        lbl(w + dep / 2, dep / 2 - 0.05, sh.get("depth_label"), va="top")
        return f
    if sub == "cylinder":
        r = _num(sh.get("radius_label")) or 1.0
        h = _num(sh.get("height_label")) or 2.0
        ry = 0.28 * r
        ax.add_patch(_mpatches.Ellipse((0, h), 2 * r, 2 * ry, fill=False,
                     edgecolor=SOLID, lw=_GEO_LW))
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 2 * ry, theta1=180, theta2=360,
                     color=SOLID, lw=_GEO_LW))
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 2 * ry, theta1=0, theta2=180,
                     color=SOLID, lw=_GEO_ARC_LW, linestyle="--"))
        ax.plot([-r, -r], [0, h], color=SOLID, lw=_GEO_LW)
        ax.plot([r, r], [0, h], color=SOLID, lw=_GEO_LW)
        if sh.get("radius_label"):
            ax.plot([0, r], [h, h], color=SOLID, lw=_GEO_ARC_LW)
            lbl(r / 2, h + 0.08, sh.get("radius_label"), va="bottom")
        lbl(r + 0.15, h / 2, sh.get("height_label"), ha="left")
        return None
    if sub == "cone":
        r = _num(sh.get("radius_label")) or 1.0
        h = _num(sh.get("height_label")) or 2.0
        ry = 0.28 * r
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 2 * ry, theta1=180, theta2=360,
                     color=SOLID, lw=_GEO_LW))
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 2 * ry, theta1=0, theta2=180,
                     color=SOLID, lw=_GEO_ARC_LW, linestyle="--"))
        ax.plot([-r, 0], [0, h], color=SOLID, lw=_GEO_LW)
        ax.plot([r, 0], [0, h], color=SOLID, lw=_GEO_LW)
        lbl(r / 2, -0.1, sh.get("radius_label"), va="top")
        lbl(0.12, h / 2, sh.get("height_label"), ha="left")
        return None
    if sub == "sphere":
        r = _num(sh.get("radius_label")) or 1.0
        ax.add_patch(_mpatches.Circle((0, 0), r, fill=False, edgecolor=SOLID, lw=_GEO_LW))
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 0.7 * r, theta1=180, theta2=360,
                     color=SOLID, lw=_GEO_LW))
        ax.add_patch(_mpatches.Arc((0, 0), 2 * r, 0.7 * r, theta1=0, theta2=180,
                     color=SOLID, lw=_GEO_ARC_LW, linestyle="--"))
        if sh.get("radius_label"):
            ax.plot([0, r * 0.92], [0, r * 0.38], color=SOLID, lw=_GEO_ARC_LW)
            lbl(r * 0.5, r * 0.28, sh.get("radius_label"), va="bottom")
        return None
    raise ValueError(f"unknown solid subtype: {sub}")


def _draw_shape(ax, sh, center=(0, 0), radius=1.0, registry=None):
    kind = (sh.get("kind") or "").lower()
    if kind == "triangle":
        return _draw_triangle(ax, sh, registry=registry)
    if kind in ("rectangle", "square"):
        return _draw_rect(ax, sh, registry=registry)
    if kind == "circle":
        return _draw_circle(ax, sh, center, radius, registry=registry)
    if kind == "regular_polygon":
        return _draw_regular_polygon(ax, sh, center, radius, registry=registry)
    if kind in ("quadrilateral", "parallelogram", "trapezoid", "rhombus"):
        if kind != "quadrilateral":
            sh = {**sh, "subtype": kind, "kind": "quadrilateral"}
        return _draw_quad(ax, sh, registry=registry)
    if kind in ("parallel_lines", "transversal"):
        return _draw_parallel_lines(ax, sh, registry=registry)
    if kind == "composite":
        return _draw_composite(ax, sh, registry=registry)
    if kind == "solid":
        return _draw_solid(ax, sh, registry=registry)
    raise ValueError(f"unknown shape kind: {kind}")


def render_geometry(spec, out_path):
    geo = spec.get("geometry") or {}
    shapes = geo.get("shapes") or []
    if not shapes:
        return False
    relation = geo.get("relation") or {}
    try:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.set_aspect("equal")
        ax.axis("off")

        by_id = {sh.get("id"): sh for sh in shapes if sh.get("id")}
        drawn = set()
        registry = {}

        # One level of nesting: inner inscribed in outer.
        inner = relation.get("inner")
        outer = relation.get("outer")
        if inner and outer and inner in by_id and outer in by_id:
            inner_sh, outer_sh = by_id[inner], by_id[outer]
            ik = (inner_sh.get("kind") or "").lower()
            ok_ = (outer_sh.get("kind") or "").lower()
            poly = ("triangle", "square", "rectangle", "regular_polygon")
            if ok_ == "circle" and ik in poly:
                # polygon inscribed in circle -> circle is the circumcircle
                in_pts = (_draw_triangle(ax, inner_sh, registry=registry) if ik == "triangle"
                          else _draw_rect(ax, inner_sh, registry=registry) if ik in ("square", "rectangle")
                          else _draw_regular_polygon(ax, inner_sh, registry=registry))
                cen, r = _circumcircle_pts(in_pts)
                _draw_circle(ax, outer_sh, center=cen, radius=r, registry=registry)
            elif ok_ in ("square", "rectangle") and ik == "circle":
                # circle inscribed in square -> r = half the side
                sq_pts = _draw_rect(ax, outer_sh, registry=registry)
                cen = _centroid(sq_pts)
                r = (max(p[0] for p in sq_pts) - min(p[0] for p in sq_pts)) / 2
                _draw_circle(ax, inner_sh, center=cen, radius=r, registry=registry)
            else:
                _draw_shape(ax, outer_sh, registry=registry)
                _draw_shape(ax, inner_sh, registry=registry)
            drawn.update({inner, outer})

        for sh in shapes:
            if sh.get("id") in drawn:
                continue
            _draw_shape(ax, sh, registry=registry)

        # internal segments (diagonals, chords, radii) drawn last, over the shapes
        _draw_segments(ax, geo.get("segments"), registry)

        ax.margins(0.18)
        ax.relim()
        ax.autoscale_view()
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path.exists()
    except Exception as e:
        gen.log(f"[geometry] render failed: {e}", level="warning")
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ============================================================
# 2b. GEMINI IMAGE RENDERER (geometry / science / free figures)
# ============================================================
# Lazy: don't build the image client at import (a serve-only deploy has no key).
_img_client_obj = None


def _img_client():
    global _img_client_obj
    if _img_client_obj is None:
        _img_client_obj = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _img_client_obj


# When True, skip the rate-limited Gemini image model entirely. Charts go to
# matplotlib and shapes to the geometry kernel; anything that would need the
# image model simply renders no figure (and is filtered out upstream).
DISABLE_GEMINI_IMAGE = os.getenv("DISABLE_GEMINI_IMAGE", "0") in ("1", "true", "True")


def render_gemini_image(prompt: str, out_path: Path, max_retries: int = 4) -> bool:
    if DISABLE_GEMINI_IMAGE:
        return False
    if not prompt:
        return False
    full = (
        "Create a clean, precise black-and-white geometric/diagrammatic figure "
        "for a math/science exam question. Plain white background, no shading "
        "unless required, do NOT reveal the answer, no decorative elements.\n"
        "CRITICAL LABELING RULES:\n"
        "- Use ONLY Latin letters (A, B, C, x, y) and Western digits (0-9) for "
        "all labels and measurements.\n"
        "- Do NOT write ANY Arabic words or Arabic-Indic digits (٠١٢٣) in the "
        "figure — this model renders Arabic text as garbled nonsense.\n"
        "- Keep labels minimal: vertex letters, side lengths as numbers, angle "
        "values with the ° symbol. No sentences, no titles in Arabic.\n"
        "Figure description:\n" + prompt
    )
    import time
    for attempt in range(1, max_retries + 1):
        try:
            resp = _img_client().models.generate_content(
                model=IMAGE_MODEL, contents=full
            )
            for cand in (resp.candidates or []):
                for part in (cand.content.parts or []):
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        out_path.write_bytes(inline.data)
                        return out_path.exists()
            gen.log("[gemini-image] no image part returned", level="warning")
            return False
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg
                    or "UNAVAILABLE" in msg) and attempt < max_retries:
                wait = 15 * attempt
                gen.log(f"[gemini-image] rate/again, wait {wait}s "
                        f"({attempt}/{max_retries})", level="warning")
                time.sleep(wait)
                continue
            gen.log(f"[gemini-image] failed: {e}", level="warning")
            return False
    return False


# ============================================================
# IMAGE STAGE — run on every exported row
# ============================================================
# Vision critic: judge a RENDERED figure against the question + answer. Catches
# "figure drawn but doesn't actually support the answer / is mislabeled / ambiguous"
# — which existence-only checks miss. Set VISION_CRITIC_ENABLED=0 to skip.
VISION_CRITIC_ENABLED = os.getenv("VISION_CRITIC_ENABLED", "1") not in ("0", "false", "False")
VISION_CRITIC_MODEL = "gemini-2.5-flash"


def vision_critique_figure(image_path: Path, question: str, answer: str,
                           options: dict, lang: str) -> dict:
    """Return {"ok": bool, "reason": str}. Fail-open on any error (ok=True)."""
    if not VISION_CRITIC_ENABLED:
        return {"ok": True, "reason": ""}
    try:
        data = Path(image_path).read_bytes()
    except Exception:
        return {"ok": True, "reason": ""}
    opts = "\n".join(f"  {k}) {v}" for k, v in (options or {}).items() if v)
    prompt = (
        "You are checking whether a FIGURE correctly supports an exam question.\n"
        "Look ONLY at the image and judge:\n"
        "1. Does the figure contain the data/shape needed to reach the stated answer?\n"
        "2. Are the values/labels in the figure CONSISTENT with that answer "
        "(not contradicting it, not revealing it as a label)?\n"
        "3. Is the figure unambiguous and readable (no garbled/Arabic-as-glyph text, "
        "no missing labels)?\n\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        f"Options:\n{opts}\n\n"
        'Return ONLY JSON: {"ok": true|false, "reason": "<short why, if not ok>"}'
    )
    try:
        with gen._api_semaphore:
            resp = _img_client().models.generate_content(
                model=VISION_CRITIC_MODEL,
                contents=[
                    genai.types.Part.from_bytes(data=data, mime_type="image/png"),
                    prompt,
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0,
                    http_options=genai.types.HttpOptions(timeout=45_000),
                ),
            )
        txt = resp.text or "{}"
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0) if m else txt)
        return {"ok": bool(d.get("ok", True)), "reason": str(d.get("reason", ""))}
    except Exception as e:
        gen.log(f"[vision-critic] failed, passing open: {e}", level="warning")
        return {"ok": True, "reason": ""}


def add_images(rows: list, category: str) -> list:
    cat_slug = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_") or "misc"
    cat_img_dir = IMG_DIR / cat_slug
    cat_img_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        row.setdefault("NeedsImage", False)
        row.setdefault("RenderMethod", "")
        row.setdefault("ImagePath", "")
        row.setdefault("ImageSpec", "")
        row.setdefault("FigureFailReason", "")
        row.setdefault("FigureCriticPass", True)

        spec = classify(row)
        row["ImageSpec"] = json.dumps(spec, ensure_ascii=False)
        if not spec.get("needs_image"):
            row["FigureFailReason"] = "classifier marked question as not needing a figure"
            continue
        row["NeedsImage"] = True
        method = (spec.get("render") or "").lower()

        fname = f"{uuid.uuid4().hex}.png"
        out_path = cat_img_dir / fname
        ok = False
        if method == "matplotlib":
            ok = render_chart(spec, out_path)
            if not ok:  # fallback to gemini if chart spec was unusable
                method = "gemini"
        elif method == "geometry":
            ok = render_geometry(spec, out_path)
            if not ok:  # deterministic render failed -> gemini fallback
                method = "gemini"
        if method == "gemini" and not ok:
            ok = render_gemini_image(spec.get("image_prompt", ""), out_path)

        # A render that "succeeded" but produced no/!tiny file is a broken figure.
        MIN_PNG_BYTES = 1024
        if ok:
            try:
                if not out_path.exists() or out_path.stat().st_size < MIN_PNG_BYTES:
                    ok = False
            except OSError:
                ok = False

        if not ok:
            row["RenderMethod"] = ""
            row["ImagePath"] = ""
            row["FigureFailReason"] = f"render failed (method={method or 'none'})"
            gen.log("[image] generation failed; question kept text-only", level="warning")
            continue

        # Figure rendered — now judge its CORRECTNESS with the vision critic.
        rel_path = str(Path("images") / cat_slug / fname)
        verdict = vision_critique_figure(
            out_path, row.get("Question", ""), row.get("Answer", ""),
            {L: row.get(f"Option{L}", "") for L in ("A", "B", "C", "D")},
            row.get("Language", ""),
        )
        row["RenderMethod"] = method
        row["ImagePath"] = rel_path
        row["FigureCriticPass"] = verdict["ok"]
        if verdict["ok"]:
            gen.log(f"[image] {method} -> {rel_path} (vision OK)")
        else:
            row["FigureFailReason"] = f"figure mismatch: {verdict['reason']}"
            gen.log(f"[vision-critic] rejected {rel_path}: {verdict['reason']}",
                    level="warning")
    return rows


# ============================================================
# GENERATION (reusable — used by CLI and the unified web app)
# ============================================================
_DF_ALL = None
_PROMPTS = None


def _load_data():
    """Lazy-load + cache the sample sheet and category prompts."""
    global _DF_ALL, _PROMPTS
    if _DF_ALL is None:
        _DF_ALL = gen.load_exam_data(gen.FILE_PATH, gen.SHEETS_TO_LOAD)
    if _PROMPTS is None:
        try:
            _PROMPTS = gen.load_category_prompts(gen.PROMPTS_JSON)
        except Exception:
            _PROMPTS = {}
    return _DF_ALL, _PROMPTS


def options_for(section: str, lang: str = None):
    """Return {category: [subcats]} that actually have example rows for the
    selected section (and language, if given) — so the UI never offers a lesson
    the generator has no examples for."""
    df_all, _ = _load_data()
    df = df_all
    if lang:
        df = df[df["Language"] == lang]
    sec = "quantitative" if section == "Quantitative" else "verbal"
    df = df[df["Section"].astype(str).str.strip().str.lower() == sec]
    if section == "Verbal":
        df = df[df["Category"].isin(gen.ALLOWED_VERBAL_CATS)]
    out = {}
    for cat in sorted(c for c in df["Category"].dropna().astype(str).str.strip().unique()
                      if c and c.lower() not in ("nan", "none", "")):
        subs = sorted(s for s in df[df["Category"] == cat]["Sub-Category"]
                      .dropna().astype(str).str.strip().unique()
                      if s and s.lower() not in ("nan", "none", ""))
        out[cat] = subs
    return out


# Scenario-first geometry kernel: when on, Quantitative/Geometry is generated by
# geo_scenarios (figure-first, kernel-verified) instead of the text-first + LLM
# image stage. Set GEO_KERNEL_ENABLED=0 to fall back to the legacy path.
GEO_KERNEL_ENABLED = os.getenv("GEO_KERNEL_ENABLED", "1") not in ("0", "false", "False")

# Data-first Data-Interpretation path (di_scenarios): chart data is sampled and the
# answer computed from it, so the figure always supports a correct answer. Replaces
# the fragile text-first DI path. Set DI_KERNEL_ENABLED=0 to fall back.
DI_KERNEL_ENABLED = os.getenv("DI_KERNEL_ENABLED", "1") not in ("0", "false", "False")
CAT = "Data Interpretation & Reasoning"


def generate_questions(section, category, subcat, lang, num, save=True, gen_feedback=""):
    """Generate questions + run the image stage. Returns the list of row dicts.

    gen_feedback: optional guidance (e.g. why prior figures failed) injected into the
    generation prompt so the model fixes the issue instead of being blindly retried.
    """
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    # Scenario-first geometry path (figure drives question, kernel-verified answer).
    if GEO_KERNEL_ENABLED and section == "Quantitative" and category == "Geometry":
        try:
            import geo_scenarios
            rows = geo_scenarios.generate_geometry_questions(num, lang=lang, save=save)
            if rows:
                gen.log(f"[geo-kernel] generated {len(rows)} geometry questions")
                return rows
            gen.log("[geo-kernel] produced no rows; falling back to text-first", level="warning")
        except Exception as e:
            gen.log(f"[geo-kernel] failed ({e}); falling back to text-first", level="warning")

    # Data-first DI path: chart data drives the question, answer computed from data.
    if DI_KERNEL_ENABLED and section == "Quantitative" and category == CAT:
        try:
            import di_scenarios
            rows = di_scenarios.generate_di_questions(num, lang=lang, save=save)
            if rows:
                gen.log(f"[di-kernel] generated {len(rows)} data-interpretation questions")
                return rows
            gen.log("[di-kernel] produced no rows; falling back to text-first", level="warning")
        except Exception as e:
            gen.log(f"[di-kernel] failed ({e}); falling back to text-first", level="warning")

    df_all, prompts = _load_data()
    subcat_arg = subcat if (category != "ANY" and subcat not in (None, "", "ANY")) else None

    if section == "Quantitative":
        _, export_rows = gen.generate_valid_quant_mcq_questions(
            df_all, target_language=lang, category=category,
            subcat=subcat_arg, num_required=num, seen_questions=set(),
            extra_gen_feedback=gen_feedback,
        )
    else:
        _, export_rows = gen.generate_valid_verbal_mcq_questions(
            df_all, target_language=lang, category=category,
            subcat=subcat_arg, num_required=num, prompts_dict=prompts,
            seen_questions=set(), extra_gen_feedback=gen_feedback,
        )

    if not export_rows:
        gen.log("No questions generated.", level="error")
        return []

    gen.log(f"Generated {len(export_rows)} questions. Running image stage...")
    label = category if category != "ANY" else section
    rows = add_images(export_rows, label)

    if save:
        df = pd.DataFrame(rows)
        for c in IMG_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        cat_slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        out_xlsx = OUT_DIR / f"{cat_slug}.xlsx"
        df.to_excel(out_xlsx, index=False)
        n_img = int(df["NeedsImage"].astype(bool).sum())
        gen.log(f"Saved {len(df)} rows ({n_img} with images) -> {out_xlsx}")
    return rows


def run(section, category, subcat, lang, num):
    generate_questions(section, category, subcat, lang, num, save=True)


def main():
    ap = argparse.ArgumentParser(description="Image-capable Qudrat question generator")
    ap.add_argument("--section", required=True, choices=["Quantitative", "Verbal"])
    ap.add_argument("--category", default="ANY")
    ap.add_argument("--subcat", default="ANY")
    ap.add_argument("--lang", default="Arabic")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--image-model", default=None, help="override Gemini image model")
    args = ap.parse_args()

    if args.image_model:
        global IMAGE_MODEL
        IMAGE_MODEL = args.image_model

    run(args.section, args.category, args.subcat, args.lang, args.num)


if __name__ == "__main__":
    main()
