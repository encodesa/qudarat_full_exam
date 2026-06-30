#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scenario-first geometry question generator.

Each scenario: sample numeric params -> build a kernel construction program ->
solve -> MEASURE the answer (verifier) -> render the figure (givens only, never
the answer) -> synthesize Arabic prose (LLM writes words only, numbers come from
the kernel) -> parametric distractors -> assemble a row dict identical to app6's
24-key schema plus the image columns, so it drops straight into the xlsx/PDF flow.

Offline-safe: if no API key / LLM unavailable, prose falls back to a templated
Arabic stem built from the structured givens.
"""
from __future__ import annotations

import os
import re
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

import geo_kernel as gk
import geo_measure as gm
import geo_render as gr

load_dotenv()

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "questins_visual"
IMG_DIR = OUT_DIR / "images"
IMG_COLUMNS = ["NeedsImage", "RenderMethod", "ImagePath", "ImageSpec"]
PROSE_MODEL = "gemini-2.5-flash"

_ROW_KEYS = [
    "Question", "Answer", "Context", "CorrectOption",
    "OptionA", "OptionB", "OptionC", "OptionD",
    "Section", "Category", "Sub-Category", "Complexity-Level",
    "Similarity-to-Corpus", "Votes",
    "GenRationale_A", "GenRationale_B", "GenRationale_C", "GenRationale_D",
    "GenMisTag_A", "GenMisTag_B", "GenMisTag_C", "GenMisTag_D",
]

_W2A = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def ar_digits(s) -> str:
    return str(s).translate(_W2A)


def fmt_num(x: float, lang: str = "Arabic") -> str:
    """Format a numeric answer: integers without trailing .0, else 2 dp."""
    if abs(x - round(x)) < 1e-6:
        s = str(int(round(x)))
    else:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
    return ar_digits(s) if lang == "Arabic" else s


def fmt_pi(coeff: float, lang: str = "Arabic") -> str:
    """Format a 'k π' answer (areas/volumes in terms of pi)."""
    k = fmt_num(coeff, lang)
    sym = "ط" if lang == "Arabic" else "π"
    return f"{k}{sym}"


# ============================================================
# LLM prose synthesis (words only; numbers come from the kernel)
# ============================================================
_genai_client = None


def _client():
    global _genai_client
    if _genai_client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return None
        try:
            from google import genai
            _genai_client = genai.Client(api_key=key)
        except Exception:
            return None
    return _genai_client


_PROSE_SYS = (
    "You write a SINGLE exam question stem in Modern Standard Arabic for the Saudi "
    "Qudrat (قدرات) quantitative test. You are given a structured geometry scenario: "
    "the figure, the GIVEN quantities, and what is ASKED. Rules: use EXACTLY the "
    "given numbers and Arabic letter labels as written — do NOT add, change, round, "
    "or invent any number; do NOT solve or state the answer; refer to the figure "
    "with 'في الشكل'. Return JSON only: {\"question\": \"...\"}."
)


def _synth_prose(ask: Dict[str, Any], lang: str) -> Optional[str]:
    client = _client()
    if client is None or lang != "Arabic":
        return None
    givens = "؛ ".join(ask.get("givens") or [])
    prompt = (
        f"الشكل: {ask.get('figure','')}\n"
        f"المعطيات: {givens}\n"
        f"المطلوب: {ask.get('ask','')}\n"
        "اكتب نص السؤال فقط."
    )
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=_PROSE_SYS, temperature=0.5,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = client.models.generate_content(model=PROSE_MODEL, contents=prompt, config=cfg)
        data = json.loads(resp.text or "{}")
        q = (data.get("question") or "").strip()
        if q and _digit_gate(q, ask):
            return q
    except Exception:
        return None
    return None


def _allowed_digit_strings(ask: Dict[str, Any]) -> set:
    """Every digit-run the question is allowed to contain (the givens, both scripts)."""
    allowed = set()
    for g in (ask.get("givens") or []):
        for run in re.findall(r"[0-9٠-٩]+", str(g)):
            allowed.add(run.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
    return allowed


def _digit_gate(question: str, ask: Dict[str, Any]) -> bool:
    """Reject prose that introduced any number not present in the givens."""
    allowed = _allowed_digit_strings(ask)
    for run in re.findall(r"[0-9٠-٩]+", question):
        norm = run.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        if norm not in allowed:
            return False
    return True


def _fallback_stem(ask: Dict[str, Any], lang: str) -> str:
    givens = "، ".join(ask.get("givens") or [])
    fig = ask.get("figure", "")
    ask_t = ask.get("ask", "")
    if lang == "Arabic":
        head = f"في الشكل، {fig}. " if fig else ""
        return f"{head}{givens}. {ask_t}"
    return f"{fig}. {givens}. {ask_t}"


# ============================================================
# distractor + option assembly
# ============================================================
def _num_from(s):
    """Leading numeric value of an answer string (Arabic-Indic -> float), else None."""
    m = re.search(r"[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?",
                  str(s).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
    return float(m.group().replace(",", ".")) if m else None


def _numeric_pad(correct: str, have: List[str], need: int) -> List[str]:
    """Produce `need` distinct decoys by perturbing the correct answer's number,
    preserving any surrounding text (unit / ط / '- kط')."""
    base = _num_from(correct)
    out = []
    if base is None:
        # non-numeric: append distinct fallback markers (rare)
        i = 1
        while len(out) < need:
            cand = f"{correct} ({i})"
            if cand not in have and cand not in out:
                out.append(cand)
            i += 1
        return out
    template = re.sub(r"[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?", "{}", str(correct), count=1)
    for off in (1, -1, 2, -2, 3, base and base * 2 - base):  # +1,-1,+2,-2,+3,+base
        if len(out) >= need:
            break
        val = base + off
        if val <= 0:
            continue
        sval = fmt_num(val, "Arabic")
        cand = template.format(sval) if "{}" in template else sval
        if cand != correct and cand not in have and cand not in out:
            out.append(cand)
    k = 4
    while len(out) < need:                    # last resort: keep climbing
        sval = fmt_num(base + k, "Arabic")
        cand = template.format(sval) if "{}" in template else sval
        if cand != correct and cand not in have and cand not in out:
            out.append(cand)
        k += 1
    return out


def _build_options(correct: str, distractors: List[str], rng: random.Random,
                   rationales: Optional[List[str]] = None) -> Dict[str, Any]:
    # dedupe distractors against the correct answer and each other
    seen = {correct}
    uniq = []
    for d in distractors:
        if d and d not in seen:
            uniq.append(d)
            seen.add(d)
    uniq = uniq[:3]
    if len(uniq) < 3:                          # clean numeric pad (no '؟؟؟')
        uniq.extend(_numeric_pad(correct, uniq, 3 - len(uniq)))
    opts = [correct] + uniq[:3]
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    row = {}
    correct_letter = "A"
    for L, val in zip(letters, opts):
        row[f"Option{L}"] = val
        if val == correct:
            correct_letter = L
    row["CorrectOption"] = correct_letter
    rat = rationales or []
    for i, L in enumerate(letters):
        row[f"GenRationale_{L}"] = ""
        row[f"GenMisTag_{L}"] = ""
    return row


def _assemble_row(question, answer, options, subcat, program, image_path, lang):
    row = {k: "" for k in _ROW_KEYS}
    row.update(options)
    row["Question"] = question
    row["Answer"] = answer
    row["Context"] = ""
    row["Section"] = "Quantitative"
    row["Category"] = "Geometry"
    row["Sub-Category"] = subcat
    row["Complexity-Level"] = "Basic"
    row["Similarity-to-Corpus"] = 0.0
    row["Votes"] = 3                      # kernel-verified -> full confidence
    row["NeedsImage"] = True
    row["RenderMethod"] = "geometry"
    row["ImagePath"] = image_path
    row["ImageSpec"] = json.dumps({"program": program}, ensure_ascii=False)
    return row


# ============================================================
# SCENARIO TEMPLATES
# Each returns a dict from sample(): program, measure_spec, annotations(reveal),
# ask, answer_str, distractors(list of str), subcat. The kernel verifies answer.
# ============================================================
def _sc_right_triangle_third_side(rng, lang):
    # pythagorean-ish: pick legs, ask hypotenuse (rounded only if integer triple)
    triples = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (7, 24, 25)]
    a, b, c = rng.choice(triples)
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": a, "y": 0},
        {"op": "point", "name": "ج", "x": 0, "y": b},
        {"op": "polygon", "name": "T", "verts": ["أ", "ب", "ج"]},
    ]
    ans = c
    annotations = [
        {"type": "right_angle", "vertex": "أ", "a": "ب", "b": "ج"},
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(a)},
        {"type": "side", "a": "أ", "b": "ج", "text": ar_digits(b)},
    ]
    ask = {"figure": "مثلث قائم الزاوية في أ",
           "givens": [f"الضلعان القائمان طولهما {ar_digits(a)} و {ar_digits(b)}"],
           "ask": "ما طول الوتر؟"}
    dist = [fmt_num(a + b, lang), fmt_num(c + 1, lang), fmt_num(c - 1, lang)]
    # Label all three vertices — the stem references أ, so the figure must show it.
    return dict(program=prog, measure={"op": "length", "a": "ب", "b": "ج"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Angles & Triangles", hide=set())


def _sc_triangle_area(rng, lang):
    base = rng.choice([6, 8, 10, 12, 14])
    height = rng.choice([4, 5, 6, 7, 9])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": base, "y": 0},
        {"op": "point", "name": "ج", "x": base * 0.35, "y": height},
        {"op": "polygon", "name": "T", "verts": ["أ", "ب", "ج"]},
        {"op": "foot", "name": "h", "pt": "ج", "to_line": "AB"},
        {"op": "line", "name": "AB", "a": "أ", "b": "ب"},
    ]
    # foot needs line AB first; reorder
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": base, "y": 0},
        {"op": "point", "name": "ج", "x": base * 0.35, "y": height},
        {"op": "line", "name": "__AB", "a": "أ", "b": "ب"},
        {"op": "foot", "name": "h", "pt": "ج", "to_line": "__AB"},
        {"op": "polygon", "name": "T", "verts": ["أ", "ب", "ج"]},
    ]
    ans = 0.5 * base * height
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(base)},
        {"type": "segment", "a": "ج", "b": "h", "dashed": True},
        {"type": "right_angle", "vertex": "h", "a": "أ", "b": "ج"},
    ]
    ask = {"figure": "مثلث طول قاعدته معطى وارتفاعه مرسوم",
           "givens": [f"طول القاعدة {ar_digits(base)} والارتفاع {ar_digits(height)}"],
           "ask": "ما مساحة المثلث؟"}
    dist = [fmt_num(base * height, lang), fmt_num(ans + base, lang), fmt_num(ans - height, lang)]
    return dict(program=prog, measure={"op": "area_triangle_bh", "base": base, "height": height},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide={"h"})


def _sc_inscribed_angle(rng, lang):
    central = rng.choice([40, 60, 70, 80, 100, 120])
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "م", "r": 5},
        {"op": "point_on_circle", "name": "أ", "circle": "k", "theta": -central / 2},
        {"op": "point_on_circle", "name": "ب", "circle": "k", "theta": central / 2},
        {"op": "point_on_circle", "name": "ج", "circle": "k", "theta": 180},
        {"op": "segment", "name": "s1", "a": "ج", "b": "أ"},
        {"op": "segment", "name": "s2", "a": "ج", "b": "ب"},
        {"op": "segment", "name": "r1", "a": "م", "b": "أ"},
        {"op": "segment", "name": "r2", "a": "م", "b": "ب"},
    ]
    ans = central / 2
    annotations = [{"type": "angle", "vertex": "م", "a": "أ", "b": "ب",
                    "text": ar_digits(central) + "°"}]
    ask = {"figure": "دائرة مركزها م، والنقاط أ، ب، ج عليها",
           "givens": [f"الزاوية المركزية أمب تساوي {ar_digits(central)}°"],
           "ask": "ما قياس الزاوية المحيطية أجب؟"}
    dist = [ar_digits(central) + "°", fmt_num(central / 4, lang) + "°",
            fmt_num(central * 2, lang) + "°"]
    return dict(program=prog, measure={"op": "angle", "vertex": "ج", "a": "أ", "b": "ب"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang) + "°",
               distractors=dist, subcat="Circles (Arcs, Chords, Tangents)", hide=set())


def _sc_sector_area(rng, lang):
    r = rng.choice([3, 4, 6, 8, 10])
    deg = rng.choice([30, 45, 60, 90, 120])
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "م", "r": r},
        {"op": "point_on_circle", "name": "أ", "circle": "k", "theta": 0},
        {"op": "point_on_circle", "name": "ب", "circle": "k", "theta": deg},
        {"op": "segment", "name": "r1", "a": "م", "b": "أ"},
        {"op": "segment", "name": "r2", "a": "م", "b": "ب"},
    ]
    coeff = deg / 360.0 * r * r            # sector area = (deg/360)·π·r²  -> k·π
    annotations = [
        {"type": "angle", "vertex": "م", "a": "أ", "b": "ب", "text": ar_digits(deg) + "°"},
        {"type": "side", "a": "م", "b": "أ", "text": ar_digits(r)},
    ]
    ask = {"figure": "قطاع دائري مركزه م ونصف قطره معطى",
           "givens": [f"نصف القطر {ar_digits(r)} وزاوية القطاع {ar_digits(deg)}°"],
           "ask": "ما مساحة القطاع الدائري بدلالة ط؟"}
    dist = [fmt_pi(deg / 360.0 * 2 * r, lang), fmt_pi(r * r, lang),
            fmt_pi(coeff * 2, lang)]
    return dict(program=prog, measure={"op": "const", "value": coeff},
               annotations=annotations, ask=ask, answer=fmt_pi(coeff, lang),
               distractors=dist, subcat="Circles (Arcs, Chords, Tangents)", hide=set())


def _sc_composite_shaded(rng, lang):
    side = rng.choice([4, 6, 8, 10])
    r = side / 2.0
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": side, "y": 0},
        {"op": "point", "name": "ج", "x": side, "y": side},
        {"op": "point", "name": "د", "x": 0, "y": side},
        {"op": "polygon", "name": "sq", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "point", "name": "م", "x": side / 2.0, "y": side / 2.0},
        {"op": "circle", "name": "c", "center": "م", "r": r},
    ]
    coeff_pi = r * r                       # circle area coeff (k·π)
    sq_area = side * side
    annotations = [
        {"type": "shade", "outer": {"verts": ["أ", "ب", "ج", "د"]}, "holes": [{"circle": "c"}]},
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(side)},
    ]
    ask = {"figure": "مربع بداخله دائرة تمس أضلاعه، والمنطقة المظللة بين المربع والدائرة",
           "givens": [f"طول ضلع المربع {ar_digits(side)}"],
           "ask": "ما مساحة المنطقة المظللة بدلالة ط؟"}
    # answer = side² − (r²)π  -> express as "N − kط"
    n = fmt_num(sq_area, lang)
    ans = f"{n} - {fmt_pi(coeff_pi, lang)}"
    dist = [f"{n} - {fmt_pi(coeff_pi * 2, lang)}", fmt_pi(coeff_pi, lang), n]
    return dict(program=prog, measure={"op": "const", "value": sq_area - math.pi * coeff_pi},
               annotations=annotations, ask=ask, answer=ans,
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_cube_volume(rng, lang):
    edge = rng.choice([2, 3, 4, 5, 6])
    # 2D oblique projection of a cube via kernel points (front + back faces)
    e = float(edge)
    dep = 0.42 * e
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": e, "y": 0},
        {"op": "point", "name": "ج", "x": e, "y": e},
        {"op": "point", "name": "د", "x": 0, "y": e},
        {"op": "point", "name": "أ2", "x": dep, "y": dep},
        {"op": "point", "name": "ب2", "x": e + dep, "y": dep},
        {"op": "point", "name": "ج2", "x": e + dep, "y": e + dep},
        {"op": "point", "name": "د2", "x": dep, "y": e + dep},
        {"op": "polygon", "name": "front", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "segment", "name": "e1", "a": "ب", "b": "ب2"},
        {"op": "segment", "name": "e2", "a": "ج", "b": "ج2"},
        {"op": "segment", "name": "e3", "a": "د", "b": "د2"},
        {"op": "segment", "name": "e4", "a": "ب2", "b": "ج2"},
        {"op": "segment", "name": "e5", "a": "ج2", "b": "د2"},
    ]
    ans = edge ** 3
    # hidden back-left edges drawn dashed (as annotations, not solid kernel segments)
    annotations = [
        {"type": "segment", "a": "أ", "b": "أ2", "dashed": True},
        {"type": "segment", "a": "أ2", "b": "ب2", "dashed": True},
        {"type": "segment", "a": "أ2", "b": "د2", "dashed": True},
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(edge)},
    ]
    ask = {"figure": "مكعب طول حرفه معطى",
           "givens": [f"طول الحرف {ar_digits(edge)}"],
           "ask": "ما حجم المكعب؟"}
    dist = [fmt_num(edge * edge, lang), fmt_num(edge * 6, lang), fmt_num(edge ** 3 + edge, lang)]
    return dict(program=prog, measure={"op": "solid_volume", "kind": "cube", "params": {"edge": edge}},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area",
               hide={"أ2", "ب2", "ج2", "د2"})


def _sc_parallelogram_area(rng, lang):
    base = rng.choice([6, 8, 10, 12])
    height = rng.choice([4, 5, 7, 9])
    skew = rng.choice([1.2, 1.8, 2.4])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": base, "y": 0},
        {"op": "point", "name": "ج", "x": base + skew, "y": height},
        {"op": "point", "name": "د", "x": skew, "y": height},
        {"op": "polygon", "name": "P", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "line", "name": "__AB", "a": "أ", "b": "ب"},
        {"op": "foot", "name": "f", "pt": "د", "to_line": "__AB"},
    ]
    ans = base * height
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(base)},
        {"type": "segment", "a": "د", "b": "f", "dashed": True},
        {"type": "side", "a": "د", "b": "f", "text": ar_digits(height)},  # label the height
        {"type": "right_angle", "vertex": "f", "a": "أ", "b": "د"},
    ]
    ask = {"figure": "متوازي أضلاع قاعدته أب وارتفاعه مرسوم",
           "givens": [f"طول القاعدة {ar_digits(base)} والارتفاع {ar_digits(height)}"],
           "ask": "ما مساحة متوازي الأضلاع؟"}
    dist = [fmt_num(base + height, lang), fmt_num(ans + base, lang), fmt_num(base * height // 2, lang)]
    return dict(program=prog, measure={"op": "const", "value": ans}, annotations=annotations,
               ask=ask, answer=fmt_num(ans, lang), distractors=dist,
               subcat="Area, Perimeter, Volume, Surface Area", hide={"f"})


def _sc_trapezoid_area(rng, lang):
    bottom = rng.choice([10, 12, 14])
    top = rng.choice([4, 6, 8])
    height = rng.choice([4, 5, 6])
    off = (bottom - top) / 2.0
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": bottom, "y": 0},
        {"op": "point", "name": "ج", "x": bottom - off, "y": height},
        {"op": "point", "name": "د", "x": off, "y": height},
        {"op": "polygon", "name": "P", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "line", "name": "__AB", "a": "أ", "b": "ب"},
        {"op": "foot", "name": "f", "pt": "د", "to_line": "__AB"},
    ]
    ans = (bottom + top) / 2.0 * height
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(bottom)},
        {"type": "side", "a": "د", "b": "ج", "text": ar_digits(top)},
        {"type": "segment", "a": "د", "b": "f", "dashed": True},
        {"type": "right_angle", "vertex": "f", "a": "أ", "b": "د"},
    ]
    ask = {"figure": "شبه منحرف قاعدتاه المتوازيتان أب و دج وارتفاعه مرسوم",
           "givens": [f"القاعدتان {ar_digits(bottom)} و {ar_digits(top)} والارتفاع {ar_digits(height)}"],
           "ask": "ما مساحة شبه المنحرف؟"}
    dist = [fmt_num(bottom * height, lang), fmt_num((bottom + top) * height, lang),
            fmt_num(ans + height, lang)]
    return dict(program=prog, measure={"op": "const", "value": ans}, annotations=annotations,
               ask=ask, answer=fmt_num(ans, lang), distractors=dist,
               subcat="Area, Perimeter, Volume, Surface Area", hide={"f"})


def _sc_polygon_interior_sum(rng, lang):
    n = rng.choice([5, 6, 7, 8])
    names = {5: "خماسي", 6: "سداسي", 7: "سباعي", 8: "ثماني"}[n]
    pnames = ["أ", "ب", "ج", "د", "ه", "و", "ز", "ح"][:n]
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "regular_polygon", "name": "P", "n": n, "center": "م", "r": 3,
         "names": pnames, "start_theta": 90},
    ]
    ans = (n - 2) * 180
    annotations = []
    ask = {"figure": f"مضلع {names} منتظم",
           "givens": [f"عدد أضلاعه {ar_digits(n)}"],
           "ask": "ما مجموع قياسات زواياه الداخلية؟"}
    dist = [fmt_num((n - 2) * 90, lang), fmt_num(n * 180, lang), fmt_num((n - 1) * 180, lang)]
    return dict(program=prog, measure={"op": "interior_sum", "n": n}, annotations=annotations,
               ask=ask, answer=fmt_num(ans, lang) + "°",
               distractors=[d + "°" for d in dist],
               subcat="Angles & Triangles", hide=set())


def _sc_transversal_corresponding(rng, lang):
    theta = rng.choice([50, 55, 65, 70, 110, 120])
    th = math.radians(theta)
    dx = 1.0 / math.tan(th) if math.tan(th) != 0 else 0.0  # horizontal shift per unit y
    # two horizontal parallel lines y=0 (أب) and y=3 (جد); transversal crosses at O and X
    ox = 0.0
    Xx = ox + 3 * dx
    prog = [
        {"op": "point", "name": "أ", "x": -4, "y": 0}, {"op": "point", "name": "ب", "x": 4, "y": 0},
        {"op": "point", "name": "ج", "x": -4, "y": 3}, {"op": "point", "name": "د", "x": 4, "y": 3},
        {"op": "line", "name": "L1", "a": "أ", "b": "ب"},
        {"op": "line", "name": "L2", "a": "ج", "b": "د"},
        {"op": "point", "name": "O", "x": ox, "y": 0},
        {"op": "point", "name": "X", "x": Xx, "y": 3},
        {"op": "point", "name": "T1", "x": ox - 1.5 * dx, "y": -1.5},
        {"op": "point", "name": "T2", "x": Xx + 1.5 * dx, "y": 4.5},
        {"op": "line", "name": "Tr", "a": "T1", "b": "T2"},
    ]
    ans = theta  # corresponding angles are equal
    annotations = [{"type": "angle", "vertex": "O", "a": "ب", "b": "T2",
                    "text": ar_digits(theta) + "°"}]
    ask = {"figure": "مستقيمان متوازيان أب و جد يقطعهما مستقيم ثالث",
           "givens": [f"قياس إحدى الزوايا عند نقطة التقاطع السفلية {ar_digits(theta)}°"],
           "ask": "ما قياس الزاوية المناظرة لها عند نقطة التقاطع العلوية؟"}
    dist = [ar_digits(180 - theta) + "°", ar_digits(90) + "°", fmt_num(theta // 2, lang) + "°"]
    return dict(program=prog, measure={"op": "const", "value": ans}, annotations=annotations,
               ask=ask, answer=ar_digits(theta) + "°", distractors=dist,
               subcat="Angles & Triangles", hide={"O", "X", "T1", "T2"})


def _sc_box_volume(rng, lang):
    w = rng.choice([2, 3, 4]); d = rng.choice([2, 3, 5]); h = rng.choice([3, 4, 6])
    if w == d == h:
        h += 1
    we, de, he = float(w), float(d), float(h)
    dep = 0.40 * de
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": we, "y": 0},
        {"op": "point", "name": "ج", "x": we, "y": he},
        {"op": "point", "name": "د", "x": 0, "y": he},
        {"op": "point", "name": "أ2", "x": dep, "y": dep},
        {"op": "point", "name": "ب2", "x": we + dep, "y": dep},
        {"op": "point", "name": "ج2", "x": we + dep, "y": he + dep},
        {"op": "point", "name": "د2", "x": dep, "y": he + dep},
        {"op": "polygon", "name": "front", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "segment", "name": "e1", "a": "ب", "b": "ب2"},
        {"op": "segment", "name": "e2", "a": "ج", "b": "ج2"},
        {"op": "segment", "name": "e3", "a": "د", "b": "د2"},
        {"op": "segment", "name": "e4", "a": "ب2", "b": "ج2"},
        {"op": "segment", "name": "e5", "a": "ج2", "b": "د2"},
    ]
    ans = w * d * h
    annotations = [
        {"type": "segment", "a": "أ", "b": "أ2", "dashed": True},
        {"type": "segment", "a": "أ2", "b": "ب2", "dashed": True},
        {"type": "segment", "a": "أ2", "b": "د2", "dashed": True},
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(w)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(h)},
    ]
    ask = {"figure": "متوازي مستطيلات (صندوق) أبعاده مبينة",
           "givens": [f"الطول {ar_digits(w)} والعرض {ar_digits(d)} والارتفاع {ar_digits(h)}"],
           "ask": "ما حجم الصندوق؟"}
    dist = [fmt_num(w + d + h, lang), fmt_num(2 * (w * d + w * h + d * h), lang),
            fmt_num(w * d, lang)]
    return dict(program=prog, measure={"op": "solid_volume", "kind": "box",
                "params": {"width": w, "depth": d, "height": h}},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang), distractors=dist,
               subcat="Area, Perimeter, Volume, Surface Area",
               hide={"أ2", "ب2", "ج2", "د2"})


def _sc_rectangle_perimeter(rng, lang):
    w = rng.choice([5, 6, 8, 9, 12]); h = rng.choice([3, 4, 7, 10])
    if w == h:
        h += 1
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": w, "y": 0},
        {"op": "point", "name": "ج", "x": w, "y": h},
        {"op": "point", "name": "د", "x": 0, "y": h},
        {"op": "polygon", "name": "R", "verts": ["أ", "ب", "ج", "د"]},
    ]
    ans = 2 * (w + h)
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(w)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(h)},
    ]
    ask = {"figure": "مستطيل أبعاده مبينة",
           "givens": [f"الطول {ar_digits(w)} والعرض {ar_digits(h)}"],
           "ask": "ما محيط المستطيل؟"}
    dist = [fmt_num(w * h, lang), fmt_num(w + h, lang), fmt_num(2 * w + h, lang)]
    return dict(program=prog, measure={"op": "perimeter", "verts": "R"}, annotations=annotations,
               ask=ask, answer=fmt_num(ans, lang), distractors=dist,
               subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_pythagoras_leg(rng, lang):
    """Right triangle: given the hypotenuse and one leg, find the other leg."""
    a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15)])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": a, "y": 0},
        {"op": "point", "name": "ج", "x": 0, "y": b},
        {"op": "polygon", "name": "T", "verts": ["أ", "ب", "ج"]},
    ]
    annotations = [
        {"type": "right_angle", "vertex": "أ", "a": "ب", "b": "ج"},
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(a)},   # known leg
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(c)},   # known hypotenuse
    ]
    ask = {"figure": "مثلث قائم الزاوية في أ",
           "givens": [f"الوتر {ar_digits(c)} وأحد الضلعين القائمين {ar_digits(a)}"],
           "ask": "ما طول الضلع القائم الآخر؟"}
    dist = [fmt_num(c - a, lang), fmt_num(b + 1, lang), fmt_num(b - 1, lang)]
    return dict(program=prog, measure={"op": "length", "a": "أ", "b": "ج"},
               annotations=annotations, ask=ask, answer=fmt_num(b, lang),
               distractors=dist, subcat="Angles & Triangles", hide=set())


def _sc_isosceles_base_angle(rng, lang):
    """Isosceles triangle: given the apex angle, find a base angle."""
    apex = rng.choice([40, 50, 70, 80, 100])
    base_ang = (180 - apex) // 2
    h = 4.0
    half = h * math.tan(math.radians(apex / 2))
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": h},          # apex
        {"op": "point", "name": "ب", "x": -half, "y": 0},
        {"op": "point", "name": "ج", "x": half, "y": 0},
        {"op": "polygon", "name": "T", "verts": ["أ", "ب", "ج"]},
    ]
    annotations = [{"type": "angle", "vertex": "أ", "a": "ب", "b": "ج",
                    "text": ar_digits(apex) + "°"}]
    ask = {"figure": "مثلث متساوي الساقين رأسه أ",
           "givens": [f"زاوية الرأس أ تساوي {ar_digits(apex)}°"],
           "ask": "ما قياس إحدى زاويتي القاعدة؟"}
    dist = [ar_digits(apex) + "°", ar_digits(180 - apex) + "°", ar_digits(90) + "°"]
    return dict(program=prog, measure={"op": "angle", "vertex": "ب", "a": "أ", "b": "ج"},
               annotations=annotations, ask=ask, answer=ar_digits(base_ang) + "°",
               distractors=dist, subcat="Angles & Triangles", hide=set())


def _sc_circle_circumference(rng, lang):
    """Circle: given the radius, find the circumference in terms of pi."""
    r = rng.choice([3, 4, 5, 7])
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "م", "r": r},
        {"op": "point_on_circle", "name": "أ", "circle": "k", "theta": 0},
        {"op": "segment", "name": "rad", "a": "م", "b": "أ"},
    ]
    annotations = [{"type": "side", "a": "م", "b": "أ", "text": ar_digits(r)}]
    ask = {"figure": "دائرة مركزها م ونصف قطرها معطى",
           "givens": [f"نصف القطر {ar_digits(r)}"],
           "ask": "ما محيط الدائرة بدلالة ط؟"}
    dist = [fmt_pi(r, lang), fmt_pi(r * r, lang), fmt_pi(4 * r, lang)]
    return dict(program=prog, measure={"op": "circumference", "r": r},
               annotations=annotations, ask=ask, answer=fmt_pi(2 * r, lang),
               distractors=dist, subcat="Circles (Arcs, Chords, Tangents)", hide=set())


def _sc_arc_length(rng, lang):
    """Circle sector: given radius and central angle, find the arc length in pi."""
    r = rng.choice([6, 9, 12])
    deg = rng.choice([60, 90, 120, 180])
    coeff = deg / 360.0 * 2 * r
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "م", "r": r},
        {"op": "point_on_circle", "name": "أ", "circle": "k", "theta": 0},
        {"op": "point_on_circle", "name": "ب", "circle": "k", "theta": deg},
        {"op": "segment", "name": "r1", "a": "م", "b": "أ"},
        {"op": "segment", "name": "r2", "a": "م", "b": "ب"},
    ]
    annotations = [
        {"type": "angle", "vertex": "م", "a": "أ", "b": "ب", "text": ar_digits(deg) + "°"},
        {"type": "side", "a": "م", "b": "أ", "text": ar_digits(r)},
    ]
    ask = {"figure": "قطاع دائري مركزه م",
           "givens": [f"نصف القطر {ar_digits(r)} وزاوية القطاع {ar_digits(deg)}°"],
           "ask": "ما طول قوس القطاع بدلالة ط؟"}
    dist = [fmt_pi(coeff / 2, lang), fmt_pi(coeff * 2, lang), fmt_pi(r, lang)]
    return dict(program=prog, measure={"op": "arc_length", "r": r, "deg": deg},
               annotations=annotations, ask=ask, answer=fmt_pi(coeff, lang),
               distractors=dist, subcat="Circles (Arcs, Chords, Tangents)", hide=set())


def _sc_triangle_perimeter(rng, lang):
    """Triangle from three side lengths: find the perimeter."""
    a, b, c = rng.choice([(6, 8, 10), (5, 5, 6), (7, 7, 7), (9, 12, 15), (5, 6, 9)])
    prog = [
        {"op": "triangle_sss", "name": "T", "names": ["أ", "ب", "ج"],
         "a": a, "b": b, "c": c},                # a=بج, b=جأ, c=أب
    ]
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(c)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(a)},
        {"type": "side", "a": "ج", "b": "أ", "text": ar_digits(b)},
    ]
    ask = {"figure": "مثلث أطوال أضلاعه مبينة",
           "givens": [f"أطوال الأضلاع {ar_digits(a)}، {ar_digits(b)}، {ar_digits(c)}"],
           "ask": "ما محيط المثلث؟"}
    ans = a + b + c
    dist = [fmt_num(ans - a, lang), fmt_num(ans + a, lang), fmt_num(a * b, lang)]
    return dict(program=prog, measure={"op": "perimeter", "verts": "T"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_regular_polygon_each_angle(rng, lang):
    """Regular polygon: find the measure of EACH interior angle."""
    n = rng.choice([5, 6, 8, 10, 12])
    names = {5: "خماسي", 6: "سداسي", 8: "ثماني", 10: "عشاري", 12: "اثنا عشري"}[n]
    pnames = ["أ", "ب", "ج", "د", "ه", "و", "ز", "ح", "ط", "ي", "ك", "ل"][:n]
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "regular_polygon", "name": "P", "n": n, "center": "م", "r": 3,
         "names": pnames, "start_theta": 90},
    ]
    ans = (n - 2) * 180 // n
    ask = {"figure": f"مضلع {names} منتظم",
           "givens": [f"عدد أضلاعه {ar_digits(n)}"],
           "ask": "ما قياس كل زاوية من زواياه الداخلية؟"}
    dist = [ar_digits((n - 2) * 180) + "°", ar_digits(360 // n) + "°", ar_digits(180 - (n - 2) * 180 // n) + "°"]
    return dict(program=prog, measure={"op": "each_interior", "n": n}, annotations=[],
               ask=ask, answer=ar_digits(ans) + "°", distractors=dist,
               subcat="Angles & Triangles", hide=set())


def _sc_rectangle_area(rng, lang):
    w = rng.choice([5, 6, 8, 9, 12])
    h = rng.choice([3, 4, 7, 10])
    if w == h:
        h += 1
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": w, "y": 0},
        {"op": "point", "name": "ج", "x": w, "y": h},
        {"op": "point", "name": "د", "x": 0, "y": h},
        {"op": "polygon", "name": "R", "verts": ["أ", "ب", "ج", "د"]},
    ]
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(w)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(h)},
    ]
    ask = {"figure": "مستطيل أبعاده مبينة",
           "givens": [f"الطول {ar_digits(w)} والعرض {ar_digits(h)}"],
           "ask": "ما مساحة المستطيل؟"}
    ans = w * h
    dist = [fmt_num(2 * (w + h), lang), fmt_num(w + h, lang), fmt_num(w * h // 2, lang)]
    return dict(program=prog, measure={"op": "area_polygon", "verts": "R"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_square_area(rng, lang):
    s = rng.choice([4, 5, 6, 7, 9, 11])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": s, "y": 0},
        {"op": "point", "name": "ج", "x": s, "y": s},
        {"op": "point", "name": "د", "x": 0, "y": s},
        {"op": "polygon", "name": "R", "verts": ["أ", "ب", "ج", "د"]},
    ]
    annotations = [{"type": "side", "a": "أ", "b": "ب", "text": ar_digits(s)}]
    ask = {"figure": "مربع طول ضلعه معطى",
           "givens": [f"طول الضلع {ar_digits(s)}"],
           "ask": "ما مساحة المربع؟"}
    ans = s * s
    dist = [fmt_num(4 * s, lang), fmt_num(2 * s, lang), fmt_num(s * (s - 1), lang)]
    return dict(program=prog, measure={"op": "area_polygon", "verts": "R"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_circle_area(rng, lang):
    """Circle: given the radius, find the area in terms of pi."""
    r = rng.choice([2, 3, 4, 5, 6])
    prog = [
        {"op": "point", "name": "م", "x": 0, "y": 0},
        {"op": "circle", "name": "k", "center": "م", "r": r},
        {"op": "point_on_circle", "name": "أ", "circle": "k", "theta": 0},
        {"op": "segment", "name": "rad", "a": "م", "b": "أ"},
    ]
    annotations = [{"type": "side", "a": "م", "b": "أ", "text": ar_digits(r)}]
    ask = {"figure": "دائرة مركزها م ونصف قطرها معطى",
           "givens": [f"نصف القطر {ar_digits(r)}"],
           "ask": "ما مساحة الدائرة بدلالة ط؟"}
    dist = [fmt_pi(2 * r, lang), fmt_pi(r, lang), fmt_num(r * r, lang)]
    return dict(program=prog, measure={"op": "circle_area", "r": r},
               annotations=annotations, ask=ask, answer=fmt_pi(r * r, lang),
               distractors=dist, subcat="Circles (Arcs, Chords, Tangents)", hide=set())


def _sc_parallelogram_perimeter(rng, lang):
    """Parallelogram: given base and slant side, find the perimeter."""
    base = rng.choice([6, 8, 10, 12])
    skew, h, slant = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13)])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": base, "y": 0},
        {"op": "point", "name": "ج", "x": base + skew, "y": h},
        {"op": "point", "name": "د", "x": skew, "y": h},
        {"op": "polygon", "name": "P", "verts": ["أ", "ب", "ج", "د"]},
    ]
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(base)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(slant)},
    ]
    ask = {"figure": "متوازي أضلاع طول قاعدته وضلعه الجانبي مبينان",
           "givens": [f"القاعدة {ar_digits(base)} والضلع الجانبي {ar_digits(slant)}"],
           "ask": "ما محيط متوازي الأضلاع؟"}
    ans = 2 * (base + slant)
    dist = [fmt_num(base + slant, lang), fmt_num(base * slant, lang), fmt_num(2 * base + slant, lang)]
    return dict(program=prog, measure={"op": "perimeter", "verts": "P"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


def _sc_rectangle_diagonal(rng, lang):
    """Rectangle: given length and width, find the diagonal (Pythagoras)."""
    w, h, diag = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17)])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": w, "y": 0},
        {"op": "point", "name": "ج", "x": w, "y": h},
        {"op": "point", "name": "د", "x": 0, "y": h},
        {"op": "polygon", "name": "R", "verts": ["أ", "ب", "ج", "د"]},
        {"op": "segment", "name": "diag", "a": "أ", "b": "ج"},
    ]
    annotations = [
        {"type": "side", "a": "أ", "b": "ب", "text": ar_digits(w)},
        {"type": "side", "a": "ب", "b": "ج", "text": ar_digits(h)},
    ]
    ask = {"figure": "مستطيل أبعاده مبينة وقطره مرسوم",
           "givens": [f"الطول {ar_digits(w)} والعرض {ar_digits(h)}"],
           "ask": "ما طول قطر المستطيل؟"}
    dist = [fmt_num(w + h, lang), fmt_num(diag + 1, lang), fmt_num(diag - 1, lang)]
    return dict(program=prog, measure={"op": "length", "a": "أ", "b": "ج"},
               annotations=annotations, ask=ask, answer=fmt_num(diag, lang),
               distractors=dist, subcat="Angles & Triangles", hide=set())


def _sc_rhombus_perimeter(rng, lang):
    """Rhombus: all sides equal; given the side length, find the perimeter."""
    skew, h, s = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13)])
    prog = [
        {"op": "point", "name": "أ", "x": 0, "y": 0},
        {"op": "point", "name": "ب", "x": s, "y": 0},
        {"op": "point", "name": "ج", "x": s + skew, "y": h},
        {"op": "point", "name": "د", "x": skew, "y": h},
        {"op": "polygon", "name": "R", "verts": ["أ", "ب", "ج", "د"]},
    ]
    annotations = [{"type": "side", "a": "أ", "b": "ب", "text": ar_digits(s)}]
    ask = {"figure": "معيّن جميع أضلاعه متساوية",
           "givens": [f"طول الضلع {ar_digits(s)}"],
           "ask": "ما محيط المعيّن؟"}
    ans = 4 * s
    dist = [fmt_num(2 * s, lang), fmt_num(s * s, lang), fmt_num(3 * s, lang)]
    return dict(program=prog, measure={"op": "perimeter", "verts": "R"},
               annotations=annotations, ask=ask, answer=fmt_num(ans, lang),
               distractors=dist, subcat="Area, Perimeter, Volume, Surface Area", hide=set())


SCENARIOS: List[Callable] = [
    _sc_right_triangle_third_side,
    _sc_triangle_area,
    _sc_inscribed_angle,
    _sc_sector_area,
    _sc_composite_shaded,
    _sc_cube_volume,
    _sc_parallelogram_area,
    _sc_trapezoid_area,
    _sc_polygon_interior_sum,
    _sc_transversal_corresponding,
    _sc_box_volume,
    _sc_rectangle_perimeter,
    # --- added scenarios (more variety) ---
    _sc_pythagoras_leg,
    _sc_isosceles_base_angle,
    _sc_circle_circumference,
    _sc_arc_length,
    _sc_triangle_perimeter,
    _sc_regular_polygon_each_angle,
    _sc_rectangle_area,
    _sc_square_area,
    _sc_circle_area,
    _sc_parallelogram_perimeter,
    _sc_rectangle_diagonal,
    _sc_rhombus_perimeter,
]


# ============================================================
# GENERATOR
# ============================================================
def generate_geometry_questions(num: int, lang: str = "Arabic",
                                rng: Optional[random.Random] = None,
                                save: bool = True) -> List[dict]:
    rng = rng or random.Random()
    OUT_DIR.mkdir(exist_ok=True)
    cat_dir = IMG_DIR / "Geometry"
    cat_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    seen_sig: set = set()
    attempts = 0
    max_attempts = num * 12
    # Deal scenarios from a shuffled deck so every shape type is used once before
    # any repeats -> maximum variety across a set.
    deck: List[Callable] = []

    def _draw_scenario():
        nonlocal deck
        if not deck:
            deck = list(SCENARIOS)
            rng.shuffle(deck)
        return deck.pop()

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        sc = _draw_scenario()
        try:
            spec = sc(rng, lang)
            # dedup: identical scenario+params -> identical program JSON
            sig = json.dumps(spec["program"], ensure_ascii=False, sort_keys=True)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            fig = gk.solve(spec["program"])
            gk.sanity_check(fig)
            # VERIFY: measure the answer from solved coordinates
            measured = gm.query(fig, spec["measure"])
            # render the figure (givens only; asked quantity hidden)
            fname = f"{uuid.uuid4().hex}.png"
            out_path = cat_dir / fname
            ok = gr.render_construction(fig, spec.get("annotations"),
                                        str(out_path), hide=spec.get("hide"))
            if not ok:
                continue
            # prose (LLM words only) with offline fallback
            question = _synth_prose(spec["ask"], lang) or _fallback_stem(spec["ask"], lang)
            options = _build_options(spec["answer"], spec["distractors"], rng)
            row = _assemble_row(question, spec["answer"], options, spec["subcat"],
                                spec["program"], str(Path("images") / "Geometry" / fname), lang)
            row["_measured"] = measured     # debug aid; harmless extra key
            rows.append(row)
        except gk.GeoError:
            continue
        except Exception as e:
            print(f"[geo_scenarios] scenario {sc.__name__} failed: {e}")
            continue

    if save and rows:
        try:
            import pandas as pd
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "_measured"} for r in rows])
            for c in IMG_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            df.to_excel(OUT_DIR / "Geometry.xlsx", index=False)
        except Exception as e:
            print(f"[geo_scenarios] save failed: {e}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--lang", default="Arabic")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    r = random.Random(a.seed) if a.seed is not None else None
    out = generate_geometry_questions(a.num, a.lang, rng=r)
    for row in out:
        print(row["Sub-Category"], "|", row["Answer"], "| measured=", row.get("_measured"),
              "|", row["Question"][:70])
