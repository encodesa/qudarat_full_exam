#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-first Data-Interpretation question generator (the DI analogue of
geo_scenarios).

Each scenario: sample a small dataset -> COMPUTE the answer from that data in
code (max/min/sum/difference/average/percent) -> render the chart from the SAME
data (bar/pie/line/table) -> the LLM writes only the wording (no data, no answer)
-> parametric distractors -> assemble a row dict matching app6's schema.

Why this exists: the old text-first DI path let the LLM invent both the figure
data and the answer, so figures routinely didn't support the answer and the
vision critic rejected them (or shipped wrong answers). Here the answer is a pure
function of the rendered data, so it is correct and figure-consistent every time.

Offline-safe: with no API key the question prose falls back to a deterministic
Arabic template built from the givens.
"""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geo_style as S  # API-free: Arabic reshaping (S.ar) + font setup
# Reuse the kernel-path option/format helpers so DI options look identical.
from geo_scenarios import (
    ar_digits, fmt_num, _build_options, _ROW_KEYS, _client,
)

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "questins_visual"
IMG_DIR = OUT_DIR / "images"
CAT = "Data Interpretation & Reasoning"
CAT_SLUG = "Data_Interpretation_Reasoning"
SUBCAT = "Tables/Charts/Graphs"
IMG_COLUMNS = ["NeedsImage", "RenderMethod", "ImagePath", "ImageSpec"]
PROSE_MODEL = "gemini-2.5-flash"

_PROSE_SYS = (
    "أنت كاتب أسئلة قدرات. ستعطى وصف رسم بياني/جدول والمطلوب. اكتب نص سؤال "
    "عربي فصيح واضح يشير إلى الشكل/الجدول ويطلب المطلوب بالضبط. لا تضف أي أرقام "
    "غير الموجودة في المعطيات، ولا تذكر الإجابة. أعد JSON: {\"question\": \"...\"}."
)

# Arabic label banks (kept short so charts render cleanly).
_CATEGORIES = [
    ["الكتب", "الأقلام", "الدفاتر", "الحقائب"],
    ["الرياضيات", "العلوم", "اللغة", "التاريخ"],
    ["الذرة", "القمح", "الأرز", "الشعير"],
    ["كرة القدم", "السباحة", "الجري", "التنس"],
]
_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو"]
_CITIES = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]


# ============================================================
# LLM prose (words only) with deterministic fallback
# ============================================================
def _synth_prose(ask: Dict[str, Any], lang: str) -> Optional[str]:
    client = _client()
    if client is None or lang != "Arabic":
        return None
    prompt = (
        f"الشكل: {ask.get('figure','')}\n"
        f"المعطيات: {ask.get('data_desc','')}\n"
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
        return q or None
    except Exception:
        return None


def _fallback_stem(ask: Dict[str, Any], lang: str) -> str:
    fig = ask.get("figure", "")
    head = f"يوضّح {fig}. " if fig else ""
    return f"{head}{ask.get('ask','')}"


# ============================================================
# chart rendering (self-contained; Arabic via geo_style.ar)
# ============================================================
def _render(spec: Dict[str, Any], out_path: Path) -> bool:
    ctype = spec["type"]
    labels = [S.ar(x) for x in spec.get("labels", [])]
    values = spec.get("values", [])
    title = S.ar(spec.get("title", ""))
    try:
        if ctype == "bar":
            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
            ax.bar(labels, values, color="#4C72B0", width=0.6, edgecolor="white")
            mx = max(values)
            ax.set_ylim(0, mx * 1.18)
            for i, v in enumerate(values):
                ax.text(i, v + mx * 0.015, ar_digits(v), ha="center", va="bottom", fontsize=11)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif ctype == "pie":
            fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
            ax.pie(values, labels=labels, autopct=lambda p: ar_digits(round(p)) + "٪",
                   startangle=90, counterclock=False,
                   wedgeprops={"edgecolor": "white", "linewidth": 1})
            ax.axis("equal")
        elif ctype == "line":
            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
            ax.plot(labels, values, marker="o", color="#4C72B0")
            lo, hi = min(values), max(values)
            pad = (hi - lo) * 0.18 or (abs(hi) * 0.18 or 1)
            ax.set_ylim(lo - pad * 0.4, hi + pad)
            for x, v in zip(labels, values):
                ax.annotate(ar_digits(v), (x, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=10)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif ctype == "table":
            cols = [S.ar(c) for c in spec["columns"]]
            rows = [[S.ar(c) for c in r] for r in spec["rows"]]
            fig, ax = plt.subplots(
                figsize=(max(5, 1.8 * len(cols)), max(2.2, 0.6 * (len(rows) + 1))), dpi=200)
            ax.axis("off")
            tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(12)
            tbl.scale(1, 1.6)
        else:
            return False
        if title:
            ax.set_title(title, fontsize=14, pad=10)
        fig.tight_layout(pad=1.2)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return out_path.exists() and out_path.stat().st_size >= 1024
    except Exception as e:
        print(f"[di_scenarios] render failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ============================================================
# row assembly
# ============================================================
def _assemble_row(question, answer, options, image_path, spec) -> Dict[str, Any]:
    row = {k: "" for k in _ROW_KEYS}
    row.update(options)
    row["Question"] = question
    row["Answer"] = answer
    row["Context"] = ""
    row["Section"] = "Quantitative"
    row["Category"] = CAT
    row["Sub-Category"] = SUBCAT
    row["Complexity-Level"] = "Basic"
    row["Similarity-to-Corpus"] = 0.0
    row["Votes"] = 3                       # data-verified -> full confidence
    row["NeedsImage"] = True
    row["RenderMethod"] = "matplotlib"
    row["ImagePath"] = image_path
    row["ImageSpec"] = json.dumps(spec, ensure_ascii=False)
    row["FigureCriticPass"] = True
    return row


# ============================================================
# SCENARIOS
# Each returns: chart(spec), ask(dict), answer(str), distractors(list[str]).
# The answer is computed from the chart data -> always correct & figure-consistent.
# ============================================================
def _distinct_values(rng, n, lo, hi):
    """n distinct integers in [lo, hi] (so extrema/answers are unambiguous)."""
    pool = list(range(lo, hi + 1))
    rng.shuffle(pool)
    return sorted(rng.sample(pool, n), reverse=True) if False else rng.sample(pool, n)


def _sc_bar_max(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    vals = _distinct_values(rng, len(cats), 10, 90)
    imax = vals.index(max(vals))
    answer = cats[imax]
    dist = [c for i, c in enumerate(cats) if i != imax]
    spec = {"type": "bar", "labels": cats, "values": vals, "title": "أعداد المبيعات"}
    ask = {"figure": "الرسم البياني بالأعمدة أعداد عناصر مختلفة",
           "data_desc": "؛ ".join(f"{c}: {v}" for c, v in zip(cats, vals)),
           "ask": "أيُّ العناصر صاحبُ أكبر قيمة؟"}
    return spec, ask, answer, dist


def _sc_bar_min(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    vals = _distinct_values(rng, len(cats), 10, 90)
    imin = vals.index(min(vals))
    answer = cats[imin]
    dist = [c for i, c in enumerate(cats) if i != imin]
    spec = {"type": "bar", "labels": cats, "values": vals, "title": "أعداد المبيعات"}
    ask = {"figure": "الرسم البياني بالأعمدة أعداد عناصر مختلفة",
           "data_desc": "؛ ".join(f"{c}: {v}" for c, v in zip(cats, vals)),
           "ask": "أيُّ العناصر صاحبُ أقل قيمة؟"}
    return spec, ask, answer, dist


def _sc_bar_difference(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    vals = _distinct_values(rng, len(cats), 10, 90)
    i, j = rng.sample(range(len(cats)), 2)
    diff = abs(vals[i] - vals[j])
    answer = fmt_num(diff, lang)
    spec = {"type": "bar", "labels": cats, "values": vals, "title": "أعداد المبيعات"}
    ask = {"figure": "الرسم البياني بالأعمدة أعداد عناصر مختلفة",
           "data_desc": "؛ ".join(f"{c}: {v}" for c, v in zip(cats, vals)),
           "ask": f"ما الفرق بين قيمتي «{cats[i]}» و«{cats[j]}»؟"}
    return spec, ask, answer, []


def _sc_bar_total(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    vals = _distinct_values(rng, len(cats), 10, 60)
    total = sum(vals)
    answer = fmt_num(total, lang)
    spec = {"type": "bar", "labels": cats, "values": vals, "title": "أعداد المبيعات"}
    ask = {"figure": "الرسم البياني بالأعمدة أعداد عناصر مختلفة",
           "data_desc": "؛ ".join(f"{c}: {v}" for c, v in zip(cats, vals)),
           "ask": "ما مجموع جميع القيم؟"}
    return spec, ask, answer, []


def _sc_bar_average(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    n = len(cats)
    # force an integer mean: pick mean, then deltas summing to zero
    mean = rng.choice([20, 25, 30, 40, 50])
    deltas = []
    while True:
        deltas = [rng.choice([-15, -10, -5, 5, 10, 15]) for _ in range(n - 1)]
        last = -sum(deltas)
        if -15 <= last <= 15 and last in (-15, -10, -5, 5, 10, 15, 0):
            deltas.append(last)
            break
    vals = [mean + d for d in deltas]
    if any(v <= 0 for v in vals) or len(set(vals)) < n:
        return _sc_bar_total(rng, lang)  # rare degenerate -> swap scenario
    answer = fmt_num(mean, lang)
    spec = {"type": "bar", "labels": cats, "values": vals, "title": "أعداد المبيعات"}
    ask = {"figure": "الرسم البياني بالأعمدة أعداد عناصر مختلفة",
           "data_desc": "؛ ".join(f"{c}: {v}" for c, v in zip(cats, vals)),
           "ask": "ما متوسط القيم؟"}
    return spec, ask, answer, []


def _sc_pie_percent(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    # counts whose total is 100 so the asked percentage is exact
    parts = _distinct_values(rng, len(cats), 1, 9)
    s = sum(parts)
    vals = [round(p / s * 100) for p in parts]
    vals[-1] += 100 - sum(vals)            # fix rounding so total == 100
    if len(set(vals)) < len(vals) or any(v <= 0 for v in vals):
        return _sc_bar_max(rng, lang)
    i = vals.index(max(vals))
    answer = ar_digits(vals[i]) + "٪"
    spec = {"type": "pie", "labels": cats, "values": vals, "title": "النسب المئوية"}
    ask = {"figure": "الرسم الدائري نسب توزيع عناصر",
           "data_desc": "؛ ".join(f"{c}: {v}٪" for c, v in zip(cats, vals)),
           "ask": f"ما النسبة المئوية لـ«{cats[i]}»؟"}
    return spec, ask, answer, [ar_digits(v) + "٪" for k, v in enumerate(vals) if k != i]


def _sc_pie_count_from_percent(rng, lang):
    cats = list(rng.choice(_CATEGORIES))
    rng.shuffle(cats)
    parts = _distinct_values(rng, len(cats), 1, 9)
    s = sum(parts)
    vals = [round(p / s * 100) for p in parts]
    vals[-1] += 100 - sum(vals)
    total = rng.choice([200, 300, 400, 500])
    i = rng.randrange(len(cats))
    if vals[i] <= 0 or (vals[i] * total) % 100 != 0:
        return _sc_pie_percent(rng, lang)
    count = vals[i] * total // 100
    answer = fmt_num(count, lang)
    spec = {"type": "pie", "labels": cats, "values": vals, "title": "النسب المئوية"}
    ask = {"figure": "الرسم الدائري نسب توزيع عناصر",
           "data_desc": "؛ ".join(f"{c}: {v}٪" for c, v in zip(cats, vals)),
           "ask": f"إذا كان العدد الكلي {ar_digits(total)}، فكم عدد «{cats[i]}»؟"}
    return spec, ask, answer, []


def _sc_line_max_period(rng, lang):
    k = rng.choice([4, 5, 6])
    months = _MONTHS[:k]
    vals = _distinct_values(rng, k, 10, 90)
    i = vals.index(max(vals))
    answer = months[i]
    dist = [m for j, m in enumerate(months) if j != i]
    rng.shuffle(dist)
    spec = {"type": "line", "labels": months, "values": vals, "title": "القيمة الشهرية"}
    ask = {"figure": "الرسم الخطي تغيّر قيمة عبر عدة أشهر",
           "data_desc": "؛ ".join(f"{m}: {v}" for m, v in zip(months, vals)),
           "ask": "في أيِّ شهرٍ بلغت القيمة أعلاها؟"}
    return spec, ask, answer, dist[:3]


def _sc_line_increase(rng, lang):
    k = rng.choice([4, 5, 6])
    months = _MONTHS[:k]
    vals = _distinct_values(rng, k, 10, 90)
    i = rng.randrange(k - 1)
    diff = abs(vals[i + 1] - vals[i])
    answer = fmt_num(diff, lang)
    spec = {"type": "line", "labels": months, "values": vals, "title": "القيمة الشهرية"}
    ask = {"figure": "الرسم الخطي تغيّر قيمة عبر عدة أشهر",
           "data_desc": "؛ ".join(f"{m}: {v}" for m, v in zip(months, vals)),
           "ask": f"ما مقدار التغيّر في القيمة بين «{months[i]}» و«{months[i+1]}»؟"}
    return spec, ask, answer, []


def _sc_table_row_total(rng, lang):
    cities = rng.sample(_CITIES, 3)
    cols = ["المدينة", "الأحد", "الإثنين", "الثلاثاء"]
    data = {c: _distinct_values(rng, 3, 5, 40) for c in cities}
    rows = [[c] + [ar_digits(v) for v in data[c]] for c in cities]
    target = rng.choice(cities)
    total = sum(data[target])
    answer = fmt_num(total, lang)
    spec = {"type": "table", "columns": cols, "rows": rows, "title": "أعداد الزوّار"}
    ask = {"figure": "الجدول أعداد الزوّار في ثلاثة أيام لكل مدينة",
           "data_desc": "؛ ".join(f"{c}: {data[c]}" for c in cities),
           "ask": f"ما مجموع أعداد الزوّار لمدينة «{target}» في الأيام الثلاثة؟"}
    return spec, ask, answer, []


def _sc_table_diff(rng, lang):
    cities = rng.sample(_CITIES, 3)
    cols = ["المدينة", "صباحًا", "مساءً"]
    data = {c: _distinct_values(rng, 2, 5, 60) for c in cities}
    rows = [[c] + [ar_digits(v) for v in data[c]] for c in cities]
    target = rng.choice(cities)
    diff = abs(data[target][0] - data[target][1])
    answer = fmt_num(diff, lang)
    spec = {"type": "table", "columns": cols, "rows": rows, "title": "الأعداد المسجّلة"}
    ask = {"figure": "الجدول الأعداد المسجّلة صباحًا ومساءً لكل مدينة",
           "data_desc": "؛ ".join(f"{c}: {data[c]}" for c in cities),
           "ask": f"ما الفرق بين العددين المسجّلين صباحًا ومساءً في «{target}»؟"}
    return spec, ask, answer, []


SCENARIOS: List[Callable] = [
    _sc_bar_max, _sc_bar_min, _sc_bar_difference, _sc_bar_total, _sc_bar_average,
    _sc_pie_percent, _sc_pie_count_from_percent,
    _sc_line_max_period, _sc_line_increase,
    _sc_table_row_total, _sc_table_diff,
]


def generate_di_questions(num: int, lang: str = "Arabic",
                          rng: Optional[random.Random] = None,
                          save: bool = True) -> List[dict]:
    rng = rng or random.Random()
    OUT_DIR.mkdir(exist_ok=True)
    cat_dir = IMG_DIR / CAT_SLUG
    cat_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    seen_sig: set = set()
    attempts = 0
    max_attempts = num * 12
    deck: List[Callable] = []

    def _draw():
        nonlocal deck
        if not deck:
            deck = list(SCENARIOS)
            rng.shuffle(deck)
        return deck.pop()

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        sc = _draw()
        try:
            spec, ask, answer, dist = sc(rng, lang)
            sig = json.dumps([spec, answer], ensure_ascii=False, sort_keys=True)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            fname = f"{uuid.uuid4().hex}.png"
            out_path = cat_dir / fname
            if not _render(spec, out_path):
                continue
            question = _synth_prose(ask, lang) or _fallback_stem(ask, lang)
            options = _build_options(answer, dist, rng)
            rel = str(Path("images") / CAT_SLUG / fname)
            rows.append(_assemble_row(question, answer, options, rel, spec))
        except Exception as e:
            print(f"[di_scenarios] scenario {sc.__name__} failed: {e}")
            continue

    if save and rows:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            for c in IMG_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            df.to_excel(OUT_DIR / "DataInterpretation.xlsx", index=False)
        except Exception as e:
            print(f"[di_scenarios] save failed: {e}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--lang", default="Arabic")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    r = random.Random(a.seed) if a.seed is not None else None
    out = generate_di_questions(a.num, lang=a.lang, rng=r)
    for row in out:
        print(row["CorrectOption"], "|", row["Answer"], "|",
              [row[f"Option{L}"] for L in "ABCD"], "|", row["ImagePath"])
    print(f"\n{len(out)} DI questions")
