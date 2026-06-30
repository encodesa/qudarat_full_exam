"""Generate a set of 15 math (Quantitative) questions that ALL include a figure.

Thin driver over the existing visual pipeline in generate_visual.py. It calls
generate_questions() across the two visual-heavy Quantitative categories:
  - Geometry                              -> shapes / figures (Gemini render)
  - Data Interpretation & Reasoning /
    Tables/Charts/Graphs                  -> tables / charts (matplotlib render)

The generator sometimes produces text-only questions the classifier marks
needs_image=False. To guarantee every question has a figure, this driver
over-generates and keeps ONLY rows that actually rendered an image
(ImagePath set), looping per category until the target count is reached.

Usage:
    python generate_math_set.py
    python generate_math_set.py --num-geometry 8 --num-data 7 --lang Arabic
"""

import argparse

import pandas as pd

import generate_visual as gv

# Speed up generation with a working key, mirroring app_visual.py.
gv.gen._MAX_CONCURRENT_API_CALLS = 12

MERGED_XLSX = gv.OUT_DIR / "math_set_15.xlsx"


def _has_figure(row: dict) -> bool:
    """A row counts only if a figure was actually rendered to disk."""
    return bool(row.get("NeedsImage")) and bool(str(row.get("ImagePath") or "").strip())


def collect_visual(section, category, subcat, lang, target, max_rounds=6):
    """Generate until `target` rows with a real figure are collected."""
    kept = []
    seen_q = set()
    for rnd in range(1, max_rounds + 1):
        if len(kept) >= target:
            break
        need = target - len(kept)
        # Over-provision: ask for ~2x what's missing (min 4) to absorb misses.
        ask = max(need * 2, 4)
        gv.gen.log(f"[{category}] round {rnd}: have {len(kept)}/{target}, "
                   f"requesting {ask}")
        rows = gv.generate_questions(section, category, subcat, lang, ask, save=False)
        for r in rows:
            q = (r.get("Question") or "").strip()
            if q in seen_q:
                continue
            seen_q.add(q)
            if _has_figure(r):
                kept.append(r)
                if len(kept) >= target:
                    break
    if len(kept) < target:
        gv.gen.log(f"[{category}] only got {len(kept)}/{target} with figures after "
                   f"{max_rounds} rounds", level="warning")
    return kept[:target]


def main():
    ap = argparse.ArgumentParser(
        description="Generate 15 math questions that all include figures")
    ap.add_argument("--num-geometry", type=int, default=8,
                    help="Geometry questions (shapes/figures)")
    ap.add_argument("--num-data", type=int, default=7,
                    help="Data Interpretation questions (tables/charts)")
    ap.add_argument("--lang", default="Arabic")
    args = ap.parse_args()

    gv.OUT_DIR.mkdir(exist_ok=True)
    gv.IMG_DIR.mkdir(exist_ok=True)

    plan = [
        ("Quantitative", "Geometry", "ANY", args.num_geometry),
        ("Quantitative", "Data Interpretation & Reasoning",
         "Tables/Charts/Graphs", args.num_data),
    ]

    all_rows = []
    for section, category, subcat, num in plan:
        if num <= 0:
            continue
        gv.gen.log(f"=== {category}: target {num} figure questions ({args.lang}) ===")
        rows = collect_visual(section, category, subcat, args.lang, num)
        gv.gen.log(f"--- {category}: kept {len(rows)} with figures ---")
        all_rows.extend(rows)

    if not all_rows:
        gv.gen.log("No questions with figures generated.", level="error")
        return

    df = pd.DataFrame(all_rows)
    for c in gv.IMG_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df.to_excel(MERGED_XLSX, index=False)

    n_img = int(df["NeedsImage"].astype(bool).sum())
    print(f"\nTotal questions: {len(df)}  |  with figures: {n_img}")
    print(f"Merged workbook: {MERGED_XLSX}")
    print(f"Images dir:      {gv.IMG_DIR}")


if __name__ == "__main__":
    main()
