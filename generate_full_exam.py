"""Generate a full-scale Qudrat exam from an authored blueprint and assemble it
into sections like the real paper.

Pipeline reuse (no changes to the core):
  exam_blueprint.load_blueprint(sheet) -> generation tasks
  generate_visual.generate_questions(section, category, subcat, lang, n)
      -> LLM question + distractors + figure stage
         (geo kernel for shapes, matplotlib for tables/charts, Gemini fallback)

The generated pool is then dealt round-robin across N sections so every section
gets the SAME proportional mix of categories/sub-categories (like the real exam),
quant/verbal interleaved within each section. Output:
  questins_visual/full_exam.xlsx   - flat workbook (all IMG_COLUMNS)
  questins_visual/full_exam.json   - manifest the web interface reads

Usage:
  python generate_full_exam.py
  python generate_full_exam.py --sheet Khaled-Sample_1_Arabic --sections 5 --per-section 24
  python generate_full_exam.py --per-section 2 --sections 2   # quick smoke test
"""

import argparse
import json
import os
import threading
from pathlib import Path

import pandas as pd

import generate_visual as gv
import exam_blueprint as bp

# Speed-up: the app6 semaphore is built ONCE at import with value 3, so merely
# setting _MAX_CONCURRENT_API_CALLS has no effect. Rebuild the semaphore so calls
# actually run 12-wide (~4x throughput, no quality change).
# API call cap (shared across all parallel tasks via the app6 semaphore).
# Env-overridable: on a paid tier with high rate limits, raise GEN_CONCURRENCY to
# cut wall-clock for the full 96-question run; the per-call 429 backoff + timeout
# retry in gemini_call keeps a too-high value safe (it just backs off). Default 40
# suits a Tier-3 key (Flash 20K RPM, Pro 2K RPM); push to 60-100 to go faster.
# Real generation calls take ~10-15s, so the call timeout must stay well above that.
_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "40"))
gv.gen._MAX_CONCURRENT_API_CALLS = _CONCURRENCY
gv.gen._api_semaphore = threading.Semaphore(_CONCURRENCY)

# Run tasks themselves in parallel. The semaphore above is the real throughput cap;
# parallel tasks keep it saturated during each task's serial gaps (the one generation
# call, image rendering) instead of leaving slots idle. Tasks are independent.
TASK_PARALLELISM = int(os.getenv("GEN_TASK_PARALLELISM", "5"))

# --- Global tuning, set ONCE (no per-task mutation) so parallel tasks don't race ---
# Generator stays on flash for speed (batch-gen on pro can take minutes;
# gemini-2.5-flash hung under load — gemini-flash-latest is fast and reliable).
#
# Quality levers that the standalone app6 path had and the earlier batch config
# dropped, now restored because they are CHEAP relative to generation:
#   * 3 solvers (parallel, varied temperature) -> real 2-of-3 majority answer check,
#     instead of a single solver that rubber-stamps whatever was generated.
#   * A STRONGER critic/vision model (runs once per batch / once per figure, not per
#     question) so format/notation leaks and bad figures are actually rejected. These
#     are deterministically backstopped by the homogeneity + figure gates, so even if
#     the model is downgraded for reliability, quality no longer depends on it.
# All three models are env-overridable if the stronger tier proves unreliable.
_MODEL = os.getenv("GEN_MODEL", "gemini-flash-latest")
_REVIEW_MODEL = os.getenv("REVIEW_MODEL", "gemini-2.5-pro")
gv.gen.GENERATOR_MODEL = _MODEL
gv.gen.SOLVER_MODELS = [
    {"name": _MODEL, "temperature": 0.0},
    {"name": _MODEL, "temperature": 0.3},
    {"name": _MODEL, "temperature": 0.0},
]
gv.gen.CRITIC_MODEL = _REVIEW_MODEL
gv.VISION_CRITIC_MODEL = _REVIEW_MODEL
gv.gen.GEN_OVERSAMPLE_FACTOR = 1.5
FIGURE_MAX_ROUNDS = 3

# Serialize checkpoint appends across task threads.
_CKPT_LOCK = threading.Lock()

# Skip the rate-limited Gemini image model. Tables/charts -> matplotlib (hardened),
# shapes -> geometry kernel. Removes the 429-backoff stalls that dominated runtime.
gv.DISABLE_GEMINI_IMAGE = True

OUT_DIR = gv.OUT_DIR
FULL_XLSX = OUT_DIR / "full_exam.xlsx"
FULL_JSON = OUT_DIR / "full_exam.json"
# Append-as-you-go checkpoint so a crash mid-run never loses completed tasks.
CKPT_JSONL = OUT_DIR / "full_exam_checkpoint.jsonl"


def _task_key(t: "bp.GenTask") -> str:
    return f"{t.section}|{t.category}|{t.subcat}|{t.ar_subcat}"


def _ckpt_load() -> dict:
    """Return {task_key: [rows]} from a prior run's checkpoint, if any."""
    done: dict = {}
    if not CKPT_JSONL.exists():
        return done
    for line in CKPT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.setdefault(r.get("_task_key", ""), []).append(r)
    return done


def _ckpt_append(key: str, rows: list):
    """Append a completed task's rows to the checkpoint immediately (thread-safe)."""
    with _CKPT_LOCK:
        with CKPT_JSONL.open("a", encoding="utf-8") as f:
            for r in rows:
                r2 = dict(r)
                r2["_task_key"] = key
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            f.flush()

SECTION_TITLES_AR = [
    "القسم الأول", "القسم الثاني", "القسم الثالث",
    "القسم الرابع", "القسم الخامس", "القسم السادس",
]
FIGURE_CATEGORIES = {"Geometry", "Data Interpretation & Reasoning"}


def _has_figure(row: dict) -> bool:
    """A row counts as figure-bearing only if a figure rendered to disk AND the
    vision critic judged it to actually support the question/answer."""
    return (bool(row.get("NeedsImage"))
            and bool(str(row.get("ImagePath") or "").strip())
            and bool(row.get("FigureCriticPass", True)))


def _generate_task(task: bp.GenTask, scale: float):
    """Generate questions for one blueprint task (scaled to target exam size).

    For figure categories, over-generate and keep only rows that actually
    rendered a figure, looping until the target is met (mirrors
    generate_math_set.collect_visual).
    """
    target = max(1, round(task.count * scale))
    needs_fig = task.category in FIGURE_CATEGORIES
    subcat = task.subcat if task.subcat != "ANY" else "ANY"

    def _gen(sc):
        """Generate, retrying at category level if the subcat has no corpus
        examples (some authored subcats live under a different parent category)."""
        try:
            return gv.generate_questions(task.section, task.category, sc,
                                         task.lang, target, save=False)
        except RuntimeError as e:
            if sc != "ANY" and "No example questions" in str(e):
                gv.gen.log(f"[{task.category}] subcat '{sc}' has no examples; "
                           f"falling back to category level", level="warning")
                return gv.generate_questions(task.section, task.category, "ANY",
                                             task.lang, target, save=False)
            raise

    if not needs_fig:
        rows = _gen(subcat)
        return rows[:target]

    # Solver count / oversample are set globally (no per-task mutation) so tasks can
    # run in parallel without racing on shared state.
    return _generate_figure_task(task, target, subcat)


def _generate_figure_task(task, target, subcat):
    kept, seen = [], set()
    feedback = ""  # why the previous round's figures failed — fed into next round
    for rnd in range(1, FIGURE_MAX_ROUNDS + 1):
        if len(kept) >= target:
            break
        ask = max(round((target - len(kept)) * 1.25), target - len(kept) + 1, 3)
        gv.gen.log(f"[{task.category}/{task.ar_subcat}] round {rnd}: "
                   f"{len(kept)}/{target}, requesting {ask}"
                   + (f" [feedback: {feedback[:120]}]" if feedback else ""))
        rows = gv.generate_questions(task.section, task.category, subcat,
                                     task.lang, ask, save=False,
                                     gen_feedback=feedback)
        fails = []
        for r in rows:
            q = (r.get("Question") or "").strip()
            if q in seen:
                continue
            seen.add(q)
            if _has_figure(r):
                kept.append(r)
                if len(kept) >= target:
                    break
            else:
                reason = str(r.get("FigureFailReason") or "").strip()
                if reason:
                    fails.append(reason)
        # Distill this round's failure reasons into actionable guidance for the next.
        feedback = _summarize_figure_fails(fails)
    # Final guard: never ship a figure question whose PNG is missing/empty on disk.
    verified = [r for r in kept if _figure_png_ok(r)]
    dropped = len(kept) - len(verified)
    if dropped:
        gv.gen.log(f"[{task.category}] dropped {dropped} question(s) with missing/empty "
                   f"figure files on disk", level="warning")
    if len(verified) < target:
        gv.gen.log(f"[{task.category}] only {len(verified)}/{target} with valid figures",
                   level="warning")
    return verified[:target]


def _figure_png_ok(row: dict) -> bool:
    """The row's ImagePath must resolve to an existing, non-trivial PNG on disk."""
    rel = str(row.get("ImagePath") or "").strip()
    if not rel:
        return False
    p = OUT_DIR / rel
    try:
        return p.exists() and p.stat().st_size >= 1024
    except OSError:
        return False


def _summarize_figure_fails(reasons: list) -> str:
    """Turn raw per-row figure failures into a short 'do better' instruction."""
    if not reasons:
        return ""
    from collections import Counter
    tally = Counter()
    for r in reasons:
        low = r.lower()
        if "not needing" in low or "classifier" in low:
            tally["needs_figure"] += 1
        elif "render failed" in low:
            tally["render"] += 1
        elif "mismatch" in low:
            tally["mismatch"] += 1
        else:
            tally["other"] += 1
    parts = []
    if tally["needs_figure"]:
        parts.append("Write questions that REQUIRE reading a chart/table/figure to answer "
                     "(reference 'the figure/table/graph' explicitly and depend on its data).")
    if tally["render"]:
        parts.append("Provide complete, self-consistent figure data (all values/labels needed; "
                     "valid numeric values) so the figure renders.")
    if tally["mismatch"]:
        sample = next((r for r in reasons if "mismatch" in r.lower()), "")
        parts.append("Ensure the figure's data exactly supports the correct answer and does "
                     f"not reveal or contradict it. Prior issue: {sample[:140]}")
    return " ".join(parts)


def _deal_balanced(groups, n_sections, per_section):
    """Round-robin deal each group's rows across sections for an even mix, then
    interleave within each section so quant/verbal alternate. Returns a list of
    `n_sections` lists, each capped at `per_section`."""
    sections = [[] for _ in range(n_sections)]
    # Deal each (category/subcat) group round-robin, rotating the start section
    # so small groups don't all pile onto section 0.
    start = 0
    for rows in groups:
        for j, r in enumerate(rows):
            sections[(start + j) % n_sections].append(r)
        start = (start + len(rows)) % n_sections

    # Interleave within each section: alternate Quantitative / Verbal.
    out = []
    for sec in sections:
        quant = [r for r in sec if r.get("Section") == "Quantitative"]
        verbal = [r for r in sec if r.get("Section") != "Quantitative"]
        merged, i, j = [], 0, 0
        while i < len(quant) or j < len(verbal):
            if i < len(quant):
                merged.append(quant[i]); i += 1
            if j < len(verbal):
                merged.append(verbal[j]); j += 1
        out.append(merged[:per_section])
    return out


EXPLAIN_MODEL = os.getenv("EXPLAIN_MODEL", "gemini-2.5-flash")


def _explanation_for(r: dict) -> str:
    """LLM-written short Arabic explanation of WHY the correct answer is correct."""
    opts = "\n".join(f"{L}: {r.get('Option'+L,'')}" for L in ("A", "B", "C", "D"))
    ctx = str(r.get("Context") or "").strip()
    prompt = (
        (f"النص: {ctx}\n" if ctx else "")
        + f"السؤال: {r.get('Question','')}\n"
        f"الخيارات:\n{opts}\n"
        f"الإجابة الصحيحة هي الخيار {r.get('CorrectOption','')}: {r.get('Answer','')}\n\n"
        "اكتب شرحًا موجزًا بالعربية الفصحى (جملتان إلى ثلاث) يوضّح لماذا هذه الإجابة هي "
        "الصحيحة، مع خطوات الحل إن كان السؤال حسابيًا. لا تُشِر إلى الخيارات الأخرى بالحرف. "
        'أعد JSON فقط: {"explanation":"..."}'
    )
    try:
        raw = gv.gen.gemini_call(prompt, model=EXPLAIN_MODEL,
                                 response_mime_type="application/json", temperature=0.2)
        data = gv.gen.safe_json_load(raw) or {}
        exp = str(data.get("explanation", "")).strip()
        if r.get("Language", "Arabic") == "Arabic" or True:
            exp = gv.gen.to_arabic_digits(exp)
        return exp
    except Exception as e:
        gv.gen.log(f"[explain] failed: {e}", level="warning")
        return ""


def _add_explanations(rows: list):
    """Attach an 'Explanation' to every row (concurrent). Skips rows already done
    so a resume doesn't re-pay for cached questions."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [r for r in rows if not str(r.get("Explanation") or "").strip()]
    if not todo:
        return
    gv.gen.log(f"[explain] generating explanations for {len(todo)} question(s)")
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        for r, exp in zip(todo, ex.map(_explanation_for, todo)):
            r["Explanation"] = exp


def _manifest(sections, time_min):
    data = {"n_sections": len(sections), "sections": []}
    qno = 0
    for idx, sec in enumerate(sections):
        questions = []
        for r in sec:
            qno += 1
            img = str(r.get("ImagePath") or "").strip()
            questions.append({
                "qno": qno,
                "question": str(r.get("Question") or "").strip(),
                "context": str(r.get("Context") or "").strip(),
                "options": {L: str(r.get(f"Option{L}") or "").strip()
                            for L in ("A", "B", "C", "D")},
                "correct": str(r.get("CorrectOption") or "").strip().upper(),
                "answer": str(r.get("Answer") or "").strip(),
                "image_path": img,
                "explanation": str(r.get("Explanation") or "").strip(),
                "section_name": r.get("Section", ""),
                "category": r.get("Category", ""),
                "subcat": r.get("Sub-Category", ""),
            })
        data["sections"].append({
            "index": idx + 1,
            "title": SECTION_TITLES_AR[idx] if idx < len(SECTION_TITLES_AR)
                     else f"القسم {idx + 1}",
            "time_min": time_min,
            "count": len(questions),
            "questions": questions,
        })
    return data


def main():
    ap = argparse.ArgumentParser(description="Generate a full sectioned exam")
    ap.add_argument("--sheet", default=bp.DEFAULT_SHEET)
    ap.add_argument("--lang", default="Arabic")
    ap.add_argument("--sections", type=int, default=4)
    ap.add_argument("--per-section", type=int, default=24)
    ap.add_argument("--time-min", type=int, default=26)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore any existing checkpoint and regenerate every task")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the checkpoint before starting")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    gv.IMG_DIR.mkdir(exist_ok=True)

    if args.fresh and CKPT_JSONL.exists():
        CKPT_JSONL.unlink()
    done = _ckpt_load() if args.resume else {}
    if done:
        gv.gen.log(f"=== Resuming: {len(done)} task(s) already in checkpoint "
                   f"{CKPT_JSONL.name} ===")

    tasks = bp.load_blueprint(args.sheet, args.lang)
    blueprint_total = sum(t.count for t in tasks)
    target_total = args.sections * args.per_section
    scale = target_total / blueprint_total
    gv.gen.log(f"=== Full exam: {target_total} questions "
               f"({args.sections}x{args.per_section}), scale {scale:.2f} ===")

    # Pre-warm the shared corpus cache once so parallel task threads don't all
    # trigger (and race) the lazy load on first access.
    try:
        gv._load_data()
    except Exception as e:
        gv.gen.log(f"[warn] corpus pre-warm failed: {e}", level="warning")

    def _run_task(t):
        """Generate (or resume) one task. Returns its rows; checkpoints new ones."""
        key = _task_key(t)
        if key in done:
            rows = done[key]
            gv.gen.log(f"--- {t.category}/{t.ar_subcat}: resumed {len(rows)} "
                       f"from checkpoint ---")
            return rows
        try:
            rows = _generate_task(t, scale)
        except Exception as e:
            gv.gen.log(f"[{t.category}/{t.ar_subcat}] task failed, skipping: {e}",
                       level="error")
            rows = []
        for r in rows:               # tag section for interleave/manifest
            r.setdefault("Section", t.section)
        _ckpt_append(key, rows)      # persist immediately — survive a crash
        gv.gen.log(f"--- {t.category}/{t.ar_subcat}: got {len(rows)} (checkpointed) ---")
        return rows

    # Run tasks in parallel; preserve task order in `groups` for deterministic dealing.
    results = [None] * len(tasks)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=TASK_PARALLELISM) as ex:
        fut_to_idx = {ex.submit(_run_task, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()

    groups = []
    for rows in results:
        if not rows:
            continue
        for r in rows:               # drop internal checkpoint tag before assembly
            r.pop("_task_key", None)
        groups.append(rows)

    if not groups:
        gv.gen.log("No questions generated.", level="error")
        return

    sections = _deal_balanced(groups, args.sections, args.per_section)

    # Flat workbook for inspection / PDF.
    flat = [r for sec in sections for r in sec]

    # Per-question solution explanations (shown in the end-of-exam review screen).
    _add_explanations(flat)
    df = pd.DataFrame(flat)
    for c in gv.IMG_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df.to_excel(FULL_XLSX, index=False)

    manifest = _manifest(sections, args.time_min)
    FULL_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    n_img = int(df["NeedsImage"].astype(bool).sum()) if "NeedsImage" in df else 0
    print(f"\nSections: {len(sections)}  |  questions: {len(flat)}  |  "
          f"with figures: {n_img}")
    for s in manifest["sections"]:
        print(f"  {s['title']}: {s['count']} questions")
    print(f"Workbook: {FULL_XLSX}")
    print(f"Manifest: {FULL_JSON}")


if __name__ == "__main__":
    main()
