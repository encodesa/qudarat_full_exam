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
# Extra generation rounds allowed to replace cross-section duplicates.
DEDUP_MAX_ROUNDS = int(os.getenv("GEN_DEDUP_ROUNDS", "3"))

# Push the generator toward genuinely exam-level difficulty. Injected into every
# generation call's feedback channel so questions aren't trivial one-steppers.
DIFFICULTY_DIRECTIVE = (
    "Aim for real Qudrat exam difficulty, NOT trivial one-step questions. Vary the "
    "difficulty: at least a third should be genuinely challenging — requiring 2-3 "
    "reasoning steps, combining concepts, larger or non-round numbers, or a less "
    "obvious approach. Avoid questions a student can answer in one glance."
)

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


# ---------------------------------------------------------------------------
# Cross-section duplicate registry
# ---------------------------------------------------------------------------
# The per-call `seen_questions` inside generate_questions only dedupes within a
# single task. Two tasks mapping to the same category could still emit identical
# questions that land in different sections (reviewer note: "Exact questions
# repeated in two different sections", and the same question with its options
# reversed). This registry is shared across the WHOLE run and keys on semantic
# content (order-independent), so duplicates are caught no matter the option
# order or which task produced them.
import re as _re
import threading as _threading


_AR_DIACRITICS = _re.compile(r"[ً-ْٰ]")
_AR_TO_EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm_text(s: str) -> str:
    """Normalize Arabic/EN text for content comparison: strip diacritics, unify
    digits, collapse whitespace, drop punctuation, lowercase."""
    s = (s or "").translate(_AR_TO_EN_DIGITS)
    s = _AR_DIACRITICS.sub("", s)
    s = _re.sub(r"[^\w؀-ۿ]+", " ", s)  # keep letters/digits/Arabic
    return _re.sub(r"\s+", " ", s).strip().lower()


def _is_analogy(row: dict) -> bool:
    blob = f"{row.get('Category','')} {row.get('Sub-Category','')}".lower()
    return "analog" in blob or "تناظر" in blob


def _analogy_pairs(row: dict):
    """Return the set of normalized term-pairs (frozensets) a verbal-analogy row
    involves: the stem pair plus the correct-answer pair. Two analogy questions
    that reuse the same relationship pair (even reversed across stem/answer, as in
    the reviewer's طبيب:علاج / معلم:تدريس case) will share a pair."""
    pairs = set()
    stem = row.get("Question", "")
    correct = row.get("CorrectOption", "").strip().upper()
    ans = row.get(f"Option{correct}", "") or row.get("Answer", "")
    for text in (stem, ans):
        parts = _re.split(r"[:：]", text)  # split on the raw colon FIRST
        if len(parts) == 2:
            a, b = _norm_text(parts[0]), _norm_text(parts[1])
            if a and b:
                pairs.add(frozenset((a, b)))
    return pairs


def _content_key(row: dict) -> str:
    """Order-independent semantic key: normalized stem + sorted option set."""
    stem = _norm_text(row.get("Question", ""))
    ctx = _norm_text(row.get("Context", ""))
    opts = sorted(_norm_text(row.get(f"Option{L}", "")) for L in ("A", "B", "C", "D"))
    return "||".join([ctx, stem] + opts)


class SeenRegistry:
    """Thread-safe registry of already-accepted questions for one exam run."""

    def __init__(self):
        self._lock = _threading.Lock()
        self._content = set()
        self._analogy_pairs = set()

    def add_if_new(self, row: dict) -> bool:
        """Register `row` if it is not a duplicate of anything seen so far.
        Returns True if accepted (and recorded), False if it is a duplicate."""
        ckey = _content_key(row)
        apairs = _analogy_pairs(row) if _is_analogy(row) else set()
        with self._lock:
            if ckey in self._content:
                return False
            if apairs and (apairs & self._analogy_pairs):
                return False
            self._content.add(ckey)
            self._analogy_pairs |= apairs
            return True


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


def _generate_task(task: bp.GenTask, scale: float, target_override: int = None):
    """Generate questions for one blueprint task (scaled to target exam size).

    For figure categories, over-generate and keep only rows that actually
    rendered a figure, looping until the target is met (mirrors
    generate_math_set.collect_visual).

    target_override lets the dedup top-up loop request a specific batch size
    (e.g. only the questions still needed after cross-section duplicates were
    dropped) instead of the full scaled target.
    """
    target = target_override if target_override is not None else max(1, round(task.count * scale))
    needs_fig = task.category in FIGURE_CATEGORIES
    subcat = task.subcat if task.subcat != "ANY" else "ANY"

    def _gen(sc):
        """Generate, retrying at category level if the subcat has no corpus
        examples (some authored subcats live under a different parent category)."""
        try:
            return gv.generate_questions(task.section, task.category, sc,
                                         task.lang, target, save=False,
                                         gen_feedback=DIFFICULTY_DIRECTIVE)
        except RuntimeError as e:
            if sc != "ANY" and "No example questions" in str(e):
                gv.gen.log(f"[{task.category}] subcat '{sc}' has no examples; "
                           f"falling back to category level", level="warning")
                return gv.generate_questions(task.section, task.category, "ANY",
                                             task.lang, target, save=False,
                                             gen_feedback=DIFFICULTY_DIRECTIVE)
            raise

    if not needs_fig:
        rows = _gen(subcat)
        return rows[:target]

    # Solver count / oversample are set globally (no per-task mutation) so tasks can
    # run in parallel without racing on shared state.
    return _generate_figure_task(task, target, subcat)


def _generate_figure_task(task, target, subcat):
    kept, seen = [], set()
    feedback = DIFFICULTY_DIRECTIVE  # difficulty push + why prior figures failed
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
        # Distill this round's failure reasons into actionable guidance for the next,
        # keeping the difficulty push in front of it.
        feedback = (DIFFICULTY_DIRECTIVE + " " + _summarize_figure_fails(fails)).strip()
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


# ---------------------------------------------------------------------------
# Question-type grouping + difficulty ordering (real-paper structure)
# ---------------------------------------------------------------------------
# Real Qudrat groups a section by question TYPE, each with its own instruction
# box, and the difficulty steps up as you progress. We reproduce both: within a
# section, questions are grouped contiguously by type (canonical order below),
# and each type group is ordered easy -> hard by a deterministic score.
def _qtype(row: dict) -> str:
    cat = str(row.get("Category", ""))
    if row.get("Section") == "Quantitative":
        return "quant_mcq"
    if "Analog" in cat or "تناظر" in cat:
        return "analogy"
    if "Sentence Completion" in cat or "إكمال" in cat:
        return "sentence"
    if "Reading" in cat or "استيعاب" in cat:
        return "reading"
    return "verbal_other"

QTYPE_ORDER = ["quant_mcq", "analogy", "sentence", "reading", "verbal_other"]
QTYPE_TITLE = {
    "quant_mcq": "الاختيار من متعدد",
    "analogy": "التناظر اللفظي",
    "sentence": "إكمال الجمل",
    "reading": "استيعاب المقروء",
    "verbal_other": "أسئلة لفظية",
}
QTYPE_INSTRUCTION = {
    "quant_mcq": ("فيما يلي عدد من الأسئلة، يتبع كلًّا منها أربعة خيارات، المطلوب هو "
                  "اختيار الإجابة الصحيحة ثم تظليل الحرف المقابل لها في ورقة الإجابة."),
    "analogy": ("في بداية كل سؤال مما يأتي كلمتان ترتبطان بعلاقة معينة، تتبعها أربعة أزواج "
                "من الكلمات، واحد منها ترتبط فيه الكلمتان بعلاقة مشابهة للعلاقة بين الكلمتين "
                "في رأس السؤال، المطلوب هو اختيار الإجابة الصحيحة."),
    "sentence": ("في كل جملة مما يأتي فراغ أو أكثر، وتتبعها أربعة اختيارات، المطلوب هو "
                 "اختيار الإجابة التي تكمل الفراغ أو الفراغات بشكل صحيح."),
    "reading": ("اقرأ النص التالي بعناية ثم أجب عن الأسئلة التي تليه باختيار الإجابة الصحيحة "
                "وفقًا لما ورد في النص."),
    "verbal_other": "اختر الإجابة الصحيحة.",
}

_EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_AR_TASHKEEL = _re.compile(r"[ً-ْٰ]")

# Relation/skill difficulty weights for verbal subtypes (0 easy -> 2 hard).
_VERBAL_SUBCAT_DIFF = {
    "inference": 2.0, "logical connector": 1.6, "cause–effect": 1.5, "cause-effect": 1.5,
    "function/use": 1.1, "object and part of the whole": 0.9, "object and location": 0.9,
    "performer and action": 0.9, "vocabulary-in-context": 1.2,
    "opposites": 0.3, "synonyms": 0.3,
}


def _difficulty_quant(row: dict) -> float:
    q = str(row.get("Question", "")).translate(_EN_DIGITS)
    score = 0.0
    nums = _re.findall(r"\d+", q)
    score += min(len(nums), 6) * 0.5                       # more quantities -> harder
    biggest = max((int(n) for n in nums), default=0)
    score += min(biggest / 100.0, 5.0)                     # larger magnitudes
    score += len(_re.findall(r"[+\-×÷*/=]", q)) * 0.4      # operations / steps
    if _re.search(r"√|sqrt|π|pi|\^|<sup>", q, _re.I):      # symbolic content
        score += 1.5
    score += min(len(q) / 120.0, 4.0)                      # length ~ multi-step
    return score


def _difficulty_verbal(row: dict) -> float:
    """Verbal difficulty proxy: reading-passage depth + lexical complexity +
    dual-blank sentence completion + relation-type difficulty. Tuned to span a
    similar 0-~9 range as the quant proxy so both ramp/tier on one scale."""
    ctx = _AR_TASHKEEL.sub("", str(row.get("Context") or ""))
    q = _AR_TASHKEEL.sub("", str(row.get("Question") or ""))
    opts = " ".join(str(row.get(f"Option{L}") or "") for L in ("A", "B", "C", "D"))
    opts = _AR_TASHKEEL.sub("", opts)

    ctx_words = len(ctx.split())
    lex = (q + " " + opts).split()
    long_words = sum(1 for w in lex if len(w) >= 7)          # rarer/longer vocabulary
    avg_len = (sum(len(w) for w in lex) / len(lex)) if lex else 0.0

    score = 0.0
    score += min(ctx_words / 25.0, 3.5)                     # reading passage depth
    score += min(long_words * 0.3, 2.5)                     # lexical rarity
    score += max(0.0, avg_len - 4.5) * 0.8                  # lexical density
    # dual-blank sentence completion is markedly harder than single-blank
    blanks = len(_re.findall(r"_{2,}|\.{3,}|…", q))
    if blanks >= 2:
        score += 1.5
    # relation / skill type
    sub = str(row.get("Sub-Category") or "").strip().lower()
    score += _VERBAL_SUBCAT_DIFF.get(sub, 0.6)
    # longer answer options tend to demand finer discrimination
    score += min(len(opts) / 90.0, 1.5)
    return score


def difficulty_score(row: dict) -> float:
    """Deterministic difficulty proxy (NOT the LLM self-report). Quant questions
    are scored on operands/steps/symbols; verbal questions on passage depth,
    vocabulary, blanks, and relation type — both normalized to one 0-~10 scale so
    a section can ramp easy->hard across mixed question types. Higher = harder."""
    is_quant = (row.get("Section") == "Quantitative") or _qtype(row) == "quant_mcq"
    score = _difficulty_quant(row) if is_quant else _difficulty_verbal(row)
    lvl = str(row.get("Complexity-Level", "")).lower()
    score += {"basic": 0.0, "intermediate": 1.5, "advanced": 3.0}.get(lvl, 0.0)
    return score


def _distribute(total: int, n_sections: int) -> list:
    """Split `total` questions across `n_sections` as evenly as possible, the
    remainder landing on the earliest sections (e.g. 95/4 -> [24,24,24,23])."""
    base, rem = divmod(total, n_sections)
    return [base + (1 if i < rem else 0) for i in range(n_sections)]


def _deal_balanced(groups, n_sections, per_section_counts):
    """Round-robin deal each group's rows across sections for an even mix, then
    within each section group by question type (canonical order) and order each
    group easy -> hard. `per_section_counts` is a per-section cap list. Returns a
    list of `n_sections` lists, each capped at its target count."""
    if isinstance(per_section_counts, int):
        per_section_counts = [per_section_counts] * n_sections
    sections = [[] for _ in range(n_sections)]
    # Deal each (category/subcat) group round-robin, rotating the start section
    # so small groups don't all pile onto section 0.
    start = 0
    for rows in groups:
        for j, r in enumerate(rows):
            sections[(start + j) % n_sections].append(r)
        start = (start + len(rows)) % n_sections

    def _stratified(sorted_grp, k):
        """Pick k items spread across the difficulty range of an ascending-sorted
        group, so trimming an over-sampled pool KEEPS the hard tail (naive [:k]
        would drop every hard item). Returns them still ascending."""
        n = len(sorted_grp)
        if k >= n:
            return sorted_grp
        # evenly-spaced indices across [0, n-1], inclusive of the hardest (n-1)
        idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [n - 1]
        # top up if rounding collided
        i = n - 1
        while len(idx) < k and i >= 0:
            if i not in idx:
                idx.append(i)
            i -= 1
        return [sorted_grp[j] for j in sorted(idx)][:k]

    out = []
    for si, sec in enumerate(sections):
        cap = per_section_counts[si] if si < len(per_section_counts) else len(sec)
        # Group by type in canonical order.
        by_type = {}
        for r in sec:
            by_type.setdefault(_qtype(r), []).append(r)
        present = [qt for qt in QTYPE_ORDER if by_type.get(qt)]
        # Allocate the section cap across present types proportional to availability.
        avail = {qt: len(by_type[qt]) for qt in present}
        tot = sum(avail.values()) or 1
        quota = {qt: min(avail[qt], max(1, round(cap * avail[qt] / tot))) for qt in present}
        # Fix rounding drift so quotas sum exactly to cap (respecting availability).
        while sum(quota.values()) > cap:
            qt = max(quota, key=lambda k: quota[k]); quota[qt] -= 1
        while sum(quota.values()) < cap and any(quota[qt] < avail[qt] for qt in present):
            qt = max(present, key=lambda k: avail[k] - quota[k]); quota[qt] += 1
        ordered = []
        for qt in present:
            grp = sorted(by_type[qt], key=difficulty_score)   # easy -> hard
            picked = _stratified(grp, quota[qt])              # keep hard tail
            picked.sort(key=difficulty_score)                 # ramp within type
            ordered.extend(picked)
        out.append(ordered[:cap])
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


import math as _math


def _opt_value(s: str):
    """Real numeric value of an option for sorting: leading number × π if the
    option is a π-term (ط/π), else the plain number. None if not numeric."""
    t = str(s).translate(_EN_DIGITS)
    m = _re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    v = float(m.group())
    if "ط" in t or "π" in t or "pi" in t.lower():
        v *= _math.pi
    return v


def _sort_options_asc(options: dict, correct: str):
    """Return (options, correct) with numeric options reordered ascending (low→high)
    and the correct letter remapped. No-op unless all four options are numeric."""
    letters = ["A", "B", "C", "D"]
    vals = {L: _opt_value(options.get(L, "")) for L in letters}
    if any(vals[L] is None or not str(options.get(L, "")).strip() for L in letters):
        return options, correct
    order = sorted(letters, key=lambda L: vals[L])
    new_opts = {L: options[order[i]] for i, L in enumerate(letters)}
    new_correct = letters[order.index(correct)] if correct in order else correct
    return new_opts, new_correct


def _manifest(sections, time_min):
    data = {"n_sections": len(sections), "sections": []}
    qno = 0
    for idx, sec in enumerate(sections):
        questions = []
        for r in sec:
            qno += 1
            img = str(r.get("ImagePath") or "").strip()
            qt = _qtype(r)
            _opts = {L: str(r.get(f"Option{L}") or "").strip()
                     for L in ("A", "B", "C", "D")}
            _corr = str(r.get("CorrectOption") or "").strip().upper()
            _opts, _corr = _sort_options_asc(_opts, _corr)  # low→high for numeric sets
            questions.append({
                "qno": qno,
                "question": str(r.get("Question") or "").strip(),
                "context": str(r.get("Context") or "").strip(),
                "options": _opts,
                "correct": _corr,
                "answer": str(r.get("Answer") or "").strip(),
                "image_path": img,
                "explanation": str(r.get("Explanation") or "").strip(),
                "section_name": r.get("Section", ""),
                "category": r.get("Category", ""),
                "subcat": r.get("Sub-Category", ""),
                "qtype": qt,
                "qtype_title": QTYPE_TITLE.get(qt, ""),
                "qtype_instruction": QTYPE_INSTRUCTION.get(qt, ""),
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
    ap.add_argument("--total", type=int, default=95,
                    help="total questions; split evenly across sections "
                         "(remainder on earliest sections, e.g. 95/4 -> 24,24,24,23)")
    ap.add_argument("--per-section", type=int, default=None,
                    help="override: fixed questions per section (total = sections*this)")
    ap.add_argument("--time-min", type=int, default=26)
    ap.add_argument("--overshoot", type=float, default=1.3,
                    help="over-generate by this factor so cross-section dedup and "
                         "figure drops still leave enough to fill every section; the "
                         "dealer then caps each section to its exact target")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore any existing checkpoint and regenerate every task")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the checkpoint before starting")
    ap.add_argument("--builtin", action="store_true", default=True,
                    help="use the balanced real-Qudrat built-in blueprint (default) "
                         "instead of the authored xlsx sheet")
    ap.add_argument("--from-sheet", dest="builtin", action="store_false",
                    help="use the authored xlsx sheet instead of the built-in blueprint")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    gv.IMG_DIR.mkdir(exist_ok=True)

    if args.fresh and CKPT_JSONL.exists():
        CKPT_JSONL.unlink()
    done = _ckpt_load() if args.resume else {}
    if done:
        gv.gen.log(f"=== Resuming: {len(done)} task(s) already in checkpoint "
                   f"{CKPT_JSONL.name} ===")

    if args.per_section is not None:
        _total_for_bp = args.per_section * args.sections
    else:
        _total_for_bp = args.total
    if args.builtin:
        tasks = bp.builtin_blueprint(_total_for_bp, args.lang)
        gv.gen.log("=== Using built-in balanced blueprint (real-Qudrat proportions) ===")
    else:
        tasks = bp.load_blueprint(args.sheet, args.lang)
    blueprint_total = sum(t.count for t in tasks)
    if args.per_section is not None:
        per_section_counts = [args.per_section] * args.sections
    else:
        per_section_counts = _distribute(args.total, args.sections)
    target_total = sum(per_section_counts)
    # Generate with an overshoot margin (dedup + figure drops shrink the pool),
    # but the dealer still caps each section to its exact target below.
    scale = target_total * max(1.0, args.overshoot) / blueprint_total
    gv.gen.log(f"=== Full exam: {target_total} questions "
               f"(sections {per_section_counts}), scale {scale:.2f} "
               f"(overshoot {args.overshoot}) ===")

    # Pre-warm the shared corpus cache once so parallel task threads don't all
    # trigger (and race) the lazy load on first access.
    try:
        gv._load_data()
    except Exception as e:
        gv.gen.log(f"[warn] corpus pre-warm failed: {e}", level="warning")

    # One registry for the whole run — guarantees every accepted question is
    # unique across all sections (content + reversed options + analogy pairs).
    registry = SeenRegistry()

    def _run_task(t):
        """Generate (or resume) one task. Returns its rows; checkpoints new ones.

        Every returned row is unique across the whole exam: rows are filtered
        through the shared `registry`, and if cross-section duplicates drop the
        count below target we generate additional rounds to top up."""
        key = _task_key(t)
        if key in done:
            # Keep only rows still unique against everything already accepted.
            rows = [r for r in done[key] if registry.add_if_new(r)]
            gv.gen.log(f"--- {t.category}/{t.ar_subcat}: resumed {len(rows)} "
                       f"from checkpoint ---")
            return rows

        target = max(1, round(t.count * scale))
        kept = []
        for rnd in range(1, DEDUP_MAX_ROUNDS + 1):
            need = target - len(kept)
            if need <= 0:
                break
            try:
                rows = _generate_task(t, scale, target_override=need)
            except Exception as e:
                gv.gen.log(f"[{t.category}/{t.ar_subcat}] task failed, skipping: {e}",
                           level="error")
                break
            fresh = [r for r in rows if registry.add_if_new(r)]
            kept.extend(fresh)
            dropped = len(rows) - len(fresh)
            if dropped:
                gv.gen.log(f"[{t.category}/{t.ar_subcat}] round {rnd}: dropped {dropped} "
                           f"cross-section duplicate(s), {len(kept)}/{target}")

        if len(kept) < target:
            gv.gen.log(f"[{t.category}/{t.ar_subcat}] only {len(kept)}/{target} unique "
                       f"question(s) after {DEDUP_MAX_ROUNDS} round(s)", level="warning")
        for r in kept:               # tag section for interleave/manifest
            r.setdefault("Section", t.section)
        _ckpt_append(key, kept)      # persist immediately — survive a crash
        gv.gen.log(f"--- {t.category}/{t.ar_subcat}: got {len(kept)} (checkpointed) ---")
        return kept

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

    sections = _deal_balanced(groups, args.sections, per_section_counts)

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
