#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAT Unified Question Generator + Quiz (Quantitative + Verbal)
- Gemini generates NEW questions + correct answer (no options)
- Gemini majority-vote validates the answer
- Gemini generates ONLY 3 distractors for Quantitative (relaxed filtering)
- Verbal distractors use category-specific prompts from category_prompts_v2.json
- SPECIAL CASE: Comparison (Value A vs Value B) options are always fixed.

Install:
  pip install streamlit pandas openpyxl google-genai

Env:
  export GOOGLE_API_KEY="..."

Run:
  streamlit run app6_final.py
"""

import os
import json
import math
import random
import re
import logging
import time
from io import BytesIO
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import streamlit as st
from google import genai


# ============================================================
# LOGGING (Terminal Only)
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def log(msg: str, level: str = "info"):
    getattr(logger, level, logger.info)(msg)


# ============================================================
# API KEYS / CLIENT
# ============================================================

# Load .env so a direct import (not only via generate_visual) picks up the key.
# .env holds GEMINI_API_KEY; bridge it to GOOGLE_API_KEY which this module reads.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not set.")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Limit concurrent Gemini API calls to avoid hitting rate limits (429).
# gemini-2.5-pro has a low RPM cap; this semaphore ensures at most
# _MAX_CONCURRENT_API_CALLS requests are in-flight at any time.
_MAX_CONCURRENT_API_CALLS = 3
_api_semaphore = threading.Semaphore(_MAX_CONCURRENT_API_CALLS)


# ============================================================
# CONFIG
# ============================================================

FILE_PATH = "Qudrat Sample Sheet _ v2 (1).xlsx"
SHEETS_TO_LOAD: Optional[List[str]] = None
PROMPTS_JSON = "category_prompts_v2.json"

GENERATOR_MODEL = "gemini-2.5-pro"
# Holistic MCQ critic runs on flash (judging is cheaper than generating) and is
# batched, so a full exam costs only a handful of critic calls.
CRITIC_MODEL = "gemini-2.5-flash"
CRITIC_BATCH_SIZE = 8
# Hard per-request timeout (ms). A hung call would otherwise block a future.result()
# forever and freeze the run. Timeouts are retried inside gemini_call.
# 60s was too short for long Reading-Comprehension passage generation on thinking
# models (every RC call timed out -> 0 RC questions). 150s gives them room; still
# bounded so a truly hung call can't freeze the run. Env-overridable.
GEMINI_CALL_TIMEOUT_MS = int(os.getenv("GEMINI_CALL_TIMEOUT_MS", "150000"))
DISTRACTOR_TEMPERATURE = 0.35
DISTRACTOR_MAX_RETRIES = 2
DISTRACTOR_TOPUP_TRIES = 10
# Deterministic backstop: after distractors are built, verify none of them is ALSO
# a correct answer to the question (the "two options are both right" reviewer bug).
DISTRACTOR_VERIFY = os.getenv("DISTRACTOR_VERIFY", "1") not in ("0", "false", "False")

SOLVER_MODELS = [
    {"name": "gemini-2.5-flash", "temperature": 0.0},
    {"name": "gemini-2.5-flash", "temperature": 0.3},
    {"name": "gemini-2.5-flash", "temperature": 0.0},
]

NUM_EXAMPLES_IN_PROMPT = 8
MAX_GENERATION_ROUNDS = 5
GEN_OVERSAMPLE_FACTOR = 3.0
MAX_SOLVER_CALLS_PER_ROUND = 60

REQUIRED_COLUMNS = ["Question", "Section", "Category", "Sub-Category", "CorrectAnswer"]

# Similarity thresholds (question text)
DIVERSITY_THRESHOLD_POOL = 0.75
DIVERSITY_THRESHOLD_CORPUS = 0.75
CHAR_NGRAM_SIM_THRESHOLD = 0.82

ARABIC_DIGIT_MAP = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
EN_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

ALLOWED_VERBAL_CATS = {
    "Analogies & Word Relationships",
    "Sentence Completion",
    "Reading Comprehension",
}

GENERIC_MISCONCEPTIONS = [
    "order_of_operations",
    "formula_misuse",
    "ratio_inversion",
    "percent_base_error",
    "sign_error",
    "unit_confusion",
    "fraction_decimal_mixup",
    "misread_question",
]

_SAUDI_CULTURAL_RULES = """
ABSOLUTE CULTURAL PROHIBITIONS — Saudi Islamic Standards (zero tolerance):
- No alcohol, pork, gambling, usury/interest (riba), or any islamically prohibited (haram) content.
- No romantic, dating, or gender-mixing scenarios involving unrelated adults.
- No content disrespectful to Islam, the Quran, the Prophet Muhammad (peace be upon him), or Islamic values.
- No content that contradicts core Islamic beliefs.
- No politically sensitive content about Saudi Arabia, its government, or regional geopolitics.
- No derogatory, offensive, or culturally insensitive word choices.
- If names are needed, use gender-neutral or Arabic/Islamic-appropriate names.
- Financial scenarios must not involve interest (riba); use profit, discount, or cost-based framing only.
"""


def to_arabic_digits(s: str) -> str:
    return str(s).translate(ARABIC_DIGIT_MAP)

def to_english_digits(s: str) -> str:
    return str(s).translate(EN_DIGIT_MAP)

def clean_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# UNIFIED GEMINI CALLER
# ============================================================

def gemini_call(
    prompt: str,
    model: Optional[str] = None,
    system_instruction: str = "",
    temperature: float = 0.4,
    response_mime_type: str = "application/json",
    max_retries: int = 5,
) -> str:
    # Resolve at call time so a runtime override of GENERATOR_MODEL reaches every caller.
    model = model or GENERATOR_MODEL
    # Per-request timeout (ms) so a hung HTTP call can't freeze the whole pipeline
    # forever via a blocking future.result(). Timeouts are retried like 429s.
    config = genai.types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=temperature,
        response_mime_type=response_mime_type,
        http_options=genai.types.HttpOptions(timeout=GEMINI_CALL_TIMEOUT_MS),
    )
    base_wait = 30
    for attempt in range(1, max_retries + 1):
        try:
            with _api_semaphore:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.text or '{"error": "Empty response"}'
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            is_timeout = ("timeout" in msg.lower() or "deadline" in msg.lower()
                          or "timed out" in msg.lower())
            if is_429:
                wait = base_wait * attempt
                log(f"[429] Rate limited. Waiting {wait}s (attempt {attempt}/{max_retries})", level="warning")
                time.sleep(wait)
            elif is_timeout:
                log(f"[timeout] Request exceeded {GEMINI_CALL_TIMEOUT_MS}ms; "
                    f"retrying (attempt {attempt}/{max_retries})", level="warning")
                # immediate retry — no backoff needed for a hung connection
            else:
                return f'{{"error": "{e}"}}'
    return '{"error": "Max retries exceeded"}'


def call_model(prompt: str, model: Optional[str] = None) -> str:
    """Thin wrapper for verbal distractor pipeline (G_Distractors compatibility)."""
    return gemini_call(
        prompt,
        model=model,
        system_instruction="You are a JSON generator. Output only valid JSON.",
        temperature=0.1,
        response_mime_type="application/json",
    )


# ============================================================
# DATA LOADING / NORMALIZATION
# ============================================================

def infer_lang(sheet_name: str, df: pd.DataFrame) -> str:
    name = str(sheet_name).lower()
    if "arabic" in name or "عرب" in name:
        return "Arabic"
    sample = df["Question"].astype(str).tolist()
    arabic_ratio = sum(bool(re.search(r"[\u0600-\u06FF]", q)) for q in sample) / max(1, len(sample))
    return "Arabic" if arabic_ratio >= 0.4 else "English"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: re.sub(r"\s+", " ", str(c)).strip() for c in df.columns})
    mapping = {
        "Correct Option": "CorrectAnswer",
        "Correct Answer": "CorrectAnswer",
        "Correct answer": "CorrectAnswer",
        "Correct option": "CorrectAnswer",
        "Sub - Category": "Sub-Category",
        "Sub - Category ": "Sub-Category",
        "Sub - Category  ": "Sub-Category",
        "Category ": "Category",
        "Section ": "Section",
    }
    df = df.rename(columns=mapping)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nGot: {list(df.columns)}")

    for col in ["Section", "Category", "Sub-Category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "None": "", "NaN": ""})
    df["Question"] = df["Question"].astype(str).str.strip()
    df["CorrectAnswer"] = df["CorrectAnswer"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_exam_data(file_path: str, sheets: Optional[List[str]]) -> pd.DataFrame:
    log(f"Loading exam data from '{file_path}'")
    xls = pd.ExcelFile(file_path)
    sheets = sheets or xls.sheet_names
    frames = []
    for s in sheets:
        df = pd.read_excel(file_path, sheet_name=s)
        df = normalize_columns(df)
        df["Language"] = infer_lang(s, df)
        df["SourceSheet"] = s
        frames.append(df)
        log(f"Loaded sheet '{s}' with {len(df)} rows")
    combined = pd.concat(frames, ignore_index=True)
    log(f"Final dataset size: {len(combined)} rows")
    return combined


# ============================================================
# BUILD RULES (Sections → Categories → Subcategories)
# ============================================================

def build_rules(df: pd.DataFrame) -> Dict[str, Any]:
    rules: Dict[str, Any] = {}
    for _, r in df.iterrows():
        sec = str(r["Section"]).strip()
        cat = str(r["Category"]).strip()
        sub = str(r["Sub-Category"]).strip()
        comp = str(r.get("Complexity", "")).strip()

        # Skip rows with missing/invalid classification values
        if not sec or sec.lower() in ("nan", "none"):
            continue
        if not cat or cat.lower() in ("nan", "none"):
            continue
        if not sub or sub.lower() in ("nan", "none"):
            sub = "General"  # fall back to generic sub-category

        rules.setdefault(sec, {})
        rules[sec].setdefault(cat, {"subs": {}, "complex": set()})
        rules[sec][cat]["subs"].setdefault(sub, {"complex": set()})

        if comp:
            rules[sec][cat]["complex"].add(comp)
            rules[sec][cat]["subs"][sub]["complex"].add(comp)
    return rules


# ============================================================
# PROMPTS — QUANTITATIVE
# ============================================================

def system_prompt_quant(lang: str) -> str:
    common_rules = (
        "You generate NEW Quantitative GAT questions.\n\n"
        "Global rules (MUST follow):\n"
        "- Use gender-neutral wording (e.g., \"a person\", \"the student\", \"someone\").\n"
        "- Use culturally neutral contexts (NO nationality, tribe, religion, stereotypes).\n"
        "- Avoid cultural bias entirely.\n"
        "- Exponents: use ^ (e.g., x^2, 3^4) — never the words 'squared'/'مربع' or a "
        "raised digit. The interface renders ^n as a proper superscript.\n"
        "- Square roots: use the √ symbol ONLY (e.g., √144, √(x+1)). NEVER write the "
        "word 'جذر', 'الجذر التربيعي', 'sqrt', or 'root'.\n"
        "- If using degrees, always include the ° symbol (e.g., 30°, 45°, 90°).\n"
        "- DO NOT generate multiple-choice options.\n"
        "- DO NOT explain the solution.\n"
        "- Ensure the question is solvable and the Answer is verifiably correct.\n"
        "- PAST-PAPER STYLE (match the Examples): phrase questions tersely and "
        "directly, like real GAT/Qudrat past papers. No story wrapping, no extra "
        "context, just the precise mathematical ask.\n"
        "- FIGURE-BASED QUESTIONS (Geometry, Tables/Charts/Graphs): write the "
        "question to REFER to the figure rather than restating its data. Do NOT "
        "list all the chart/table values inside the question text; instead say "
        "things like 'وفقًا للرسم البياني', 'من الجدول', 'في الشكل المجاور' and ask "
        "about the data (highest/lowest, difference, sum, average, a specific "
        "value). The figure will carry the numbers. Keep ONLY the minimal number "
        "needed to state the task (e.g. a given total or percentage).\n"
        "- The Answer must be uniquely determined by the figure's data. Pick an "
        "Answer that is correct for a realistic, VARIED dataset (not all-equal "
        "values).\n"
        "- Return clean JSON ONLY (no markdown).\n"
    )
    if lang.lower() == "arabic":
        return common_rules + (
            "Language-specific:\n"
            "- Write the question ONLY in Modern Standard Arabic.\n"
            "- Use Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) for all numerals.\n"
            "- Keep math symbols standard: + - × ÷ = ^ √ ° ( ) . Use √ for roots.\n"
        )
    return common_rules + "Language-specific:\n- Write the question in English.\n"

def format_examples(examples: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        correct_letter = str(ex.get("CorrectAnswer", "")).strip().upper()
        option_key = f"Option {correct_letter}"  # e.g., "Option A"
        actual_answer = str(ex.get(option_key, ex.get("CorrectAnswer", ""))).strip()
        blocks.append(
            f"{i}) Q: {ex['Question']}\n"
            f"   Answer: {actual_answer}\n"
            f"   Category: {ex['Category']}\n"
            f"   Sub-Category: {ex['Sub-Category']}"
        )
    return "\n\n".join(blocks)

def user_prompt(
    examples_block: str,
    num_new: int,
    section: str,
    category: str,
    subcat: Optional[str],
    valid_subcats: Optional[List[str]] = None,
) -> str:
    req = f"- Section must be '{section}'."
    if category != "ANY":
        if subcat and subcat != "ANY":
            req += f"\n- Category must be '{category}' and Sub-Category must be EXACTLY '{subcat}'."
        else:
            req += f"\n- Category must be '{category}'."
            if valid_subcats:
                subs_str = ", ".join(f"'{s}'" for s in valid_subcats)
                req += f"\n- Sub-Category must be EXACTLY one of: {subs_str}"
            else:
                req += "\n- Sub-Category must be valid."

    return f"""
Examples (DO NOT copy text directly):

{examples_block}

Now generate {num_new} NEW questions.

STRICT REQUIREMENTS:
{req}
- Use gender-neutral and culturally neutral phrasing.
- Avoid any cultural references, stereotypes, or bias.
- If a question uses exponents, write them using ^ notation (e.g., x^2, 5^3).
- Square roots: use the √ symbol ONLY (e.g., √144); never the word 'جذر'/'sqrt'/'root'.
- If a question uses angles, use the ° symbol (e.g., 30°, 60°, 120°).
- DO NOT include answer choices.
- DO NOT explain the reasoning or steps.

Each question MUST be output as the following JSON object:
{{
  "Question": "...",
  "Answer": "...",
  "Section": "Quantitative",
  "Category": "...",
  "Sub-Category": "...",
  "Complexity-Level": "Basic"
}}

Return ONLY:
{{
  "questions": [ ... ]
}}
"""


# ============================================================
# PROMPTS — VERBAL
# ============================================================

def _analogy_subtype_rules(subcat: str) -> str:
    """Return unambiguous generation rules for a specific analogy relationship type."""
    s = subcat.strip().lower()

    if "synonym" in s or "antonym" in s:
        return (
            "\nSUB-TYPE RULES — Synonyms / Antonyms:\n"
            "- SYNONYM: both words share the SAME primary meaning in standard usage. "
            "Near-synonyms with different connotations are FORBIDDEN (e.g., SMALL : TINY is OK; "
            "SMALL : INSIGNIFICANT is NOT — 'insignificant' adds a value judgment).\n"
            "- ANTONYM: words must be DIRECT opposites with no middle ground "
            "(ALIVE : DEAD is OK; HAPPY : SAD is NOT — emotions have degrees).\n"
            "- Both words in each pair must be the same part of speech.\n"
            "- FORBIDDEN: pairs that are merely associated rather than synonymous/antonymous.\n"
        )
    if "degree" in s or "attribute" in s:
        return (
            "\nSUB-TYPE RULES — Degree of a Characteristic:\n"
            "- One word is a MILDER form; the other is a MORE INTENSE form of the SAME property. "
            "The difference is in INTENSITY ONLY — not in kind.\n"
            "- CORRECT example: WARM : SCALDING (both are heat, different intensity).\n"
            "- WRONG example: WARM : COLD (these are opposites, not degrees).\n"
            "- The direction must be consistent: both pairs must go mild→intense OR both intense→mild.\n"
            "- FORBIDDEN: pairs where the words differ in kind rather than only in degree.\n"
        )
    if "cause" in s or "effect" in s:
        return (
            "\nSUB-TYPE RULES — Cause and Effect:\n"
            "- The FIRST word is the CAUSE; the SECOND word is its DIRECT EFFECT.\n"
            "- The causal relationship must be universal and objective — not cultural, subjective, or debatable.\n"
            "- CORRECT: DROUGHT : FAMINE. WRONG: RAIN : HAPPINESS (subjective).\n"
            "- No multi-step chains — cause must DIRECTLY produce the effect.\n"
            "- Both stem and answer pairs must follow the same direction: cause→effect.\n"
            "- FORBIDDEN: pairs where the effect only sometimes follows the cause.\n"
        )
    if "category" in s or "class" in s or "inclusion" in s or "classification" in s:
        return (
            "\nSUB-TYPE RULES — Category / Class Inclusion:\n"
            "- The FIRST word is a SPECIFIC INSTANCE; the SECOND is the BROADER CATEGORY it belongs to.\n"
            "- CORRECT: SPARROW : BIRD, ALGEBRA : MATHEMATICS.\n"
            "- WRONG: BIRD : WING (that is part-whole, not class inclusion).\n"
            "- The 'is a type of' relationship must be unambiguous and universally accepted.\n"
            "- Both pairs must follow the same direction: specific→general.\n"
            "- FORBIDDEN: instance-category pairs that only hold in a specific or non-standard context.\n"
        )
    if "part" in s or "whole" in s:
        return (
            "\nSUB-TYPE RULES — Part–Whole:\n"
            "- The FIRST word is a PHYSICAL PART; the SECOND is the WHOLE it intrinsically belongs to.\n"
            "- CORRECT: PETAL : FLOWER, CHAPTER : BOOK, WHEEL : CAR.\n"
            "- WRONG: BRANCH : GOVERNMENT (metaphorical, not physical).\n"
            "- The part must be a standard, defining component — not optional or incidental.\n"
            "- FORBIDDEN: abstract or metaphorical part-whole relationships.\n"
        )
    if "function" in s or "use" in s:
        return (
            "\nSUB-TYPE RULES — Function / Use:\n"
            "- The FIRST word is a tool or instrument; the SECOND is its SINGLE PRIMARY function.\n"
            "- CORRECT: SCALPEL : CUT, THERMOMETER : MEASURE, SIEVE : FILTER.\n"
            "- WRONG: KNIFE : EAT (a knife's primary function is to cut, not to eat).\n"
            "- The function must be the item's most specific purpose — not a secondary or optional one.\n"
            "- FORBIDDEN: instruments with multiple equally primary functions.\n"
        )
    if "performer" in s or "action" in s:
        return (
            "\nSUB-TYPE RULES — Performer and Action:\n"
            "- The FIRST word is the AGENT; the SECOND is the action they CHARACTERISTICALLY perform.\n"
            "- CORRECT: SURGEON : OPERATES, JUDGE : RULES, AUTHOR : WRITES.\n"
            "- The action must be the performer's defining activity — not something anyone could do.\n"
            "- Use the same verb form in both pairs (both base form or both noun form).\n"
            "- FORBIDDEN: generic actions any person can perform (e.g., TEACHER : WALKS).\n"
        )
    if "location" in s:
        return (
            "\nSUB-TYPE RULES — Object and Location:\n"
            "- The FIRST word is an object; the SECOND is its CHARACTERISTIC and PRIMARY location.\n"
            "- CORRECT: FISH : OCEAN, BOOK : LIBRARY, PAINTING : MUSEUM.\n"
            "- FORBIDDEN: incidental or variable locations (BOOK : DESK — books can be anywhere).\n"
        )
    if "opposite" in s:
        return (
            "\nSUB-TYPE RULES — Opposites:\n"
            "- Words must be TRUE binary opposites — direct contradictions with no spectrum between them.\n"
            "- CORRECT: LIGHT : DARK, MAXIMUM : MINIMUM, EXPAND : CONTRACT.\n"
            "- Both pairs must use the SAME type of opposition.\n"
            "- FORBIDDEN: scalar opposites (FAST : SLOW) unless a clear binary context is established; "
            "words that are merely different rather than opposite.\n"
        )
    if "thing" in s or "go together" in s or "related object" in s or "object and group" in s:
        return (
            "\nSUB-TYPE RULES — Things That Go Together:\n"
            "- Words must have a FUNCTIONAL and NECESSARY co-occurrence — one cannot serve its primary purpose without the other.\n"
            "- CORRECT: NEEDLE : THREAD, LOCK : KEY.\n"
            "- FORBIDDEN: thematic association only (OCEAN : FISH — fish can live elsewhere).\n"
        )
    if "problem" in s or "solution" in s:
        return (
            "\nSUB-TYPE RULES — Problem and Solution:\n"
            "- The FIRST word names a PROBLEM; the SECOND names its DIRECT and CONVENTIONAL solution.\n"
            "- CORRECT: THIRST : WATER, INFECTION : ANTIBIOTIC.\n"
            "- The solution must be the single most direct remedy — not one of many options.\n"
            "- FORBIDDEN: solutions that are contestable or only sometimes applicable.\n"
        )
    if "characteristic" in s:
        return (
            "\nSUB-TYPE RULES — Object and Characteristic:\n"
            "- The SECOND word is the MOST DEFINING and INTRINSIC characteristic of the FIRST word.\n"
            "- CORRECT: DIAMOND : HARD, DESERT : DRY, FIRE : HOT.\n"
            "- The characteristic must be so fundamental that removing it changes the object's essential nature.\n"
            "- FORBIDDEN: incidental or variable characteristics (SKY : BLUE — the sky is also grey, black, etc.).\n"
        )
    if "verb tense" in s:
        return (
            "\nSUB-TYPE RULES — Verb Tense:\n"
            "- Both words must be DIFFERENT TENSES of the EXACT SAME verb.\n"
            "- CORRECT: RUN : RAN, WRITE : WROTE, GO : WENT.\n"
            "- Use IRREGULAR verbs — regular verb tense changes (add -ed) are too trivial.\n"
            "- Both pairs must apply the SAME tense transformation (e.g., both present→past).\n"
            "- FORBIDDEN: different verbs that merely look or sound similar.\n"
        )
    return ""


def system_prompt_verbal(lang: str, category: str, subcat: Optional[str] = None) -> str:
    lang_str = "Modern Standard Arabic" if lang.lower() == "arabic" else "English"

    if category == "Analogies & Word Relationships":
        case_rule = (
            "- Write word pairs in Arabic script (e.g., \"كَلِمَة : كَلِمَة\").\n"
            if lang.lower() == "arabic"
            else "- Use UPPERCASE for word pairs (e.g., \"TEACHER : STUDENT\").\n"
        )
        subtype_rules = _analogy_subtype_rules(subcat or "")
        base = (
            "You generate NEW Verbal GAT questions in the Analogies & Word Relationships category.\n\n"
            "Each question must:\n"
            "- Present a stem analogy pair (e.g., \"COLD : HOT\") in the Question field.\n"
            "- Provide the correct answer pair (e.g., \"DARK : LIGHT\") in the Answer field.\n"
            + case_rule +
            "- Use \" : \" (space-colon-space) as the separator between word pairs.\n"
            "- Specify the exact relationship type in the Sub-Category. Choose from:\n"
            "  Things that go together | Opposites | Synonyms | Object and classification |\n"
            "  Category and Class Inclusion | Object and group | Object and related object | Object and characteristic |\n"
            "  Object and location | Object and part of the whole | Function/Use |\n"
            "  Performer and action | Verb tense | Cause–Effect | Problem and solution |\n"
            "  Degree of a characteristic\n"
            "- VALIDATION: Before finalising a pair, confirm: (1) the relationship is UNIQUE to these two words, "
            "not merely associative; (2) no other pair of common words satisfies the same relationship at the same "
            "level of specificity.\n"
            "- EXAMPLE QUESTION:\n"
            "  Q: ARCHITECT : DESIGN\n"
            "  A: SURGEON : OPERATE\n"
            "  Sub-Category: Performer and action\n"
            "  (Stem and answer both follow: professional → their single defining action)\n"
            "- Do NOT include multiple-choice options.\n"
            "- Use vocabulary appropriate for a national standardized exam.\n"
        )
        if subtype_rules:
            base += subtype_rules + "\n"
        base += (
            f"{_SAUDI_CULTURAL_RULES}\n"
            f"Language: {lang_str}.\n"
            "Return clean JSON ONLY (no markdown)."
        )
        return base

    elif category == "Sentence Completion":
        subcat_str = (subcat or "").strip().lower()

        if "logical connector" in subcat_str or "connector" in subcat_str:
            sc_rule = (
                "- The blank must be filled by a LOGICAL CONNECTOR "
                "(e.g., however, therefore, although, consequently, despite, nevertheless).\n"
                "- The sentence must clearly establish a logical relationship (contrast, cause-effect, "
                "concession, sequence) so that only ONE connector TYPE is correct.\n"
                "- CONTEXT ANCHOR: Each clause must carry a specific semantic signal "
                "(a named cause, a named result, a clear contradiction) that eliminates all other connector types. "
                "Vague two-clause sentences are FORBIDDEN.\n"
                "- CONNECTOR TEST (apply before finalising): name at least 2 other connector types and "
                "confirm they produce a logically false or contradictory sentence — not just 'less natural'.\n"
                "- FORBIDDEN: sentences where two different connectors could produce equally valid meanings.\n"
            )
        elif "vocabulary" in subcat_str:
            sc_rule = (
                "- The blank must be filled by a CONTENT WORD.\n"
                "- CONTEXT ANCHOR: The sentence must contain a SPECIFIC detail, qualifier, or named subject "
                "that mathematically eliminates near-synonyms. Generic, topic-neutral sentences are FORBIDDEN.\n"
                "- BAD EXAMPLE (FORBIDDEN): 'One must ___ caution when dealing with sensitive information' — "
                "'exercise', 'maintain', 'employ', 'observe', and 'apply' all work equally well. Discard.\n"
                "- GOOD EXAMPLE: 'The archaeologist carefully ___ the artifact with cotton padding before "
                "sealing it' — 'wrapped' is the only word that fits because 'cotton padding' specifies the method.\n"
                "- NEAR-SYNONYM TEST (apply before finalising): name 3 near-synonyms of your chosen word "
                "and point to the SPECIFIC word or phrase in the sentence that eliminates each one. "
                "If you cannot point to a specific eliminator — the sentence is too vague. Discard and rewrite.\n"
                "- CORRECT ANSWER must be the most precise word the context demands — not just any grammatically valid word.\n"
                "- FORBIDDEN: sentences where a near-synonym of the correct word is equally acceptable.\n"
                "- Use vocabulary appropriate for a national standardized exam.\n"
            )
        elif "dual" in subcat_str:
            sc_rule = (
                "- DUAL BLANK: The sentence must contain EXACTLY TWO blanks (......).\n"
                "- LENGTH: The sentence must be SHORT — no more than 90 characters total including the blanks.\n"
                "- STYLE: Prefer well-known idioms, proverbs, or short factual phrases (e.g. 'Anger ...... with madness but ends in ......').\n"
                "- CONTEXT ANCHOR: The sentence must contain specific meaning that pins down BOTH blanks.\n"
                "- BAD EXAMPLE (too long/vague): 'In desert climates, plants have developed unique ...... that allow them to ...... water with remarkable efficiency.'\n"
                "- GOOD EXAMPLE (short, idiomatic): 'The end ...... the ......'\n"
                "- SEMANTIC LOCK: Blank 1 must logically force one specific value for Blank 2. The pair must form a LOCKED relationship.\n"
                "- CORRECT ANSWER must be the MOST PRECISE and MOST IDIOMATIC pair.\n"
                "- NEAR-SYNONYM TEST: for EACH blank, name 3 near-synonyms and point to the SPECIFIC word/phrase that eliminates each one.\n"
                "- Answer format: two words separated by a comma: 'word1, word2'.\n"
                "- FORBIDDEN: sentences longer than 90 characters.\n"
                "- FORBIDDEN: sentences where a near-synonym of either word is equally valid.\n"
                "- FORBIDDEN: pairs where the two words are merely topically related but not logically locked.\n"
            )
        else:
            sc_rule = (
                "- The blank(s) must each be filled by exactly one correct word — no near-synonym should fit equally well.\n"
                "- CONTEXT ANCHOR: The sentence must contain a specific detail that pins down the correct word.\n"
                "- FORBIDDEN: generic sentences that accept multiple equally valid completions.\n"
            )

        base = (
            "You generate NEW Verbal GAT questions in the Sentence Completion category.\n\n"
            "GLOBAL RULE — UNIQUENESS: The correct answer must be the ONE word or pair that the sentence "
            "specifically demands. It is not enough that the answer is grammatically valid — it must be "
            "the ONLY grammatically valid AND semantically precise option. If a student who knows the topic "
            "well could justify a different word, the sentence is too vague and must be discarded.\n\n"
            "Each question must:\n"
            "- Present a sentence with ONE or TWO blanks, each represented as \"______\".\n"
            "- Contain enough specific context that only ONE completion is clearly correct.\n"
            "- Provide the correct word(s) or phrase(s) in the Answer field "
            "(for two blanks, give both words separated by a comma, e.g. 'reject, mixed').\n"
        )
        if sc_rule:
            base += sc_rule
        base += (
            "- Do NOT include multiple-choice options.\n"
            "- Use formal language appropriate for a national standardized exam.\n"
            f"{_SAUDI_CULTURAL_RULES}\n"
            f"Language: {lang_str}.\n"
            "Return clean JSON ONLY (no markdown)."
        )
        return base

    elif category == "Reading Comprehension":
        subcat_str = (subcat or "").strip().lower()

        if "inference" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — INFERENCE: The question MUST require implicit reasoning.\n"
                "  * The answer must NOT appear verbatim or as a direct paraphrase in the passage.\n"
                "  * Students must combine multiple clues or read between the lines.\n"
                "  * Ask about implication, logical conclusion, unstated meaning, or author intent.\n"
                "  * FORBIDDEN: Any question whose exact answer phrase is directly stated in the passage.\n"
                "  * EXAMPLE frames: 'What can be inferred about...?', 'The passage implies that...', "
                "'What conclusion is best supported by...?'\n"
            )
        elif "main idea" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — MAIN IDEA: The question asks for the central/overall theme.\n"
                "  * The correct answer MUST be the broadest, most inclusive idea that covers ALL passage details.\n"
                "  * Correct answer = a conceptual summary of the whole passage, NOT a specific detail or example.\n"
                "  * FORBIDDEN: Using a supporting detail, statistic, or example as the correct answer.\n"
                "  * EXAMPLE frames: 'What is the main idea of this passage?', 'What is the passage primarily about?'\n"
            )
        elif "detail" in subcat_str or "fact" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — DETAIL/FACT: The question tests specific factual recall.\n"
                "  * The correct answer must be directly and explicitly stated in the passage text.\n"
                "  * EXAMPLE frames: 'According to the passage, what...?', 'The passage states that...'\n"
            )
        elif "logical organization" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — LOGICAL ORGANIZATION: The question tests passage structure.\n"
                "  * Ask about how the author organizes or sequences ideas.\n"
                "  * EXAMPLE frames: 'How does the author develop the argument?', "
                "'What is the purpose of the final sentence?'\n"
            )
        elif "vocabulary" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — VOCABULARY-IN-CONTEXT: The question tests contextual word meaning.\n"
                "  * Ask about a specific word or phrase as used in the passage.\n"
                "  * Correct answer must match the word's contextual meaning, not its general dictionary definition.\n"
                "  * EXAMPLE frames: 'As used in the passage, the word \"X\" most closely means...'\n"
            )
        elif "tone" in subcat_str or "purpose" in subcat_str:
            qtype_rule = (
                "- QUESTION TYPE — TONE/PURPOSE: The question asks about the author's intent or attitude.\n"
                "  * Ask about the passage's overall tone or the author's purpose in writing it.\n"
                "  * EXAMPLE frames: 'The author's tone in this passage is best described as...', "
                "'The primary purpose of this passage is to...'\n"
            )
        else:
            qtype_rule = ""

        base = (
            "You generate NEW Verbal GAT questions in the Reading Comprehension category.\n\n"
            "Each question must:\n"
            "- Include a short passage (3–5 sentences) in the \"Context\" field. The passage must be:\n"
            "  * Culturally appropriate for Saudi Arabia\n"
            "  * Factually accurate\n"
            "  * Self-contained (the answer must be supported by or inferable from the passage)\n"
            "- Present a specific comprehension question in the Question field.\n"
            "- Provide the correct answer in the Answer field.\n"
        )
        if qtype_rule:
            base += qtype_rule
        base += (
            "- Do NOT include multiple-choice options.\n"
            "- Use formal language appropriate for a national standardized exam.\n"
            f"{_SAUDI_CULTURAL_RULES}\n"
            f"Language: {lang_str}.\n"
            "Return clean JSON ONLY (no markdown)."
        )
        return base

    else:
        return (
            "You generate NEW Verbal GAT questions.\n"
            "Do NOT include multiple-choice options.\n"
            "Use formal language appropriate for a national standardized exam.\n"
            f"{_SAUDI_CULTURAL_RULES}\n"
            f"Language: {lang_str}.\n"
            "Return clean JSON ONLY (no markdown)."
        )


def format_verbal_examples(examples: List[Dict[str, Any]], category: str) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        correct_letter = str(ex.get("CorrectAnswer", "")).strip().upper()
        option_key = f"Option {correct_letter}"  # e.g., "Option A"
        actual_answer = str(ex.get(option_key, ex.get("CorrectAnswer", ""))).strip()
        block = (
            f"{i}) Q: {ex.get('Question', '')}\n"
            f"   Answer: {actual_answer}\n"
            f"   Category: {ex.get('Category', '')}\n"
            f"   Sub-Category: {ex.get('Sub-Category', '')}"
        )
        if category == "Reading Comprehension":
            ctx = ex.get("Context (ex. Reading)", ex.get("Context", ""))
            if ctx:
                ctx_str = str(ctx)[:200]
                block += f"\n   Context: {ctx_str}..."
        blocks.append(block)
    return "\n\n".join(blocks)


def user_prompt_verbal(examples_block: str, num_new: int, category: str, subcat: Optional[str],
                       rejection_notes: str = "") -> str:
    req = f"- Category must be '{category}'."
    if subcat and subcat != "ANY":
        req += f"\n- Sub-Category must be '{subcat}'."

    rc_note = ""
    if category == "Reading Comprehension":
        rc_note = '\nIMPORTANT: The "Context" field is REQUIRED. Include a 3–5 sentence passage.\n'
        if subcat and subcat != "ANY":
            subcat_lower = subcat.lower()
            if "inference" in subcat_lower:
                rc_note += (
                    'INFERENCE TYPE: The answer must NOT be a direct quote or paraphrase from the passage. '
                    'Students must reason from the text, not retrieve from it.\n'
                )
            elif "main idea" in subcat_lower:
                rc_note += (
                    'MAIN IDEA TYPE: The correct answer must be the broadest overarching theme of the passage, '
                    'NOT a specific detail, statistic, or example mentioned in the text.\n'
                )
            elif "detail" in subcat_lower or "fact" in subcat_lower:
                rc_note += (
                    'DETAIL/FACT TYPE: The correct answer must be explicitly stated in the passage text.\n'
                )

    if category == "Reading Comprehension":
        context_field = '  "Context": "...",  // 3-5 sentence passage (REQUIRED)'
    else:
        context_field = '  // "Context": "..."  // omit unless Reading Comprehension'

    notes_block = ""
    if rejection_notes:
        notes_block = (
            "\nPREVIOUS FAILURES — Do NOT generate questions similar to these:\n"
            + rejection_notes + "\n"
        )

    return f"""
Examples (DO NOT copy text directly):

{examples_block}

Now generate {num_new} NEW questions.

STRICT REQUIREMENTS:
{req}
- Section must be 'Verbal'.
- Use gender-neutral and culturally neutral phrasing.
- Do NOT include answer choices.
- Do NOT explain the reasoning.
{rc_note}{notes_block}
Each question MUST be output as the following JSON object:
{{
  "Question": "...",
  "Answer": "...",
{context_field}
  "Section": "Verbal",
  "Category": "{category}",
  "Sub-Category": "...",
  "Complexity-Level": "Basic"
}}

Return ONLY:
{{
  "questions": [ ... ]
}}
"""


# ============================================================
# VALIDATION HELPERS
# ============================================================

CHOICE_PATTERNS = [
    r"\bA\)", r"\bB\)", r"\bC\)", r"\bD\)",
    r"\(A\)", r"\(B\)", r"\(C\)", r"\(D\)",
    r"\bA\.", r"\bB\.", r"\bC\.", r"\bD\.",
    r"\ba\)", r"\bb\)", r"\bc\)", r"\bd\)",
    r"\(a\)", r"\(b\)", r"\(c\)", r"\(d\)"
]

def contains_choice_pattern(text: str) -> bool:
    return any(re.search(p, str(text)) for p in CHOICE_PATTERNS)

def _autocorrect_subcat(q: Dict[str, Any], rules: Dict[str, Any]) -> None:
    """Fuzzy-correct a slightly-off sub-category name before structural validation.
    If the model generated 'Word Problems' but the Excel has
    'Word Problems (Rate/Time/Work, Mixture, Age, Profit-Loss)', auto-correct in-place."""
    sec = clean_text(q.get("Section", ""))
    cat = clean_text(q.get("Category", ""))
    sub = clean_text(q.get("Sub-Category", ""))
    if not sec or not cat or not sub:
        return
    known_subs = rules.get(sec, {}).get(cat, {}).get("subs", {})
    if sub in known_subs:
        return  # already exact match
    sub_lower = sub.lower()
    for known_sub in known_subs:
        kl = known_sub.lower()
        if sub_lower == kl or sub_lower in kl or kl in sub_lower:
            log(f"[Quant] Sub-cat auto-corrected: '{sub}' → '{known_sub}'", level="warning")
            q["Sub-Category"] = known_sub
            return


def validate_structure(q: Dict[str, Any], rules: Dict[str, Any],
                       req_section: str, req_category: str, req_subcat: Optional[str]) -> List[str]:
    errors = []
    required = ["Question", "Answer", "Section", "Category", "Sub-Category", "Complexity-Level"]
    for field in required:
        if field not in q:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    sec = clean_text(q["Section"])
    cat = clean_text(q["Category"])
    sub = clean_text(q["Sub-Category"])
    question = clean_text(q["Question"])
    answer = clean_text(q["Answer"])

    if not question:
        errors.append("Question text is empty.")
    if len(question) < 10:
        errors.append("Question too short (likely invalid).")

    if sec != req_section:
        errors.append(f"Section mismatch: expected '{req_section}', got '{sec}'.")

    if sec not in rules:
        errors.append(f"Section '{sec}' not found in rules.")
    else:
        if cat not in rules[sec]:
            errors.append(f"Invalid category '{cat}' for section '{sec}'.")
        else:
            if sub not in rules[sec][cat]["subs"]:
                errors.append(f"Invalid sub-category '{sub}' for category '{cat}'.")

    if req_category != "ANY" and cat != req_category:
        errors.append(f"Category mismatch: expected '{req_category}', got '{cat}'.")
    if req_subcat and req_subcat != "ANY" and sub != req_subcat:
        errors.append(f"Sub-category mismatch: expected '{req_subcat}', got '{sub}'.")

    if contains_choice_pattern(question):
        errors.append("Question appears to include multiple-choice options.")

    if not answer:
        errors.append("Answer field is empty.")

    return errors


def validate_verbal_structure(
    q: Dict[str, Any],
    rules: Dict[str, Any],
    req_category: str,
    req_subcat: Optional[str],
) -> List[str]:
    errors = []
    required = ["Question", "Answer", "Section", "Category", "Sub-Category", "Complexity-Level"]
    for field in required:
        if field not in q:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    sec = clean_text(q["Section"])
    cat = clean_text(q["Category"])
    sub = clean_text(q["Sub-Category"])
    question = clean_text(q["Question"])
    answer = clean_text(q["Answer"])

    if not question:
        errors.append("Question text is empty.")
    if len(question) < 10:
        errors.append("Question too short (likely invalid).")

    if sec.lower() != "verbal":
        errors.append(f"Section must be 'Verbal', got '{sec}'.")

    # Category mismatch = hard fail
    if req_category != "ANY" and cat != req_category:
        errors.append(f"Category mismatch: expected '{req_category}', got '{cat}'.")

    # Sub-category mismatch = warning only
    if req_subcat and req_subcat != "ANY" and sub != req_subcat:
        log(f"[Verbal] Sub-category mismatch (warning only): expected '{req_subcat}', got '{sub}'.", level="warning")

    # RC requires Context
    if cat == "Reading Comprehension":
        context = clean_text(q.get("Context", ""))
        if not context or len(context) < 100:
            errors.append("Reading Comprehension requires 'Context' with at least 100 characters.")
        else:
            # Fix 4: Inference answers must not appear verbatim in the passage
            if "inference" in sub.lower():
                if answer.lower() in context.lower():
                    errors.append(
                        "Inference question: answer appears verbatim in the passage — "
                        "this tests retrieval, not inference. Regenerate."
                    )

    # Analogies answer must contain ':'
    if cat == "Analogies & Word Relationships":
        if ":" not in answer:
            errors.append("Analogies answer must contain ':' (e.g., 'WORD1 : WORD2').")

    # Fix 5: Two-blank Sentence Completion must have a two-part answer
    if cat == "Sentence Completion":
        blank_count = question.count("______")
        _has_dual_sep = (
            "," in answer
            or " \u2013 " in answer   # en dash (old system prompt format)
            or "\u2013" in answer
            or " - " in answer         # regular hyphen (dataset format)
        )
        if blank_count == 2 and not _has_dual_sep:
            errors.append(
                "Two-blank Sentence Completion has a single-word answer. "
                "Expected format: 'word1, word2' or 'word1 \u2013 word2'."
            )

    if contains_choice_pattern(question):
        errors.append("Question appears to include multiple-choice options.")

    if not answer:
        errors.append("Answer field is empty.")

    return errors


# ============================================================
# SIMILARITY HELPERS
# ============================================================

def norm_tokens(s: str) -> set:
    s = to_english_digits(str(s).lower())
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)  # strip Arabic diacritics for consistent hashing
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    return set(t for t in s.split() if t)

def jaccard(A: set, B: set) -> float:
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def char_ngrams(s: str, n: int = 4) -> set:
    s = to_english_digits(str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def char_ngram_sim(a: str, b: str, n: int = 4) -> float:
    return jaccard(char_ngrams(a, n), char_ngrams(b, n))

def too_similar(q1: str, q2: str) -> bool:
    if jaccard(norm_tokens(q1), norm_tokens(q2)) >= DIVERSITY_THRESHOLD_POOL:
        return True
    return char_ngram_sim(q1, q2, 4) >= CHAR_NGRAM_SIM_THRESHOLD

def max_similarity_to_corpus(q_text: str, corpus_tokens: List[set]) -> float:
    qt = norm_tokens(q_text)
    best = 0.0
    for ct in corpus_tokens:
        best = max(best, jaccard(qt, ct))
    return best


# ============================================================
# ANSWER VALIDATION — QUANTITATIVE (GEMINI MAJORITY VOTE)
# ============================================================

def try_num(s: str) -> Optional[float]:
    s = to_english_digits(clean_text(s))

    # Fraction: -3/4
    frac = re.fullmatch(r"\s*(-?\d+)\s*/\s*(\d+)\s*", s)
    if frac:
        a, b = float(frac.group(1)), float(frac.group(2))
        if b != 0:
            return a / b

    # Square root: √N  or  sqrt(N)
    m = re.fullmatch(r"\s*√\s*(\d+(?:\.\d+)?)\s*", s)
    if m:
        try:
            return math.sqrt(float(m.group(1)))
        except ValueError:
            pass
    m = re.fullmatch(r"\s*sqrt\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*", s, re.I)
    if m:
        try:
            return math.sqrt(float(m.group(1)))
        except ValueError:
            pass

    # Pi expressions: Nπ, π, N*π, π/N  (also accepts "pi")
    m = re.fullmatch(r"\s*(-?\d*\.?\d*)\s*[×*]?\s*(?:π|pi)\s*", s, re.I)
    if m:
        coef_str = m.group(1).strip()
        coef = float(coef_str) if coef_str not in ("", "-", "+") else (
            -1.0 if coef_str == "-" else 1.0
        )
        return coef * math.pi
    m = re.fullmatch(r"\s*(?:π|pi)\s*/\s*(\d+(?:\.\d+)?)\s*", s, re.I)
    if m:
        denom = float(m.group(1))
        if denom != 0:
            return math.pi / denom

    # Euler's number: Ne or e  (only as a standalone constant, not as exponent notation)
    m = re.fullmatch(r"\s*(-?\d*\.?\d*)\s*[×*]?\s*e\s*", s)
    if m:
        coef_str = m.group(1).strip()
        coef = float(coef_str) if coef_str not in ("", "-", "+") else (
            -1.0 if coef_str == "-" else 1.0
        )
        return coef * math.e

    # Plain number
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def answers_match(a: str, b: str, tol: float = 1e-3) -> bool:
    a = to_english_digits(clean_text(a))
    b = to_english_digits(clean_text(b))
    na, nb = try_num(a), try_num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= tol
    a2 = re.sub(r"\s+", " ", a.lower()).strip()
    b2 = re.sub(r"\s+", " ", b.lower()).strip()
    return a2 == b2

def solve_with_gemini(question_text: str, model_name: str, temperature: float = 0.0) -> str:
    solve_prompt = (
        "Solve the following quantitative problem and return ONLY the final answer.\n"
        "- Use ^ notation for exponents (e.g., x^2, 3^4) and √ for roots (e.g., √144).\n"
        "- Use the ° symbol for degrees (e.g., 30°, 60°).\n"
        "- Do NOT rewrite the question.\n\n"
        f"{question_text}"
    )
    result = gemini_call(
        solve_prompt,
        model=model_name,
        response_mime_type="text/plain",
        temperature=temperature,
    )
    if result.startswith('{"error"'):
        return f"ERR: {result}"
    return result.strip()

def majority_validate(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    for q in questions:
        given = clean_text(q["Answer"])
        votes = 0
        solver_answers = []

        with ThreadPoolExecutor(max_workers=len(SOLVER_MODELS)) as executor:
            futures = [
                executor.submit(solve_with_gemini, q["Question"], cfg["name"], cfg.get("temperature", 0.0))
                for cfg in SOLVER_MODELS
            ]
            for cfg, future in zip(SOLVER_MODELS, futures):
                sol = future.result()
                solver_answers.append({"model": cfg["name"], "answer": sol})
                if not sol.startswith("ERR") and answers_match(sol, given):
                    votes += 1

        q2 = dict(q)
        q2["solver_answers"] = solver_answers
        q2["votes"] = votes
        # Adaptive majority so the pipeline works with any solver count (e.g. a
        # single solver for speed on figure tasks): need > half the solvers.
        threshold = len(SOLVER_MODELS) // 2 + 1
        q2["answer_match_majority"] = (votes >= threshold)
        validated.append(q2)

    return validated


# ============================================================
# ANSWER VALIDATION — VERBAL (GEMINI MAJORITY VOTE)
# ============================================================

def remove_arabic_diacritics(s: str) -> str:
    """Strip Arabic tashkeel so 'سَعَادَة' == 'سعادة' in comparisons."""
    return re.sub(r"[\u064B-\u065F\u0670]", "", s)

def answers_match_verbal(a: str, b: str, category: str) -> bool:
    a = remove_arabic_diacritics(clean_text(a))
    b = remove_arabic_diacritics(clean_text(b))

    if category == "Analogies & Word Relationships":
        def norm_analog(s: str) -> str:
            s = re.sub(r"\s*:\s*", " : ", s.strip())
            return s.lower().strip()
        an = norm_analog(a)
        bn = norm_analog(b)
        if an == bn:
            return True
        ta = norm_tokens(an)
        tb = norm_tokens(bn)
        return jaccard(ta, tb) >= 0.6  # relaxed: diacritic-stripped tokens overlap well

    elif category == "Sentence Completion":
        # Normalize dash/en-dash to comma so dual-blank split works for both formats
        def _norm_dual(s: str) -> str:
            return re.sub(r"\s*[\u2013\u2014-]\s*", ", ", s)
        a = _norm_dual(a)
        b = _norm_dual(b)
        # Two-blank SC: match each part independently (e.g., "reject, mixed")
        if "," in a and "," in b:
            a_parts = [p.strip().lower() for p in a.split(",", 1)]
            b_parts = [p.strip().lower() for p in b.split(",", 1)]
            if len(a_parts) == 2 and len(b_parts) == 2:
                part0_ok = a_parts[0] in b_parts[0] or b_parts[0] in a_parts[0]
                part1_ok = a_parts[1] in b_parts[1] or b_parts[1] in a_parts[1]
                return part0_ok and part1_ok
        al = a.lower().strip()
        bl = b.lower().strip()
        if al in bl or bl in al:
            return True
        ta = norm_tokens(a)
        tb = norm_tokens(b)
        return jaccard(ta, tb) >= 0.6

    else:
        al = a.lower().strip()
        bl = b.lower().strip()
        if al in bl or bl in al:
            return True
        ta = norm_tokens(a)
        tb = norm_tokens(b)
        return jaccard(ta, tb) >= 0.6


def solve_with_gemini_verbal(
    question_text: str, context: str, category: str, model_name: str,
    temperature: float = 0.0,
) -> str:
    if category == "Analogies & Word Relationships":
        solve_prompt = (
            "You are solving an analogy question from a standardized exam.\n"
            "Given the stem pair, identify the relationship and provide the completing word pair.\n"
            "Use the SAME language and script as the stem pair (Arabic if the stem is Arabic, English if English).\n"
            "Return ONLY the answer pair separated by ' : ' (space-colon-space), no extra text.\n\n"
            f"Stem: {question_text}"
        )
    elif category == "Sentence Completion":
        solve_prompt = (
            "Complete the following sentence by filling in the blank(s).\n"
            "If the sentence has TWO blanks, return BOTH words separated by a comma (e.g., 'word1, word2').\n"
            "Return ONLY the word(s) that correctly fill the blank(s), no explanation.\n\n"
            f"Sentence: {question_text}"
        )
    elif category == "Reading Comprehension":
        passage = context or question_text
        solve_prompt = (
            "Read ONLY the passage below and answer the question.\n"
            "Do NOT use any external knowledge — base your answer SOLELY on the passage text.\n"
            "Return ONLY the direct answer (no explanation, no full sentences unless required).\n\n"
            f"Passage:\n{passage}\n\n"
            f"Question: {question_text}"
        )
    else:
        solve_prompt = (
            "Answer the following question.\n"
            "Return ONLY the answer (no explanation).\n\n"
            f"{question_text}"
        )

    result = gemini_call(
        solve_prompt,
        model=model_name,
        response_mime_type="text/plain",
        temperature=temperature,
    )
    if result.startswith('{"error"'):
        return f"ERR: {result}"
    return result.strip()


def validate_sc_unambiguous(question_text: str, correct_answer: str, lang: str,
                            subcat: str = "") -> Tuple[bool, str]:
    """
    Validates that a Sentence Completion question has exactly ONE defensible correct answer.
    Sends the question to Gemini (flash) and asks it to test near-synonyms.
    Returns (True, "")    → question is unambiguous (safe to keep).
    Returns (False, reason) → question is ambiguous (reject and regenerate).

    EXCEPTION — Logical Connector sub-type: near-synonym connectors (although / even though / while)
    are inherently interchangeable within the same relationship class. GAT options always span
    DIFFERENT relationship types (contrast vs cause vs addition), so the near-synonym test is
    not meaningful here. We skip the validator and always return True for LC questions.
    """
    # Logical Connector questions are valid as long as the sentence clearly requires a specific
    # relationship type — the validator cannot distinguish same-class connectors and would
    # incorrectly reject every well-formed LC question.
    if "logical connector" in subcat.lower():
        return True, ""

    is_dual = "dual" in subcat.lower() or (
        "," in correct_answer or " \u2013 " in correct_answer or " - " in correct_answer
    )
    if is_dual:
        prompt = (
            f"You are validating a DUAL BLANK Sentence Completion question.\n\n"
            f"Sentence: {question_text}\n"
            f"Correct answer pair: {correct_answer}\n\n"
            "Your task: determine if this question is AMBIGUOUS.\n"
            "A dual-blank question is AMBIGUOUS only if you can name a COMPLETE ALTERNATIVE PAIR "
            "(both words simultaneously replaced by near-synonyms) that is equally valid in the sentence.\n"
            "Replacing only ONE of the two words with a near-synonym is NOT sufficient \u2014 both must be "
            "simultaneously replaceable by a pair that is equally natural and idiomatic.\n\n"
            "VERDICT RULES:\n"
            "- AMBIGUOUS: a complete alternative word pair exists that fits the sentence just as well.\n"
            "- UNAMBIGUOUS: no complete alternative pair is equally idiomatic. Even if individual words "
            "have near-synonyms, the specific combination required makes this unambiguous. Default to UNAMBIGUOUS.\n\n"
            "Return ONLY this JSON (no markdown):\n"
            "{\n"
            '  "alternative_pair_tested": "word1 \u2013 word2",\n'
            '  "verdict": "AMBIGUOUS" or "UNAMBIGUOUS",\n'
            '  "reason": "one sentence"\n'
            "}"
        )
        raw = gemini_call(prompt, model="gemini-2.5-flash",
                          response_mime_type="application/json", temperature=0.0)
        data = safe_json_load(raw)
        if not data or "error" in data:
            return True, ""
        return data.get("verdict") == "UNAMBIGUOUS", data.get("reason", "")

    prompt = (
        f"You are a psychometrician validating a Sentence Completion question for a standardized exam.\n\n"
        f"Sentence: {question_text}\n"
        f"Intended correct answer: {correct_answer}\n\n"
        "Your task:\n"
        "1. Generate exactly 3 near-synonyms or semantically related alternatives to the correct answer.\n"
        "2. Insert each alternative into the sentence and judge: is the result grammatically correct "
        "AND semantically valid in the same context?\n"
        "3. Check whether the correct answer is the MOST PRECISE and MOST IDIOMATIC choice — "
        "or if a stronger, more canonical alternative exists.\n"
        "4. Give a final verdict.\n\n"
        "VERDICT RULES:\n"
        "- AMBIGUOUS: two or more near-synonyms are fully INTERCHANGEABLE with the correct answer — "
        "neither is clearly more precise, specific, or idiomatic than the other in this exact context. "
        "A word being 'also grammatically valid' is NOT sufficient; it must be truly EQUALLY appropriate.\n"
        "- UNAMBIGUOUS: the correct answer is clearly the most precise or idiomatic fit. Near-synonyms "
        "may be 'acceptable' in isolation but are meaningfully less precise here — you can point to a "
        "specific word or phrase that makes the correct answer distinctly better.\n\n"
        "IMPORTANT: Do NOT mark as AMBIGUOUS just because a synonym *could* fit. Only mark AMBIGUOUS "
        "if a vocabulary expert would genuinely debate which word is correct.\n\n"
        "Return ONLY this JSON (no markdown):\n"
        "{\n"
        '  "near_synonyms_tested": ["word1", "word2", "word3"],\n'
        '  "any_synonym_truly_interchangeable": true/false,\n'
        '  "verdict": "AMBIGUOUS" or "UNAMBIGUOUS",\n'
        '  "reason": "one sentence explanation"\n'
        "}"
    )
    raw = gemini_call(
        prompt,
        model="gemini-2.5-flash",
        response_mime_type="application/json",
        temperature=0.0,
    )
    data = safe_json_load(raw)
    if not data or "error" in data:
        return True, ""  # if validator fails (503, timeout, parse error), keep the question
    return data.get("verdict") == "UNAMBIGUOUS", data.get("reason", "")


def validate_analogy_relationship(stem: str, correct_answer: str, subcat: str, lang: str) -> Tuple[bool, str]:
    """
    Validates that an analogy pair has a precise, unique relationship.
    Returns True  → relationship is valid and unambiguous (keep).
    Returns False → relationship is weak, wrong direction, or many pairs satisfy it (reject).
    """
    prompt = (
        f"You are validating an analogy question for a standardized exam.\n\n"
        f"Stem pair: {stem}\n"
        f"Correct answer pair: {correct_answer}\n"
        f"Claimed relationship type: {subcat}\n\n"
        "IMPORTANT CONTEXT: In GAT analogy questions, students are given a stem pair and four answer pairs. "
        "They must select the answer pair that has the SAME relationship type as the stem. It is EXPECTED "
        "that many word pairs in the world share the same broad relationship — that is the nature of an "
        "analogy test. Do NOT reject a question just because other pairs also fit the relationship.\n\n"
        "Your task:\n"
        "1. State the exact logical relationship in the stem pair in one precise sentence.\n"
        "2. Confirm whether the correct answer pair satisfies EXACTLY the same relationship — "
        "same direction and same logical category as the stem pair.\n"
        "3. Is the claimed relationship type a reasonable label for this relationship?\n\n"
        "VERDICT RULES:\n"
        "- INVALID: the correct answer pair does NOT actually share the same relationship as the stem "
        "pair, OR the relationship is completely misidentified (wrong type entirely).\n"
        "- VALID: the correct answer pair clearly demonstrates the same relationship as the stem, and "
        "the type label is reasonable. It is OK that other pairs in the world share this relationship.\n"
        "When in doubt, choose VALID.\n\n"
        "Return ONLY this JSON (no markdown):\n"
        "{\n"
        '  "relationship": "precise one-sentence description",\n'
        '  "correct_pair_matches": true/false,\n'
        '  "relationship_correctly_typed": true/false,\n'
        '  "verdict": "VALID" or "INVALID",\n'
        '  "reason": "one sentence explanation"\n'
        "}"
    )
    raw = gemini_call(
        prompt,
        model="gemini-2.5-flash",
        response_mime_type="application/json",
        temperature=0.0,
    )
    data = safe_json_load(raw)
    if not data or "error" in data:
        return True, ""  # if validator fails (503, timeout, parse error), keep the question
    return data.get("verdict") == "VALID", data.get("reason", "")


def validate_rc_question(
    passage: str, question: str, correct_answer: str, subcat: str, lang: str
) -> bool:
    """
    Validates that an RC question is properly anchored to its passage and matches its sub-type.
    Returns True  → valid question (keep).
    Returns False → question fails sub-type requirements or answer is not clearly best (reject).
    """
    subcat_lower = subcat.strip().lower()

    if "inference" in subcat_lower:
        type_instruction = (
            "INFERENCE type — verify ALL of the following:\n"
            "- The answer is NOT word-for-word or obviously paraphrased directly from one sentence.\n"
            "- The answer requires reading beyond the surface — it cannot be retrieved by copy-paste.\n"
            "- The answer is reasonable and clearly supported by the passage as a whole.\n"
            "NOTE: Do NOT require the student to combine two or more clues — a single strong contextual "
            "clue is sufficient for a valid inference. Do NOT reject just because another inference "
            "is also possible; only reject if the passage directly contradicts the given answer, or "
            "if the answer is a direct quote/paraphrase with no inferential step needed.\n"
        )
    elif "main idea" in subcat_lower:
        type_instruction = (
            "MAIN IDEA type — verify ALL of the following:\n"
            "- The answer covers ALL passage details — not just one sentence or one example.\n"
            "- The answer is the broadest statement that unifies the entire passage.\n"
            "- The answer is NOT a supporting detail, statistic, or single example from the passage.\n"
        )
    elif "detail" in subcat_lower or "fact" in subcat_lower:
        type_instruction = (
            "DETAIL/FACT type — verify ALL of the following:\n"
            "- The answer is explicitly and directly stated in the passage text.\n"
            "- The question asks about a specific retrievable fact, not an inference.\n"
        )
    elif "vocabulary" in subcat_lower:
        type_instruction = (
            "VOCABULARY-IN-CONTEXT type — verify ALL of the following:\n"
            "- The answer matches the word's meaning AS USED IN THIS PASSAGE, not its dictionary definition.\n"
            "- The passage context clearly establishes this specific meaning over other meanings.\n"
        )
    elif "tone" in subcat_lower or "purpose" in subcat_lower:
        type_instruction = (
            "TONE/PURPOSE type — verify ALL of the following:\n"
            "- The answer accurately reflects the overall tone or purpose of the entire passage.\n"
            "- The tone/purpose is consistently supported throughout — not just in one sentence.\n"
        )
    elif "logical organization" in subcat_lower:
        type_instruction = (
            "LOGICAL ORGANIZATION type — verify ALL of the following:\n"
            "- The question asks about passage structure, not content.\n"
            "- The answer correctly describes how the author organizes or sequences ideas.\n"
        )
    else:
        type_instruction = (
            "General RC — verify ALL of the following:\n"
            "- The answer is clearly supported by the passage.\n"
            "- A careful reader would arrive at this answer and no other.\n"
        )

    prompt = (
        f"You are validating a Reading Comprehension question for a standardized exam.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question: {question}\n"
        f"Correct answer: {correct_answer}\n"
        f"Question type: {subcat}\n\n"
        "Your task:\n"
        f"{type_instruction}\n"
        "The type requirements above are GUIDANCE, not a strict gate. Only TWO things "
        "make a question INVALID:\n"
        "  (1) the passage alone is NOT sufficient to answer it (needs outside knowledge), OR\n"
        "  (2) the passage DIRECTLY CONTRADICTS the given answer (the answer is wrong).\n"
        "Everything else is VALID. Do NOT reject because another answer is also arguable, "
        "because the type fit is imperfect, or because the question is easy. When in doubt, VALID.\n\n"
        "Return ONLY this JSON (no markdown):\n"
        "{\n"
        '  "passage_self_contained": true/false,\n'
        '  "answer_supported": true/false,        // false ONLY if passage contradicts the answer\n'
        '  "verdict": "VALID" or "INVALID",\n'
        '  "reason": "one sentence explanation"\n'
        "}"
    )
    raw = gemini_call(
        prompt,
        model="gemini-2.5-flash",
        response_mime_type="application/json",
        temperature=0.0,
    )
    data = safe_json_load(raw)
    if not data or "error" in data:
        return True  # if validator fails (503, timeout, parse error), keep the question
    # Lenient gate: reject only on the two hard failures, not on strict type-fit.
    self_contained = data.get("passage_self_contained", True)
    supported = data.get("answer_supported", True)
    if self_contained is False or supported is False:
        return False
    return True


def majority_validate_verbal(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    for q in questions:
        given = clean_text(q["Answer"])
        category = clean_text(q.get("Category", ""))
        context = clean_text(q.get("Context", ""))
        votes = 0
        solver_answers = []

        with ThreadPoolExecutor(max_workers=len(SOLVER_MODELS)) as executor:
            futures = [
                executor.submit(solve_with_gemini_verbal, q["Question"], context, category,
                                cfg["name"], cfg.get("temperature", 0.0))
                for cfg in SOLVER_MODELS
            ]
            for cfg, future in zip(SOLVER_MODELS, futures):
                sol = future.result()
                solver_answers.append({"model": cfg["name"], "answer": sol})
                if not sol.startswith("ERR") and answers_match_verbal(sol, given, category):
                    votes += 1

        q2 = dict(q)
        q2["solver_answers"] = solver_answers
        q2["votes"] = votes
        # Adaptive majority so the pipeline works with any solver count (e.g. a
        # single solver for speed on figure tasks): need > half the solvers.
        threshold = len(SOLVER_MODELS) // 2 + 1
        q2["answer_match_majority"] = (votes >= threshold)
        validated.append(q2)

    return validated


# ============================================================
# UNIT HELPERS
# ============================================================

# Latin unit variants -> canonical surface form.
_LATIN_UNITS = {
    "km": "km", "m": "m", "cm": "cm", "mm": "mm",
    "kg": "kg", "g": "g", "l": "l", "ml": "ml",
    "km/h": "km/h", "m/s": "m/s", "sr": "SR",
}
# Arabic unit variants -> canonical surface form. NOTE: "ط" (pi) and bare
# prepositions ("ل","م" alone) are deliberately NOT units here — pi is notation,
# handled separately; ambiguous single letters are only matched number-adjacently.
_AR_UNITS = {
    "ريال": "ريال", "ريالات": "ريال", "ريالاً": "ريال", "ريالا": "ريال",
    "سم": "سم", "كم": "كم", "مم": "مم", "م": "م",
    "كجم": "كجم", "كغ": "كجم", "كغم": "كجم",
    "جم": "جم", "غم": "جم", "غ": "جم",
    "لتر": "لتر", "مل": "مل",
    "سم²": "سم²", "م²": "م²", "كم²": "كم²",
    "درجة": "°", "درجات": "°",
}
# Backwards-compat alias retained for any external callers.
UNIT_WHITELIST = set(_LATIN_UNITS) | set(_AR_UNITS) | {"%", "°"}

# Arabic letters only (excludes Arabic-Indic digits 0660-0669 and diacritics 064B+).
_UNIT_CHARS = r"[A-Za-zء-ي]"
# A trailing unit token: %, °, or a letter-run optionally suffixed with ²/2 (area).
_TRAILING_UNIT_RE = re.compile(r"(%|°|(?:" + _UNIT_CHARS + r")+(?:²|2)?)\s*$")
# A number (Western or Arabic-Indic) immediately followed by a unit token.
_NUM_UNIT_RE = re.compile(
    r"[0-9٠-٩]\s*(%|°|(?:" + _UNIT_CHARS + r")+(?:²|2)?)"
)

def _canonical_unit(tok: str) -> Optional[str]:
    """Map a raw token to its canonical unit, or None if it isn't a real unit."""
    if not tok:
        return None
    tok = tok.strip()
    if tok in ("%", "°"):
        return tok
    if tok in ("ط", "π", "pi", "Pi", "PI"):  # pi is notation, not a unit
        return None
    key = tok.replace("2", "²")
    if key in _AR_UNITS:
        return _AR_UNITS[key]
    low = tok.lower()
    if low in _LATIN_UNITS:
        return _LATIN_UNITS[low]
    return None

def extract_unit_from_answer(ans: str) -> Optional[str]:
    """Canonical unit trailing a single answer/option value (Arabic or Latin)."""
    t = clean_text(ans)
    if not t:
        return None
    m = _TRAILING_UNIT_RE.search(t)
    if not m:
        return None
    return _canonical_unit(m.group(1))

def _strip_known_unit(text: str) -> str:
    """Remove a trailing KNOWN unit token, leaving the bare value. No-op otherwise."""
    t = clean_text(text)
    m = _TRAILING_UNIT_RE.search(t)
    if m and _canonical_unit(m.group(1)) is not None:
        return t[: m.start()].strip()
    return t

def _unit_from_question(question: str) -> Optional[str]:
    """First number-adjacent canonical unit in the stem (e.g. '٨٠ ريالًا' -> 'ريال')."""
    if not question:
        return None
    for m in _NUM_UNIT_RE.finditer(question):
        u = _canonical_unit(m.group(1))
        if u:
            return u
    return None

# What the QUESTION ASKS FOR (not what the givens mention). The stem-unit fallback
# was grabbing the first unit in the givens even when the asked quantity has a
# different dimension — e.g. "if 30% of a number = 90, what is THE NUMBER?" was
# stamped "%" from the "30%" given. These match the *requested* quantity.
_ASK_BARE_RE = re.compile(
    r"(هذا|ذلك)\s+العدد|ما\s+العدد|ما\s+ذلك\s+العدد|قيمة\s+[سصلxy]"
)
_ASK_PERCENT_RE = re.compile(
    r"النسبة\s+المئوية|كم\s+بالمئة|بالمائة|ما\s+النسبة"
)

# Sentinel distinct from None so callers can tell "asked for a bare number"
# (force no unit) apart from "no opinion" (fall through to other heuristics).
_FORCE_BARE = "\x00bare"

def _expected_unit_from_ask(question: str) -> Optional[str]:
    """The unit implied by what the question ASKS to output, or None if the ask
    gives no signal. Returns _FORCE_BARE when the ask is explicitly for a plain
    number/count so no stray unit gets stamped on."""
    if not question:
        return None
    if _ASK_PERCENT_RE.search(question):
        return "%"
    if _ASK_BARE_RE.search(question):
        return _FORCE_BARE
    return None

def enforce_units_on_options(correct: str, options: Dict[str, str],
                             question: str = "") -> Dict[str, str]:
    """Force all options to share ONE canonical unit form (all-unit or all-bare).

    Canonical unit priority: what the question ASKS for -> the correct answer ->
    majority of options -> a unit mentioned in the stem. Numeric options get the
    unit imposed; options already carrying a (possibly variant) unit are normalized
    to the canonical token. Non-numeric options (expressions, words) are left
    untouched.
    """
    # 1. Honor what the question explicitly asks to output. An explicit "bare
    #    number / count" ask forces no unit (fixes the "%"-from-givens bug);
    #    an explicit percentage ask forces "%".
    ask = _expected_unit_from_ask(question)
    if ask == _FORCE_BARE:
        return {k: _strip_known_unit(clean_text(v)) for k, v in options.items()}
    unit = ask if ask else None

    # 2. Otherwise trust the answer's own unit, then the options', then the stem.
    if not unit:
        unit = extract_unit_from_answer(correct)
    if not unit:
        opt_units = [extract_unit_from_answer(v) for v in options.values()]
        opt_units = [u for u in opt_units if u]
        if opt_units:
            unit = max(set(opt_units), key=opt_units.count)
    if not unit:
        unit = _unit_from_question(question)

    fixed: Dict[str, str] = {}
    for k, v in options.items():
        t = clean_text(v)
        bare = _strip_known_unit(t)
        if unit is None:
            # No canonical unit anywhere -> ensure none carry a stray unit.
            fixed[k] = bare
        elif try_num(bare) is not None and _looks_plain_number(bare):
            fixed[k] = f"{bare} {unit}".strip()
        else:
            # Expression (e.g. pi/fraction) or non-numeric: leave as-is.
            fixed[k] = t
    return fixed

def _looks_plain_number(text: str) -> bool:
    """True if text is a single numeric value (int/decimal/fraction), not an
    expression like '100 - 25π' that must keep its symbolic form."""
    s = to_english_digits(clean_text(text))
    return re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", s) is not None

def options_unit_homogeneous(options: Dict[str, str]) -> bool:
    """All numeric options share the same unit presence + canonical token."""
    seen = set()
    for v in options.values():
        bare = _strip_known_unit(v)
        if try_num(bare) is None or not _looks_plain_number(bare):
            continue  # skip expressions / words
        seen.add(extract_unit_from_answer(v) or "")
    return len(seen) <= 1

def options_notation_homogeneous(options: Dict[str, str]) -> bool:
    """Options must not mix pi/√/fraction symbolic form with bare integers."""
    def is_symbolic(v: str) -> bool:
        s = to_english_digits(_strip_known_unit(v))
        return bool(re.search(r"π|pi|ط|√|sqrt|/", s, re.I))
    flags = {is_symbolic(v) for v in options.values() if clean_text(v)}
    return len(flags) <= 1


# ============================================================
# JSON EXTRACTION
# ============================================================

def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ============================================================
# SPECIAL CASE: Comparison (Value A vs Value B)
# ============================================================

COMPARE_SUBCAT_KEYS = [
    "comparison (value a vs. value b)",
    "comparison (value a vs value b)",
    "comparison value a vs value b",
    "value a vs. value b",
    "value a vs value b",
    "مقارنة",
    "القيمة أ",
    "القيمة الأولى",
    "القيمة الثانية",
]

COMPARE_FIXED_AR = {
    "A": "القيمة الأولى أكبر.",
    "B": "القيمة الثانية أكبر.",
    "C": "القيمتان متساويتان.",
    "D": "المعطيات غير كافية.",
}

COMPARE_FIXED_EN = {
    "A": "Value A is greater.",
    "B": "Value B is greater.",
    "C": "The values are equal.",
    "D": "Not enough information.",
}

def is_compare_subcategory(subcat: str) -> bool:
    s = clean_text(subcat).lower()
    return any(k in s for k in COMPARE_SUBCAT_KEYS)

def _canon_ar(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"[^\w\u0600-\u06FF\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_compare_correct_label(answer_text: str, lang: str) -> Optional[str]:
    a = clean_text(answer_text)
    fixed = COMPARE_FIXED_AR if lang.lower() == "arabic" else COMPARE_FIXED_EN

    # Direct label
    if a.upper() in {"A", "B", "C", "D"}:
        return a.upper()

    # Exact text match
    for lbl, txt in fixed.items():
        if clean_text(txt) == a:
            return lbl

    # Canonical (stripped punctuation) match
    a2 = _canon_ar(a) if lang.lower() == "arabic" else _canon_ar(a.lower())
    for lbl, txt in fixed.items():
        t2 = _canon_ar(txt) if lang.lower() == "arabic" else _canon_ar(txt.lower())
        if a2 == t2:
            return lbl

    # Pattern-based matching
    al = a.lower()
    if lang.lower() != "arabic":
        # English patterns
        if re.search(r"\bvalue\s+a\s+is\s+(greater|larger|bigger|more)\b", al):
            return "A"
        if re.search(r"\bvalue\s+b\s+is\s+(greater|larger|bigger|more)\b", al):
            return "B"
        if re.search(r"\ba\s*>\s*b\b", al) or re.search(r"\ba\s+(?:is\s+)?greater\s+than\s+b\b", al):
            return "A"
        if re.search(r"\bb\s*>\s*a\b", al) or re.search(r"\bb\s+(?:is\s+)?greater\s+than\s+a\b", al):
            return "B"
        if re.search(r"\b(are\s+)?(equal|same|equivalent|identical)\b", al):
            return "C"
        if re.search(r"\b(cannot\s+be\s+determined|not\s+enough\s+info|insufficient|indeterminate)\b", al):
            return "D"
    else:
        # Arabic patterns
        if re.search(r"الأول[ىة]\s+أكبر|قيمة\s+[aأ]\s+أكبر|أ\s+أكبر", a):
            return "A"
        if re.search(r"الثاني[ةى]\s+أكبر|قيمة\s+[bب]\s+أكبر|ب\s+أكبر", a):
            return "B"
        if re.search(r"متساويت[ان]|تساو[يى]", a):
            return "C"
        if re.search(r"غير\s+كافي|لا\s+يمكن\s+تحديد|لا\s+يكفي|المعطيات\s+غير", a):
            return "D"

    return None

def build_compare_mcq(answer_text: str, lang: str) -> Dict[str, Any]:
    fixed = COMPARE_FIXED_AR if lang.lower() == "arabic" else COMPARE_FIXED_EN
    correct_label = detect_compare_correct_label(answer_text, lang)

    if correct_label not in {"A", "B", "C", "D"}:
        correct_label = "D"

    return {
        "OptionA": fixed["A"],
        "OptionB": fixed["B"],
        "OptionC": fixed["C"],
        "OptionD": fixed["D"],
        "CorrectOption": correct_label,

        "GenRationale_A": "",
        "GenRationale_B": "",
        "GenRationale_C": "",
        "GenRationale_D": "",
        "GenMisTag_A": "",
        "GenMisTag_B": "",
        "GenMisTag_C": "",
        "GenMisTag_D": "",
    }


# ============================================================
# QUANTITATIVE DISTRACTOR GENERATOR (GEMINI, replaces OpenAI)
# ============================================================

def _norm_key(s: str) -> str:
    s = to_english_digits(clean_text(s)).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_equal_to_correct(d_text: str, correct_text: str) -> bool:
    if answers_match(d_text, correct_text):
        return True
    return _norm_key(d_text) == _norm_key(correct_text)

def _distractor_also_valid(question: str, distractor: str, correct: str) -> bool:
    """True if the model judges `distractor` to ALSO be a correct answer to the
    question (i.e. the question would have two right options). Used as a hard
    gate to drop such distractors. Conservative: on any solver error, returns
    False (keep the distractor) so a flaky call never silently drops good ones."""
    if not question or not distractor:
        return False
    if _is_equal_to_correct(distractor, correct):
        return True  # identical to the key -> obviously "also valid", drop it
    prompt = (
        "You are checking a multiple-choice question for AMBIGUITY.\n"
        "A question must have exactly ONE correct option.\n"
        "Given the question (with any passage) and a candidate option, decide whether "
        "the candidate is ALSO a fully correct answer to the question (not just close).\n"
        "Reply with ONE word only: YES (it is also correct) or NO (it is wrong).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE OPTION: {distractor}\n"
    )
    try:
        # Use the first solver model at temperature 0 for a stable verdict.
        cfg = SOLVER_MODELS[0]
        out = gemini_call(prompt, model=cfg["name"], response_mime_type="text/plain",
                          temperature=0.0)
        if out.startswith('{"error"'):
            return False
        return out.strip().upper().startswith("YES")
    except Exception:
        return False

def _verbal_ambiguous(q2: Dict[str, Any]) -> bool:
    """True if any non-correct option of a verbal MCQ is ALSO a defensible answer
    (given the passage + question). Catches the 'two correct options' bug."""
    if not DISTRACTOR_VERIFY:
        return False
    ctx = clean_text(q2.get("Context", ""))
    ques = clean_text(q2.get("Question", ""))
    if not ques:
        return False
    full = (f"النص: {ctx}\n\n" if ctx else "") + ques
    correct_lbl = str(q2.get("CorrectOption", "")).strip().upper()
    answer = clean_text(q2.get("Answer", ""))
    for L in ("A", "B", "C", "D"):
        if L == correct_lbl:
            continue
        opt = clean_text(q2.get(f"Option{L}", ""))
        if not opt or opt == "[Option not available]":
            continue
        if _distractor_also_valid(full, opt, answer):
            log(f"[Verbal] ambiguous: option {L} ('{opt}') also defensible", level="warning")
            return True
    return False

def _drop_also_correct(items: List[Dict[str, str]], question: str,
                       correct: str) -> List[Dict[str, str]]:
    """Filter out any distractor the verifier flags as also-correct."""
    if not (DISTRACTOR_VERIFY and question):
        return items
    kept = []
    for d in items:
        if _distractor_also_valid(question, d.get("text", ""), correct):
            log(f"[Distractors] dropped also-correct option: {d.get('text','')}",
                level="warning")
            continue
        kept.append(d)
    return kept

def _filter_relaxed(raw: List[Dict[str, Any]], correct_text: str) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    seen = set()

    for item in raw:
        text = clean_text(item.get("text", ""))
        tag = clean_text(item.get("misconception_tag", ""))
        rat = clean_text(item.get("rationale", ""))

        if not text:
            continue
        if _is_equal_to_correct(text, correct_text):
            continue

        key = _norm_key(text)
        if key in seen:
            continue
        seen.add(key)

        if tag not in GENERIC_MISCONCEPTIONS:
            tag = "misread_question"
        if not rat:
            rat = f"{tag.replace('_', ' ')} leads to choosing {text}."
        rat = " ".join(rat.split()[:25])

        kept.append({"text": text, "misconception_tag": tag, "rationale": rat})

    return kept

def _fallback_make_3_distractors_relaxed(correct_text: str) -> List[Dict[str, str]]:
    c = clean_text(correct_text)
    seen = {_norm_key(c)}
    out: List[Dict[str, str]] = []

    cnum = try_num(c)
    if cnum is not None:
        candidates = [
            cnum + 1, cnum - 1,
            cnum + 2, cnum - 2,
            cnum * 2,
            (cnum / 2) if cnum != 0 else 0.5,
            -cnum if cnum != 0 else 1.0,
            cnum + 5, cnum - 3,
        ]

        def fmt(x: float) -> str:
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return f"{x:.4f}".rstrip("0").rstrip(".")

        for v in candidates:
            txt = fmt(v)
            if _is_equal_to_correct(txt, c):
                continue
            k = _norm_key(txt)
            if k in seen:
                continue
            seen.add(k)
            tag = random.choice(GENERIC_MISCONCEPTIONS)
            out.append({"text": txt, "misconception_tag": tag, "rationale": f"A common mistake leads to choosing {txt}."})
            if len(out) == 3:
                return out

    candidates = [c + " 0", c + " 1", c + " 2", "≈ " + c, c.replace("-", "")]
    for txt in candidates:
        if _is_equal_to_correct(txt, c):
            continue
        k = _norm_key(txt)
        if k in seen:
            continue
        seen.add(k)
        out.append({"text": txt, "misconception_tag": "misread_question", "rationale": f"Misreading leads to choosing {txt}."})
        if len(out) == 3:
            return out

    while len(out) < 3:
        txt = str(len(out) + 7)
        if not _is_equal_to_correct(txt, c) and _norm_key(txt) not in seen:
            seen.add(_norm_key(txt))
            out.append({"text": txt, "misconception_tag": "misread_question", "rationale": f"Guessing leads to choosing {txt}."})
    return out[:3]

def _generate_one_replacement_gemini(
    question_text: str, correct_text: str, lang: str, avoid_texts: List[str]
) -> Optional[Dict[str, str]]:
    q = clean_text(question_text)
    c = clean_text(correct_text)
    avoid_block = "\n".join(f"- {t}" for t in avoid_texts[:15]) if avoid_texts else "- (none)"
    lang_hint = "Write in Modern Standard Arabic only." if lang.lower() == "arabic" else "Write in English only."

    prompt = f"""You are generating ONE distractor for a GAT (Qudrat) quantitative MCQ.

QUESTION:
{q}

CORRECT ANSWER (must NOT be produced, even in equivalent form like 2.0 vs 2 or 4/2 vs 2):
{c}

Already-used distractors (must NOT repeat):
{avoid_block}

Rules:
- Output exactly ONE distractor.
- Must be WRONG but plausible.
- Must NOT equal the correct answer (including numeric equivalence).
- Keep format consistent with the correct answer (fraction vs decimal, %, unit, etc.).
- Provide misconception_tag from this list only:
  {GENERIC_MISCONCEPTIONS}
- Provide a SPECIFIC rationale (<= 25 words) referencing BOTH the distractor value AND what mathematical error produces it.
  GOOD example: "Dividing 12 by 4 instead of multiplying gives 3."
  BAD example: "common mistake" or "misread question"

{lang_hint}

Return JSON ONLY:
{{
  "text": "...",
  "misconception_tag": "...",
  "rationale": "..."
}}
"""
    try:
        raw = gemini_call(prompt, temperature=DISTRACTOR_TEMPERATURE)
        data = safe_json_load(raw)
        if not data:
            return None
        kept = _filter_relaxed([data], c)
        return kept[0] if kept else None
    except Exception:
        return None


def generate_3_distractors_gemini(question_text: str, correct_text: str, lang: str,
                                  extra_feedback: str = "") -> Dict[str, Any]:
    q = clean_text(question_text)
    c = clean_text(correct_text)

    lang_hint = (
        "Write distractors in Modern Standard Arabic only."
        if lang.lower() == "arabic"
        else "Write distractors in English only."
    )

    feedback_block = (
        f"\nAVOID these problems flagged in a previous attempt:\n- {extra_feedback}\n"
        if extra_feedback else ""
    )

    prompt = f"""You are a distractor generator for a GAT (Qudrat) quantitative multiple-choice question.

QUESTION:
{q}

CORRECT ANSWER (must NOT appear among distractors, even as an equivalent form like 2.0 vs 2 or 4/2 vs 2):
{c}

TASK:
Generate EXACTLY 3 distractors only (WRONG but plausible).

Rules:
- Return exactly 3 distractors.
- Do NOT include the correct answer among distractors (including numeric equivalence).
- Keep formatting consistent with the correct answer (fraction vs decimal, %, unit, π, etc.).
- Distractors may be close to the correct answer (exam-style), but must be wrong.
{feedback_block}
For each distractor:
- Provide misconception_tag from this list only:
  {GENERIC_MISCONCEPTIONS}
- Provide a SPECIFIC rationale (<= 25 words) referencing BOTH the distractor value AND what mathematical error produces it.
  GOOD example: "Dividing 12 by 4 instead of multiplying gives 3."
  BAD example: "common mistake" or "misread question"

{lang_hint}

Return JSON ONLY:
{{
  "distractors": [
    {{"text":"...","misconception_tag":"...","rationale":"..."}},
    {{"text":"...","misconception_tag":"...","rationale":"..."}},
    {{"text":"...","misconception_tag":"...","rationale":"..."}}
  ]
}}
"""
    last_err = None
    for _ in range(DISTRACTOR_MAX_RETRIES + 1):
        try:
            raw = gemini_call(prompt, temperature=DISTRACTOR_TEMPERATURE)
            data = safe_json_load(raw)
            if not data:
                raise ValueError("Could not parse JSON response.")
            raw_list = data.get("distractors", [])
            if not isinstance(raw_list, list):
                raw_list = []

            kept = _filter_relaxed(raw_list, c)
            # Drop any distractor that is ALSO a valid answer (ambiguity gate).
            kept = _drop_also_correct(kept, q, c)

            tries = 0
            while len(kept) < 3 and tries < DISTRACTOR_TOPUP_TRIES:
                tries += 1
                avoid = [d["text"] for d in kept] + [c]
                repl = _generate_one_replacement_gemini(q, c, lang, avoid_texts=avoid)
                if (repl and _norm_key(repl["text"]) not in {_norm_key(d["text"]) for d in kept}
                        and not _distractor_also_valid(q, repl["text"], c)):
                    kept.append(repl)

            if len(kept) >= 3:
                return {"distractors": kept[:3]}

            raise ValueError("Could not obtain 3 valid distractors after replacement.")

        except Exception as e:
            last_err = e

    log(f"[Distractors] Falling back: {last_err}", level="warning")
    return {"distractors": _fallback_make_3_distractors_relaxed(c)}


def build_mcq_from_3_distractors(correct: str, distractors: List[Dict[str, str]], lang: str,
                                 question: str = "") -> Dict[str, Any]:
    # Build the four (text, rationale, tag) entries; the first is the correct one.
    entries = [{"text": correct, "rationale": "", "tag": "", "correct": True}]
    for d in distractors[:3]:
        entries.append({
            "text": d.get("text", ""),
            "rationale": d.get("rationale", ""),
            "tag": d.get("misconception_tag", ""),
            "correct": False,
        })
    while len(entries) < 4:
        entries.append({"text": correct, "rationale": "", "tag": "", "correct": False})

    # Order the four options. If every option is numeric, sort ascending (low→high)
    # so students see a natural progression; otherwise shuffle to avoid position bias.
    nums = [try_num(e["text"]) for e in entries]
    if all(n is not None for n in nums):
        order = sorted(range(len(entries)), key=lambda i: nums[i])
    else:
        order = list(range(len(entries)))
        random.shuffle(order)
    entries = [entries[i] for i in order]

    label_order = ["A", "B", "C", "D"]
    options, rats, tags = {}, {}, {}
    correct_label = "A"
    for lbl, e in zip(label_order, entries):
        options[lbl] = e["text"]
        rats[lbl] = e["rationale"]
        tags[lbl] = e["tag"]
        if e["correct"]:
            correct_label = lbl

    options = enforce_units_on_options(correct, options, question)
    # The displayed answer must match the (now unit-normalized) correct option.
    answer_text = options.get(correct_label, correct)

    if lang.lower() == "arabic":
        options = {k: to_arabic_digits(v) for k, v in options.items()}
        rats = {k: to_arabic_digits(v) for k, v in rats.items()}
        answer_text = to_arabic_digits(answer_text)

    return {
        "OptionA": options.get("A", ""),
        "OptionB": options.get("B", ""),
        "OptionC": options.get("C", ""),
        "OptionD": options.get("D", ""),
        "CorrectOption": correct_label,
        "Answer": answer_text,

        "GenRationale_A": rats.get("A", ""),
        "GenRationale_B": rats.get("B", ""),
        "GenRationale_C": rats.get("C", ""),
        "GenRationale_D": rats.get("D", ""),

        "GenMisTag_A": tags.get("A", ""),
        "GenMisTag_B": tags.get("B", ""),
        "GenMisTag_C": tags.get("C", ""),
        "GenMisTag_D": tags.get("D", ""),
    }


# ============================================================
# VERBAL DISTRACTOR PIPELINE (from G_Distractors.py)
# ============================================================

PROMPT_TEMPLATE = dedent("""
Category: {category}

Instructions:
{category_prompt}

---
Input:

Context:
{context}

Question:
{question}

Correct answer:
{correct_answer}
""").strip()


def norm_verbal(s: str) -> str:
    """Normalize text for verbal distractor comparison."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_PROMPT_LEAK_RE = re.compile(
    r'^(for each distractor|state which|for each option|explain why a student)', re.I)


def _clean_rationale(text: str) -> str:
    """Strip raw dict syntax and prompt-instruction leakage from rationale text."""
    if not text:
        return ""
    text = text.strip()
    # Prompt instruction leak
    if _PROMPT_LEAK_RE.match(text):
        return ""
    # Raw Python/JSON dict: {'key': 'value', ...} — extract string values only
    if text.startswith("{") and re.search(r"['\"]:\s*['\"]", text):
        values = re.findall(r"(?:\"|\')(?:[^\"\']+)(?:\"|\')\s*:\s*(?:\"|\')\s*([^\"\']+)", text)
        if values:
            return ". ".join(v.strip(". ") for v in values)
    return text


def _repair_verbal_question(
    q: Dict[str, Any],
    rejection_reason: str,
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    """
    Attempt one targeted fix of a rejected question.
    Passes the rejection reason back to the generator (Pro) so it revises
    the specific issue rather than starting from scratch.
    Returns a repaired question dict (original fields carried over) or None on failure.
    """
    repair_prompt = (
        f"The following question was generated but REJECTED by an automated validator.\n\n"
        f"Question: {q.get('Question', '')}\n"
        f"Answer: {q.get('Answer', '')}\n"
        f"Rejection reason: {rejection_reason}\n\n"
        "Fix ONLY the specific issue identified above. Keep the same topic, category, "
        "and sub-category. Revise the question/answer so the rejection reason no longer applies.\n\n"
        "Return ONLY this JSON (no markdown):\n"
        "{\n"
        '  "Question": "...",\n'
        '  "Answer": "..."\n'
        "}"
    )
    raw = gemini_call(
        repair_prompt,
        model=GENERATOR_MODEL,
        system_instruction=system_prompt,
        temperature=0.3,
    )
    data = safe_json_load(raw)
    if not data or "error" in data or "Question" not in data or "Answer" not in data:
        return None
    result = dict(q)
    result["Question"] = data["Question"]
    result["Answer"] = data["Answer"]
    return result


def _repair_question_from_solver(
    q: Dict[str, Any],
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    """
    When a question fails majority solver validation, send the question, stated answer,
    and each solver's response back to the generator so it can fix either the wrong
    answer or an ambiguous/unsolvable question.
    Returns a repaired dict (fields carried over, solver metadata cleared) or None.
    """
    stated = q.get("Answer", "")
    solver_answers = q.get("solver_answers", [])
    solver_list = "\n".join(
        f"  - Solver {i + 1}: {s.get('answer', 'ERR')}"
        for i, s in enumerate(solver_answers)
    )

    non_err = [s["answer"] for s in solver_answers
               if not str(s.get("answer", "")).startswith("ERR")]
    if len(non_err) >= 2:
        from collections import Counter
        top_answer, top_count = Counter(non_err).most_common(1)[0]
        if top_count >= 2 and not answers_match(top_answer, stated):
            issue = (
                f"Your stated answer was '{stated}', but {top_count}/3 independent solvers "
                f"computed '{top_answer}'. Either update the stated answer to '{top_answer}', "
                f"or rephrase the question so that '{stated}' is unambiguously correct."
            )
        else:
            issue = (
                f"Your stated answer was '{stated}', but solvers disagreed on the answer. "
                f"The question may be ambiguous or contain an error. Revise it so the answer is clear."
            )
    else:
        issue = (
            f"Your stated answer was '{stated}', but solvers could not compute it. "
            f"Simplify or clarify the question."
        )

    repair_prompt = (
        "The following question was REJECTED because automated solvers disagreed with the stated answer.\n\n"
        f"Question: {q.get('Question', '')}\n"
        f"Stated answer: {stated}\n"
        f"Solver responses:\n{solver_list}\n\n"
        f"Issue: {issue}\n\n"
        "Fix ONLY the identified issue. Keep the same topic, category, and sub-category. "
        "Return ONLY this JSON (no markdown):\n"
        '{"Question": "...", "Answer": "..."}'
    )
    try:
        raw = gemini_call(
            repair_prompt,
            model=GENERATOR_MODEL,
            system_instruction=system_prompt,
            temperature=0.3,
        )
    except Exception:
        return None
    data = safe_json_load(raw)
    if not data or "error" in data or "Question" not in data or "Answer" not in data:
        return None
    result = dict(q)
    result["Question"] = data["Question"]
    result["Answer"] = data["Answer"]
    result["_repaired"] = True
    result.pop("solver_answers", None)
    result.pop("votes", None)
    result.pop("answer_match_majority", None)
    result.pop("similarity_to_corpus", None)
    return result


# ============================================================
# HOLISTIC MCQ CRITIC (batched, flash) + 1-ROUND REGENERATION
# ============================================================

def _format_mcq_for_critic(idx: int, mcq: Dict[str, Any]) -> str:
    """Render one assembled MCQ (merged question + options) for the critic prompt.
    Includes each distractor's stated design intent so the critic judges options
    against what they are SUPPOSED to test."""
    correct = (mcq.get("CorrectOption") or "").strip().upper()
    ctx = clean_text(mcq.get("Context", ""))
    lines = [f"### MCQ {idx}"]
    if ctx:
        lines.append(f"Context: {ctx}")
    lines.append(f"Question: {clean_text(mcq.get('Question', ''))}")
    for L in ("A", "B", "C", "D"):
        opt = clean_text(mcq.get(f"Option{L}", ""))
        mark = "  <-- labeled CORRECT" if L == correct else ""
        intent = clean_text(mcq.get(f"GenRationale_{L}", ""))
        tag = clean_text(mcq.get(f"GenMisTag_{L}", ""))
        meta = ""
        if L != correct and (intent or tag):
            meta = f"   [intended distractor — tag={tag or 'n/a'}; rationale={intent or 'n/a'}]"
        lines.append(f"  {L}) {opt}{mark}{meta}")
    return "\n".join(lines)


def critique_mcq_batch(
    mcqs: List[Dict[str, Any]],
    lang: str,
    section: str,
    category: str,
    subcat: Optional[str],
    batch_size: int = CRITIC_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Judge a list of assembled MCQs (stem + 4 options + correct label) as a whole.
    Returns one verdict dict per input MCQ, order preserved:
        {"pass": bool, "target": "stem"|"options"|"both", "issues": [str, ...]}
    Fail-open: any parse/API error or count mismatch -> every MCQ in that chunk passes.
    """
    if not mcqs:
        return []

    lang_rule = ("All option text and stem must be in Modern Standard Arabic only "
                 "(Arabic script)." if str(lang).lower() == "arabic"
                 else "All text must be in English only.")

    def _judge_chunk(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        blocks = "\n\n".join(_format_mcq_for_critic(i + 1, m) for i, m in enumerate(chunk))
        prompt = f"""You are a strict exam-quality reviewer for GAT (Qudrat) multiple-choice questions.
Section: {section} | Category: {category} | Sub-Category: {subcat or 'ANY'}
{lang_rule}

For EACH MCQ below, check ALL of:
1. Exactly ONE option is correct, and it is the one labeled CORRECT.
2. No distractor equals or is equivalent (numerically or semantically) to the correct answer.
3. Each distractor is plausible AND traceable to a real mistake matching its stated tag/rationale — not random or absurd.
4. There is NO second defensible-correct option (watch verbal synonyms and dual-blank sentence completion).
5. The stem is self-contained, unambiguous, and answerable without seeing the options.
6. Options are homogeneous in form, length, units, and language/script.

For each MCQ return a verdict. If everything is fine, pass=true. Otherwise pass=false,
set target to "stem" (stem/answer wrong or ambiguous), "options" (distractor problems only),
or "both", and give 1-3 short actionable issue strings.

MCQs:
{blocks}

Return ONLY this JSON (no markdown), one entry per MCQ in order:
{{"verdicts": [{{"mcq": 1, "pass": true, "target": "", "issues": []}}]}}"""
        try:
            raw = gemini_call(
                prompt, model=CRITIC_MODEL,
                response_mime_type="application/json", temperature=0.0,
            )
            data = safe_json_load(raw)
            verdicts = (data or {}).get("verdicts", [])
            if not isinstance(verdicts, list) or len(verdicts) != len(chunk):
                raise ValueError("verdict count mismatch")
            out = []
            for v in verdicts:
                ok = bool(v.get("pass", True))
                tgt = (v.get("target") or "").strip().lower()
                if tgt not in ("stem", "options", "both"):
                    tgt = "options" if not ok else ""
                issues = v.get("issues") or []
                if not isinstance(issues, list):
                    issues = [str(issues)]
                out.append({"pass": ok, "target": tgt,
                            "issues": [str(x) for x in issues][:3]})
            return out
        except Exception as e:
            log(f"[critic] chunk failed, passing open: {e}", level="warning")
            return [{"pass": True, "target": "", "issues": []} for _ in chunk]

    chunks = [mcqs[i:i + batch_size] for i in range(0, len(mcqs), batch_size)]
    results: List[Dict[str, Any]] = []
    if len(chunks) == 1:
        results = _judge_chunk(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=min(len(chunks), _MAX_CONCURRENT_API_CALLS)) as ex:
            for chunk_res in ex.map(_judge_chunk, chunks):
                results.extend(chunk_res)
    return results


def regen_from_critique(
    merged: Dict[str, Any],
    critique: Dict[str, Any],
    lang: str,
    section: str,
    category: str,
    subcat: Optional[str],
    rebuild_mcq,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """One-round fix of a critic-rejected MCQ.

    `rebuild_mcq(question_dict, extra_feedback)` must return a fresh option dict
    (OptionA-D / CorrectOption / GenRationale_* / GenMisTag_*) for the given stem.
    target=stem/both -> repair stem+answer with critic feedback, then rebuild options.
    target=options    -> keep stem, rebuild options with critic feedback.
    Always re-critiques once; tags CriticPass / CriticIssues on the returned dict.
    """
    issues = critique.get("issues") or []
    feedback = "; ".join(str(x) for x in issues) or "unspecified quality issue"
    target = critique.get("target") or "options"
    q = dict(merged)

    if target in ("stem", "both"):
        repair_prompt = (
            "The following question was REJECTED by a quality reviewer.\n\n"
            f"Question: {merged.get('Question', '')}\n"
            f"Stated answer: {merged.get('Answer', '')}\n"
            f"Reviewer issues: {feedback}\n\n"
            "Fix ONLY the identified issue. Keep the same topic, category, and sub-category. "
            "Return ONLY this JSON (no markdown):\n"
            '{"Question": "...", "Answer": "..."}'
        )
        try:
            raw = gemini_call(repair_prompt, model=GENERATOR_MODEL,
                              system_instruction=system_prompt, temperature=0.3)
            data = safe_json_load(raw)
            if data and data.get("Question") and data.get("Answer"):
                q["Question"] = data["Question"]
                q["Answer"] = data["Answer"]
        except Exception as e:
            log(f"[critic-regen] stem repair failed: {e}", level="warning")

    # Rebuild options for the (possibly repaired) stem, passing critic feedback.
    try:
        new_opts = rebuild_mcq(q, feedback)
        if new_opts:
            q.update(new_opts)
    except Exception as e:
        log(f"[critic-regen] option rebuild failed: {e}", level="warning")

    if str(lang).lower() == "arabic":
        q["Question"] = to_arabic_digits(q.get("Question", ""))
        q["Answer"] = to_arabic_digits(q.get("Answer", ""))

    recheck = critique_mcq_batch([q], lang, section, category, subcat)
    v = recheck[0] if recheck else {"pass": True, "issues": []}
    q["CriticPass"] = bool(v.get("pass", True))
    q["CriticIssues"] = "; ".join(str(x) for x in (v.get("issues") or []))
    return q


def load_category_prompts(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Category prompts JSON must be a non-empty object.")
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            prompt_text = v.get("prompt", "")
            if not prompt_text:
                raise ValueError(f"Category '{k}' is missing a 'prompt' key in {path}.")
        else:
            prompt_text = str(v)
        result[str(k).strip()] = str(prompt_text).strip()
    return result


def _get_verbal_distractor_quality_rules(category: str, subcat: str = "") -> str:
    """Return per-subtype distractor quality rules appended to the base prompt."""
    subcat_lower = subcat.strip().lower()

    if category == "Reading Comprehension":
        if "main idea" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — MAIN IDEA distractors:\n"
                "- Each distractor must represent a SPECIFIC DETAIL or supporting point from the passage, "
                "NOT the overall theme.\n"
                "- Distractors should seem like 'almost the main idea' to a student who only remembered part of the passage.\n"
                "- The correct answer encompasses ALL details; each distractor covers only ONE aspect.\n"
                "- Do NOT create distractors as broad as the correct answer — they must be clearly narrower in scope.\n"
            )
        elif "inference" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — INFERENCE distractors:\n"
                "- Distractors must use information from the passage but draw the WRONG conclusion.\n"
                "- Each distractor should represent a plausible but flawed inference a careless reader might make.\n"
                "- Avoid distractors that directly contradict an explicit passage statement "
                "(too easy to eliminate without reasoning).\n"
            )
        elif "detail" in subcat_lower or "fact" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — DETAIL/FACT distractors:\n"
                "- Distractors must use subjects and concepts ACTUALLY mentioned in the passage.\n"
                "- Each distractor should misattribute a passage feature to the wrong subject, "
                "or describe a real passage element with an incorrect detail.\n"
                "- Do NOT introduce ideas completely absent from the passage.\n"
                "- Avoid distractors that directly contradict a very clear, memorable passage statement "
                "(students should need to check the passage to rule them out).\n"
            )
        elif "tone" in subcat_lower or "purpose" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — TONE/PURPOSE distractors:\n"
                "- Distractors must name real tones or purposes that partially fit the passage "
                "but miss the dominant intent.\n"
                "- Each distractor should match ONE aspect of the passage while misrepresenting the overall tone.\n"
                "- Do NOT use tones that are the complete opposite of the passage's actual tone "
                "(too easy to eliminate without reading carefully).\n"
                "- EXAMPLE: if the passage is primarily informative, a distractor of 'persuasive' is plausible; "
                "'humorous' or 'satirical' is not.\n"
            )
        elif "vocabulary" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — VOCABULARY-IN-CONTEXT distractors:\n"
                "- Distractors must be real words that are valid synonyms or near-synonyms of the target word "
                "in OTHER contexts, but wrong in this specific passage context.\n"
                "- Do NOT use words from entirely unrelated semantic fields.\n"
                "- A student who knows the word's most common meaning (not its contextual meaning) "
                "should be attracted to the wrong distractor.\n"
            )
        elif "dual" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — DUAL BLANK distractors:\n"
                "- Each distractor must be a 'word \u2013 word' pair using a space, en dash (\u2013), space separator.\n"
                "- Both words in each distractor pair must belong to the same semantic domain as the sentence.\n"
                "- Each distractor pair must be plausible but wrong when the sentence is read carefully.\n"
                "- Do NOT use comma-separated pairs \u2014 use ONLY the en-dash format: 'word1 \u2013 word2'.\n"
            )
        elif "logical organization" in subcat_lower:
            return (
                "\nSUB-TYPE RULES — LOGICAL ORGANIZATION distractors:\n"
                "- Distractors must describe real structural features of the passage "
                "(e.g., a real comparison or contrast that exists) but misstate its purpose or placement.\n"
                "- Do NOT describe structures completely absent from the passage.\n"
                "- Each distractor should sound like a plausible description of how a passage could be organized.\n"
            )
    return ""


def build_prompt(question: str, correct: str, context: str, category: str, category_prompt: str,
                 quality_rules: str = "") -> str:
    base = PROMPT_TEMPLATE.format(
        category=category,
        category_prompt=category_prompt,
        context=(context or "").strip(),
        question=(question or "").strip(),
        correct_answer=(correct or "").strip(),
    )
    if quality_rules:
        base += f"\n\n{quality_rules.strip()}"
    return base


def _parse_verbal_distractors_soft(raw_json: str, correct_text: str) -> Tuple[List[str], str, bool]:
    """Parse distractor JSON and return however many valid distractors were found (1–3).
    Does NOT raise if count < 3 — caller handles topup."""
    data = json.loads(raw_json)

    if "error" in data:
        raise ValueError(f"Model returned an error: {data['error']}")

    if data.get("skip") is True:
        return [], str(data.get("notes", "")), True

    cultural_check = str(data.get("cultural_check", "Pass"))
    if cultural_check.lower().startswith("fail"):
        log(f"[CULTURAL WARNING] Model flagged: {cultural_check}", level="warning")

    d = data.get("distractors", [])
    if not isinstance(d, list) or len(d) == 0:
        raise ValueError("Model returned no distractors.")

    cleaned = []
    seen = set()
    c_norm = norm_verbal(correct_text)

    for x in d:
        if isinstance(x, dict):
            x = x.get("text") or x.get("pair") or x.get("distractor") or ""
        x = str(x).strip()
        if not x:
            continue
        xn = norm_verbal(x)
        if xn == c_norm:
            continue
        if xn in seen:
            continue
        seen.add(xn)
        cleaned.append(x)

    if not cleaned:
        raise ValueError("All distractors were duplicates or matched the correct answer.")

    return cleaned, str(data.get("notes", "")), False


def parse_and_validate(raw_json: str, correct_text: str) -> Tuple[List[str], str, bool]:
    """Strict version — raises if count != 3. Kept for compatibility."""
    cleaned, notes, skip = _parse_verbal_distractors_soft(raw_json, correct_text)
    if not skip and len(cleaned) != 3:
        raise ValueError(f"Distractors invalid after cleaning: got {len(cleaned)}, expected 3.")
    return cleaned, notes, skip


def _generate_one_verbal_replacement(
    question: str,
    correct_text: str,
    context: str,
    category: str,
    lang: str,
    avoid_texts: List[str],
) -> Optional[str]:
    """Generate a single verbal distractor to top up when batch returns fewer than 3."""
    avoid_block = "\n".join(f"- {t}" for t in avoid_texts[:10]) if avoid_texts else "- (none)"
    lang_hint = (
        "Write in Modern Standard Arabic only." if lang.lower() == "arabic"
        else "Write in English only."
    )
    ctx_block = f"Passage:\n{context}\n\n" if context else ""

    prompt = f"""You are generating ONE distractor for a GAT (Qudrat) verbal {category} question.

{ctx_block}Question: {question}

Correct answer (must NOT be produced): {correct_text}

Already-used distractors (must NOT repeat):
{avoid_block}

Rules:
- Output exactly ONE plausible-but-wrong distractor.
- It must be clearly wrong when the question is understood, but seem reasonable at first glance.
- Must NOT equal the correct answer or any already-used distractor.
- Match the correct answer in grammatical form and approximate length.
- {lang_hint}

Return JSON ONLY:
{{
  "distractor": "..."
}}
"""
    try:
        raw = call_model(prompt)
        data = safe_json_load(raw)
        if not data:
            return None
        text = str(data.get("distractor", "")).strip()
        if not text or norm_verbal(text) == norm_verbal(correct_text):
            return None
        if any(norm_verbal(text) == norm_verbal(a) for a in avoid_texts):
            return None
        return text
    except Exception:
        return None


def generate_verbal_distractors(
    question: str,
    correct_text: str,
    context: str,
    category: str,
    prompts_dict: dict,
    subcat: str = "",
    lang: str = "English",
    extra_feedback: str = "",
) -> Dict[str, Any]:
    """Generate verbal distractors using category-specific prompts."""
    category_prompt = prompts_dict.get(category)
    if category_prompt is None:
        return {"distractors": [], "notes": f"No prompt for category: '{category}'", "skip": True}

    # For RC with no context, use question text as context
    if not context and category == "Reading Comprehension":
        context = question
        log("[Verbal] No context for RC question. Using question text as context.", level="warning")

    quality_rules = _get_verbal_distractor_quality_rules(category, subcat)
    lang_instruction = (
        "\nIMPORTANT: Generate all distractors in Modern Standard Arabic only."
        if lang.lower() == "arabic"
        else "\nIMPORTANT: Generate all distractors in English only."
    )
    feedback_block = (
        f"\n\nAVOID these problems flagged in a previous attempt:\n- {extra_feedback}"
        if extra_feedback else ""
    )
    prompt = build_prompt(question, correct_text, context, category, category_prompt,
                          quality_rules + lang_instruction + feedback_block)

    last_notes = ""
    for attempt in range(DISTRACTOR_MAX_RETRIES + 1):
        try:
            raw = call_model(prompt)
            distractors_list, notes, skip = _parse_verbal_distractors_soft(raw, correct_text)
            last_notes = notes
            if skip:
                log(f"[VerbalDistractor] Model skipped (attempt {attempt + 1}): {notes}", level="warning")
                continue  # retry — don't give up on first skip

            # Top up to 3 if batch returned fewer
            topup_tries = 0
            avoid = list(distractors_list) + [correct_text]
            while len(distractors_list) < 3 and topup_tries < DISTRACTOR_TOPUP_TRIES:
                topup_tries += 1
                repl = _generate_one_verbal_replacement(
                    question, correct_text, context, category, lang, avoid_texts=avoid
                )
                if repl and norm_verbal(repl) not in {norm_verbal(d) for d in distractors_list}:
                    distractors_list.append(repl)
                    avoid.append(repl)

            if len(distractors_list) < 3:
                log(
                    f"[VerbalDistractor] Only {len(distractors_list)} distractors after topup "
                    f"(attempt {attempt + 1}). Retrying batch.",
                    level="warning",
                )
                continue

            # Split notes into per-distractor rationales
            note_sentences = re.split(r'(?<=[.!?])\s+', notes.strip())
            distractors = []
            for i, dist_text in enumerate(distractors_list[:3]):
                rat = _clean_rationale(note_sentences[i] if i < len(note_sentences) else notes)
                distractors.append({"text": dist_text, "rationale": rat})

            return {"distractors": distractors, "notes": notes, "skip": False}
        except Exception as e:
            log(f"[VerbalDistractor] Attempt {attempt + 1} failed: {e}", level="warning")

    return {"distractors": [], "notes": last_notes or "All attempts failed.", "skip": True}


def _verbal_options_ok(q: Dict[str, Any]) -> bool:
    """True iff the row has 4 real, distinct options (no placeholder)."""
    vals = [clean_text(q.get(f"Option{L}", "")) for L in ("A", "B", "C", "D")]
    if any(not v or v == "[Option not available]" for v in vals):
        return False
    return len({norm_verbal(v) for v in vals}) == 4


def build_verbal_mcq_from_distractors(
    correct: str, distractors: List[Dict[str, str]], lang: str
) -> Dict[str, Any]:
    """Build MCQ dict from verbal distractors (no unit enforcement, no Arabic digit conversion)."""
    labels = ["A", "B", "C", "D"]
    random.shuffle(labels)

    correct_label = labels[0]
    dist_labels = labels[1:]

    options = {correct_label: correct}
    rats = {correct_label: ""}

    for lbl, d in zip(dist_labels, distractors):
        options[lbl] = d.get("text", "[Option not available]")
        rats[lbl] = d.get("rationale", "")

    # Handle < 3 distractors gracefully
    for lbl in dist_labels:
        if lbl not in options:
            options[lbl] = "[Option not available]"
            rats[lbl] = ""

    for lbl in ["A", "B", "C", "D"]:
        options.setdefault(lbl, "[Option not available]")
        rats.setdefault(lbl, "")

    # Complete only if all 4 options are real (no placeholder) and distinct.
    vals = [clean_text(options[L]) for L in ("A", "B", "C", "D")]
    complete = (all(v and v != "[Option not available]" for v in vals)
                and len({norm_verbal(v) for v in vals}) == 4)

    return {
        "OptionA": options.get("A", ""),
        "OptionB": options.get("B", ""),
        "OptionC": options.get("C", ""),
        "OptionD": options.get("D", ""),
        "CorrectOption": correct_label,
        "OptionsComplete": complete,

        "GenRationale_A": rats.get("A", ""),
        "GenRationale_B": rats.get("B", ""),
        "GenRationale_C": rats.get("C", ""),
        "GenRationale_D": rats.get("D", ""),

        "GenMisTag_A": "",
        "GenMisTag_B": "",
        "GenMisTag_C": "",
        "GenMisTag_D": "",
    }


# ============================================================
# QUANTITATIVE QUESTION GENERATION PIPELINE
# ============================================================

def generate_valid_quant_mcq_questions(
    df_all: pd.DataFrame,
    target_language: str,
    category: str,
    subcat: Optional[str],
    num_required: int,
    seen_questions: Optional[set] = None,
    extra_gen_feedback: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    df_lang = df_all[df_all["Language"] == target_language]
    df_quant = df_lang[df_lang["Section"].astype(str).str.strip().str.lower() == "quantitative"]
    if df_quant.empty:
        raise RuntimeError(f"No quantitative data found for language '{target_language}'.")

    rules = build_rules(df_quant)

    ex_df = df_quant.copy()
    if category != "ANY":
        ex_df = ex_df[ex_df["Category"] == category]
    if subcat and subcat != "ANY":
        ex_df = ex_df[ex_df["Sub-Category"] == subcat]
    if ex_df.empty:
        raise RuntimeError("No example questions for selected filters.")

    system_prompt = system_prompt_quant(target_language)

    if seen_questions is None:
        seen_questions = set()

    # Category-scoped corpus reduces false positives from unrelated sub-categories
    corpus_df = df_quant.copy()
    if category != "ANY":
        corpus_df = corpus_df[corpus_df["Category"] == category]
    corpus_questions = corpus_df["Question"].astype(str).tolist()
    corpus_tokens = [norm_tokens(q) for q in corpus_questions]

    # Valid sub-categories for the selected category — injected into prompt so the model
    # uses exact names and the fuzzy auto-correct rarely needs to fire.
    valid_subcats: List[str] = []
    if category != "ANY":
        valid_subcats = sorted(
            [s for s in df_quant[df_quant["Category"] == category]["Sub-Category"].unique()
             if s and s.lower() not in ("nan", "none", "")]
        )

    validated_pool: List[Dict[str, Any]] = []
    structurally_rejected: set = set()  # Track question texts rejected for structure errors
    solver_rejected: set = set()         # Track question texts that failed majority vote
    round_idx = 0

    # Adaptive oversample: more aggressive when few examples exist
    n_examples = len(ex_df)
    oversample = GEN_OVERSAMPLE_FACTOR
    if n_examples < 5:
        oversample = max(oversample, 5.0)
    elif n_examples < 10:
        oversample = max(oversample, 3.0)
    if category == "Arithmetic" and subcat and "Word" in str(subcat):
        oversample = max(oversample, 2.5)
    if category.startswith("Data Interpretation") and subcat and "Comparison" in str(subcat):
        oversample = max(oversample, 4.0)

    while len(validated_pool) < num_required and round_idx < MAX_GENERATION_ROUNDS:
        round_idx += 1
        remaining = num_required - len(validated_pool)
        batch_size = max(int(remaining * oversample), remaining + 2)

        # Resample examples each round for variety
        examples = ex_df.sample(
            min(NUM_EXAMPLES_IN_PROMPT, len(ex_df)),
            random_state=random.randint(0, 999999)
        ).to_dict(orient="records")
        examples_block = format_examples(examples)

        user_msg = user_prompt(examples_block, batch_size, "Quantitative", category, subcat,
                               valid_subcats=valid_subcats)
        if extra_gen_feedback:
            user_msg += ("\n\nIMPORTANT — fix issues from previous attempts:\n"
                         f"{extra_gen_feedback}")

        try:
            raw = gemini_call(
                user_msg,
                model=GENERATOR_MODEL,
                system_instruction=system_prompt,
                temperature=0.4,
            )
            data = safe_json_load(raw)
            if not data:
                continue
            gen_questions = data.get("questions", [])
        except Exception as e:
            log(f"Gemini generation error: {e}", level="error")
            continue

        structural_ok = []
        for q in gen_questions:
            q_text = clean_text(q.get("Question", ""))
            if q_text in structurally_rejected or q_text in solver_rejected:
                continue
            _autocorrect_subcat(q, rules)  # fix minor naming differences before validation
            errs = validate_structure(q, rules, "Quantitative", category, subcat)
            if not errs:
                structural_ok.append(q)
            else:
                if q_text:
                    structurally_rejected.add(q_text)

        filtered = []
        for q in structural_ok:
            q_text = clean_text(q["Question"])
            if q_text in seen_questions:
                continue
            if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                continue

            sim_to_corpus = max_similarity_to_corpus(q_text, corpus_tokens)
            q["similarity_to_corpus"] = sim_to_corpus
            if sim_to_corpus >= DIVERSITY_THRESHOLD_CORPUS:
                continue

            filtered.append(q)

        need_now = num_required - len(validated_pool)
        to_solve = filtered[: min(len(filtered), need_now * 3)]
        to_solve = to_solve[:MAX_SOLVER_CALLS_PER_ROUND]
        majority_checked = majority_validate(to_solve)

        repair_queue: List[Dict[str, Any]] = []
        for q in majority_checked:
            votes = q.get("votes", 0)
            majority_ok = q.get("answer_match_majority", False)
            q_text = clean_text(q["Question"])

            if not majority_ok and round_idx < MAX_GENERATION_ROUNDS:
                if not q.get("_repaired"):
                    repair_queue.append(q)  # send back to generator with solver feedback
                else:
                    solver_rejected.add(q_text)  # already repaired once, discard
                continue
            if not majority_ok and votes <= 0:
                solver_rejected.add(q_text)
                continue

            if q_text in seen_questions:
                continue
            if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                continue

            validated_pool.append(q)
            seen_questions.add(q_text)

            if len(validated_pool) >= num_required:
                break

        # Repair solver-rejected questions: send back to generator with solver disagreement
        if repair_queue and len(validated_pool) < num_required:
            need_now = num_required - len(validated_pool)
            to_repair = repair_queue[:need_now * 2]
            repaired_candidates: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(len(to_repair), _MAX_CONCURRENT_API_CALLS)) as ex:
                futures = [ex.submit(_repair_question_from_solver, q, system_prompt) for q in to_repair]
                for fut in futures:
                    repaired = fut.result()
                    if not repaired:
                        continue
                    rep_text = clean_text(repaired.get("Question", ""))
                    if not rep_text or rep_text in structurally_rejected or rep_text in solver_rejected:
                        continue
                    if any(too_similar(rep_text, v["Question"]) for v in validated_pool):
                        continue
                    sim = max_similarity_to_corpus(rep_text, corpus_tokens)
                    if sim >= DIVERSITY_THRESHOLD_CORPUS:
                        continue
                    repaired["similarity_to_corpus"] = sim
                    repaired_candidates.append(repaired)

            if repaired_candidates:
                re_checked = majority_validate(repaired_candidates[:need_now * 2])
                for q in re_checked:
                    if not q.get("answer_match_majority"):
                        solver_rejected.add(clean_text(q["Question"]))
                        continue
                    q_text = clean_text(q["Question"])
                    if q_text in seen_questions:
                        continue
                    if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                        continue
                    validated_pool.append(q)
                    seen_questions.add(q_text)
                    if len(validated_pool) >= num_required:
                        break

    final_questions: List[Dict[str, Any]] = []
    export_rows: List[Dict[str, Any]] = []

    def _build_quant_mcq_for_q(q):
        if is_compare_subcategory(q.get("Sub-Category", "")):
            return build_compare_mcq(q.get("Answer", ""), target_language)
        dist_pack = generate_3_distractors_gemini(q["Question"], q["Answer"], target_language)
        return build_mcq_from_3_distractors(
            correct=clean_text(q["Answer"]),
            distractors=dist_pack["distractors"],
            lang=target_language,
            question=q.get("Question", ""),
        )

    def _rebuild_quant_mcq(qd, extra_feedback):
        if is_compare_subcategory(qd.get("Sub-Category", "")):
            return build_compare_mcq(qd.get("Answer", ""), target_language)
        dist_pack = generate_3_distractors_gemini(
            qd["Question"], qd["Answer"], target_language, extra_feedback=extra_feedback)
        return build_mcq_from_3_distractors(
            correct=clean_text(qd["Answer"]),
            distractors=dist_pack["distractors"],
            lang=target_language,
            question=qd.get("Question", ""),
        )

    _quant_pool = validated_pool[:num_required]
    if not _quant_pool:
        return [], []
    with ThreadPoolExecutor(max_workers=min(len(_quant_pool), 3)) as executor:
        _quant_mcq_list = list(executor.map(_build_quant_mcq_for_q, _quant_pool))

    merged_list = [{**dict(q), **mcq} for q, mcq in zip(_quant_pool, _quant_mcq_list)]

    # Holistic critic on the assembled MCQs; 1-round regeneration for failures.
    verdicts = critique_mcq_batch(merged_list, target_language, "Quantitative",
                                  category, subcat)
    # Deterministic homogeneity gate: OR a forced "options" failure into the critic
    # verdict when the 4 options mix unit presence or symbolic notation. This catches
    # format leaks (Q14/Q58) that a weak LLM critic misses.
    for i, q2 in enumerate(merged_list):
        opts = {L: q2.get(f"Option{L}", "") for L in ("A", "B", "C", "D")}
        issues = []
        if not options_unit_homogeneous(opts):
            issues.append("options mix unit presence (some carry a unit, some don't)")
        if not options_notation_homogeneous(opts):
            issues.append("options mix symbolic notation (pi/√/fraction) with bare numbers")
        if issues:
            v = verdicts[i]
            v["pass"] = False
            v["target"] = "options"
            v["issues"] = list(v.get("issues", [])) + issues
    fail_idx = [i for i, v in enumerate(verdicts) if not v.get("pass", True)]
    if fail_idx:
        log(f"[critic] Quant: {len(fail_idx)}/{len(merged_list)} failed, regenerating")
        with ThreadPoolExecutor(max_workers=min(len(fail_idx), _MAX_CONCURRENT_API_CALLS)) as ex:
            regened = list(ex.map(
                lambda i: regen_from_critique(
                    merged_list[i], verdicts[i], target_language, "Quantitative",
                    category, subcat, _rebuild_quant_mcq, system_prompt),
                fail_idx))
        for i, r in zip(fail_idx, regened):
            merged_list[i] = r
    for i, v in enumerate(verdicts):
        if i not in fail_idx:
            merged_list[i].setdefault("CriticPass", True)
            merged_list[i].setdefault("CriticIssues", "")

    for q2 in merged_list:
        if target_language.lower() == "arabic":
            q2["Question"] = to_arabic_digits(q2["Question"])
            q2["Answer"] = to_arabic_digits(q2["Answer"])

        final_questions.append(q2)

        export_rows.append({
            "Question": q2.get("Question", ""),
            "Answer": q2.get("Answer", ""),
            "Context": "",
            "CorrectOption": q2.get("CorrectOption", ""),
            "OptionA": q2.get("OptionA", ""),
            "OptionB": q2.get("OptionB", ""),
            "OptionC": q2.get("OptionC", ""),
            "OptionD": q2.get("OptionD", ""),
            "Section": q2.get("Section", ""),
            "Category": q2.get("Category", ""),
            "Sub-Category": q2.get("Sub-Category", ""),
            "Complexity-Level": q2.get("Complexity-Level", ""),
            "Similarity-to-Corpus": q2.get("similarity_to_corpus", None),
            "Votes": q2.get("votes", None),

            "GenRationale_A": q2.get("GenRationale_A", ""),
            "GenRationale_B": q2.get("GenRationale_B", ""),
            "GenRationale_C": q2.get("GenRationale_C", ""),
            "GenRationale_D": q2.get("GenRationale_D", ""),
            "GenMisTag_A": q2.get("GenMisTag_A", ""),
            "GenMisTag_B": q2.get("GenMisTag_B", ""),
            "GenMisTag_C": q2.get("GenMisTag_C", ""),
            "GenMisTag_D": q2.get("GenMisTag_D", ""),

            "CriticPass": q2.get("CriticPass", True),
            "CriticIssues": q2.get("CriticIssues", ""),
        })

    return final_questions, export_rows


# ============================================================
# VERBAL QUESTION GENERATION PIPELINE
# ============================================================

def generate_valid_verbal_mcq_questions(
    df_all: pd.DataFrame,
    target_language: str,
    category: str,
    subcat: Optional[str],
    num_required: int,
    prompts_dict: dict,
    seen_questions: Optional[set] = None,
    extra_gen_feedback: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    df_lang = df_all[df_all["Language"] == target_language]
    df_verbal = df_lang[df_lang["Section"].astype(str).str.strip().str.lower() == "verbal"]

    # Filter to allowed verbal categories only (excludes Vocabulary-in-Context)
    if not df_verbal.empty:
        df_verbal = df_verbal[df_verbal["Category"].isin(ALLOWED_VERBAL_CATS)]

    if df_verbal.empty:
        raise RuntimeError(
            f"No verbal data found for language '{target_language}' with allowed categories. "
            f"Allowed: {ALLOWED_VERBAL_CATS}"
        )

    rules = build_rules(df_verbal)

    ex_df = df_verbal.copy()
    if category != "ANY":
        ex_df = ex_df[ex_df["Category"] == category]
    if subcat and subcat != "ANY":
        ex_df = ex_df[ex_df["Sub-Category"] == subcat]
    if ex_df.empty:
        raise RuntimeError("No example questions for selected verbal filters.")

    system_prompt = system_prompt_verbal(target_language, category, subcat)

    if seen_questions is None:
        seen_questions = set()

    # Category-scoped corpus reduces false positives from unrelated sub-categories.
    # For Analogies, also scope to sub-category: short 2-word stems share function/topic
    # words across sub-categories (e.g. "قطع" appears in Function/Use AND Cause–Effect),
    # causing high Jaccard false-positives that kill too many valid questions.
    corpus_df = df_verbal.copy()
    if category != "ANY":
        corpus_df = corpus_df[corpus_df["Category"] == category]
    if category == "Analogies & Word Relationships" and subcat:
        corpus_df = corpus_df[corpus_df["Sub-Category"] == subcat]
    corpus_questions = corpus_df["Question"].astype(str).tolist()
    corpus_tokens = [norm_tokens(q) for q in corpus_questions]

    # Verbal questions share more vocabulary than quant — use a relaxed threshold.
    # Analogies use an even higher threshold: 2-word stems have very few tokens so
    # Jaccard is unreliable at 0.60 — two different pairs sharing one word = 0.5 Jaccard.
    VERBAL_CORPUS_THRESHOLD = 0.60
    if category == "Analogies & Word Relationships":
        VERBAL_CORPUS_THRESHOLD = 0.80

    # Adaptive oversample: more aggressive when few examples exist
    n_examples = len(ex_df)
    VERBAL_OVERSAMPLE = 3.0
    if n_examples < 5:
        VERBAL_OVERSAMPLE = 6.0
    elif n_examples < 10:
        VERBAL_OVERSAMPLE = 4.0
    if category == "Analogies & Word Relationships":
        VERBAL_OVERSAMPLE = max(VERBAL_OVERSAMPLE, 8.0)

    validated_pool: List[Dict[str, Any]] = []
    structural_fallback: List[Dict[str, Any]] = []  # structurally valid, 0 votes (RC only)
    seen_contexts: set = set()           # deduplicate RC passages across the batch
    structurally_rejected: set = set()   # track question texts rejected for structure errors
    solver_rejected: set = set()         # track SC/Analogies texts that failed majority vote
    rejection_log: List[Tuple[str, str]] = []  # (question_text[:80], rejection_reason)
    round_idx = 0

    while len(validated_pool) < num_required and round_idx < MAX_GENERATION_ROUNDS:
        round_idx += 1
        remaining = num_required - len(validated_pool)
        batch_size = max(int(remaining * VERBAL_OVERSAMPLE), remaining + 3)

        # Resample examples each round for variety
        examples = ex_df.sample(
            min(NUM_EXAMPLES_IN_PROMPT, len(ex_df)),
            random_state=random.randint(0, 999999)
        ).to_dict(orient="records")
        examples_block = format_verbal_examples(examples, category)

        notes = ""
        if round_idx > 1 and rejection_log:
            recent = rejection_log[-5:]
            notes = "\n".join(f'- "{t}" → {r}' for t, r in recent)
        user_msg = user_prompt_verbal(examples_block, batch_size, category, subcat,
                                      rejection_notes=notes)
        if extra_gen_feedback:
            user_msg += ("\n\nIMPORTANT — fix issues from previous attempts:\n"
                         f"{extra_gen_feedback}")

        try:
            raw = gemini_call(
                user_msg,
                model=GENERATOR_MODEL,
                system_instruction=system_prompt,
                temperature=0.4,
            )
            data = safe_json_load(raw)
            if not data:
                continue
            gen_questions = data.get("questions", [])
        except Exception as e:
            log(f"Verbal generation error: {e}", level="error")
            continue

        structural_ok = []
        for q in gen_questions:
            q_text = clean_text(q.get("Question", ""))
            if q_text in structurally_rejected or q_text in solver_rejected:
                continue
            errs = validate_verbal_structure(q, rules, category, subcat)
            if not errs:
                structural_ok.append(q)
            else:
                if q_text:
                    structurally_rejected.add(q_text)
                log(f"[VerbalStructure] Rejected: {errs}", level="warning")

        filtered = []
        for q in structural_ok:
            q_text = clean_text(q["Question"])
            if q_text in seen_questions:
                continue
            if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                continue

            # Fix 6: skip RC questions that reuse an already-seen passage
            if category == "Reading Comprehension":
                ctx = clean_text(q.get("Context", ""))
                if ctx and ctx in seen_contexts:
                    continue

            sim_to_corpus = max_similarity_to_corpus(q_text, corpus_tokens)
            q["similarity_to_corpus"] = sim_to_corpus
            if sim_to_corpus >= VERBAL_CORPUS_THRESHOLD:
                continue

            # SC: reject questions where near-synonyms are equally valid
            # (Logical Connector sub-type is exempt — see validate_sc_unambiguous docstring)
            if category == "Sentence Completion":
                is_valid, reason = validate_sc_unambiguous(
                    q.get("Question", ""), q.get("Answer", ""),
                    target_language, subcat=clean_text(q.get("Sub-Category", "")),
                )
                if not is_valid:
                    rejection_log.append((q_text[:80], reason))
                    # LOOSENED: try a repair, but if it still reads as ambiguous keep the
                    # best-effort question instead of discarding (avoids rejecting ~2/3 of
                    # candidates, which starved the verbal pool). Quality trade-off accepted.
                    repaired = _repair_verbal_question(q, reason, system_prompt)
                    if repaired:
                        rep_text = clean_text(repaired.get("Question", ""))
                        if rep_text and rep_text not in structurally_rejected:
                            is_valid2, _ = validate_sc_unambiguous(
                                repaired.get("Question", ""), repaired.get("Answer", ""),
                                target_language, subcat=clean_text(q.get("Sub-Category", "")),
                            )
                            q = repaired
                            q_text = rep_text
                            if is_valid2:
                                log(f"[SC Repair] Accepted repaired question: '{q_text[:60]}'", level="info")
                            else:
                                log(f"[SC Ambiguity] Kept repaired best-effort: '{q_text[:60]}'", level="info")
                        # else: repaired text empty/dup -> fall through and keep original q
                    # no repaired object -> keep original q (best effort)

            # Analogies: reject questions with weak, wrong-direction, or non-unique relationships
            elif category == "Analogies & Word Relationships":
                is_valid, reason = validate_analogy_relationship(
                    q.get("Question", ""), q.get("Answer", ""),
                    clean_text(q.get("Sub-Category", "")), target_language,
                )
                if not is_valid:
                    rejection_log.append((q_text[:80], reason))
                    repaired = _repair_verbal_question(q, reason, system_prompt)
                    if repaired:
                        rep_text = clean_text(repaired.get("Question", ""))
                        if rep_text and rep_text not in structurally_rejected:
                            is_valid2, _ = validate_analogy_relationship(
                                repaired.get("Question", ""), repaired.get("Answer", ""),
                                clean_text(q.get("Sub-Category", "")), target_language,
                            )
                            if is_valid2:
                                q = repaired
                                q_text = rep_text
                                log(f"[Analogy Repair] Accepted repaired question: '{q_text[:60]}'", level="info")
                            else:
                                log(f"[Analogy] Rejected (repair also failed): '{q_text[:60]}'", level="warning")
                                structurally_rejected.add(q_text)
                                continue
                        else:
                            structurally_rejected.add(q_text)
                            continue
                    else:
                        log(f"[Analogy] Rejected (no repair): '{q_text[:60]}'", level="warning")
                        structurally_rejected.add(q_text)
                        continue

            # RC: reject questions that violate sub-type requirements or have non-best answers
            elif category == "Reading Comprehension":
                ctx = clean_text(q.get("Context", ""))
                if not validate_rc_question(
                    ctx,
                    q.get("Question", ""),
                    q.get("Answer", ""),
                    clean_text(q.get("Sub-Category", "")),
                    target_language,
                ):
                    log(f"[RC] Rejected: fails sub-type requirements — '{q.get('Question', '')[:60]}'", level="warning")
                    structurally_rejected.add(q_text)
                    continue

            filtered.append(q)

        need_now = num_required - len(validated_pool)
        to_solve = filtered[: min(len(filtered), need_now * 3)]
        to_solve = to_solve[:MAX_SOLVER_CALLS_PER_ROUND]
        majority_checked = majority_validate_verbal(to_solve)

        sc_repair_queue: List[Dict[str, Any]] = []
        for q in majority_checked:
            votes = q.get("votes", 0)
            majority_ok = q.get("answer_match_majority", False)
            q_text = clean_text(q["Question"])
            q_cat = clean_text(q.get("Category", ""))

            if q_text in seen_questions:
                continue
            if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                continue

            # RC: solver paraphrase often differs from the exact answer → accept ≥ 1 vote.
            # Analogies: solver independently generates a new valid pair that rarely matches
            # the generator's exact pair → majority agreement is unachievable; accept ≥ 1 vote.
            # SC: answers are unambiguous → require majority; fall back to
            # 1-vote only on the last round (same logic as the quant pipeline).
            if q_cat in ("Reading Comprehension", "Analogies & Word Relationships"):
                accept = majority_ok or votes >= 1
            else:
                if not majority_ok and round_idx < MAX_GENERATION_ROUNDS:
                    if not q.get("_repaired"):
                        sc_repair_queue.append(q)  # send back to generator with solver feedback
                    else:
                        solver_rejected.add(q_text)  # already repaired once, discard
                    accept = False
                elif not majority_ok and votes <= 0:
                    solver_rejected.add(q_text)
                    accept = False
                else:
                    accept = True

            if accept:
                validated_pool.append(q)
                seen_questions.add(q_text)
                # Register the RC passage so it isn't reused by another question
                if q_cat == "Reading Comprehension":
                    ctx = clean_text(q.get("Context", ""))
                    if ctx:
                        seen_contexts.add(ctx)
                # Remove from fallback if it was added there
                structural_fallback = [f for f in structural_fallback if clean_text(f["Question"]) != q_text]
            elif votes == 0:
                # Keep as fallback candidate only for RC (structurally valid, model generated it)
                if q_cat == "Reading Comprehension":
                    if not any(clean_text(f["Question"]) == q_text for f in structural_fallback):
                        structural_fallback.append(q)

            if len(validated_pool) >= num_required:
                break

        # Repair solver-rejected SC questions: send back to generator with solver disagreement
        if sc_repair_queue and len(validated_pool) < num_required:
            need_now = num_required - len(validated_pool)
            to_repair = sc_repair_queue[:need_now * 2]
            sc_repaired_candidates: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(len(to_repair), _MAX_CONCURRENT_API_CALLS)) as ex:
                futures = [ex.submit(_repair_question_from_solver, q, system_prompt) for q in to_repair]
                for fut in futures:
                    repaired = fut.result()
                    if not repaired:
                        continue
                    rep_text = clean_text(repaired.get("Question", ""))
                    if not rep_text or rep_text in structurally_rejected or rep_text in solver_rejected:
                        continue
                    if any(too_similar(rep_text, v["Question"]) for v in validated_pool):
                        continue
                    sim = max_similarity_to_corpus(rep_text, corpus_tokens)
                    if sim >= VERBAL_CORPUS_THRESHOLD:
                        continue
                    repaired["similarity_to_corpus"] = sim
                    sc_repaired_candidates.append(repaired)

            if sc_repaired_candidates:
                re_checked = majority_validate_verbal(sc_repaired_candidates[:need_now * 2])
                for q in re_checked:
                    if not q.get("answer_match_majority"):
                        solver_rejected.add(clean_text(q["Question"]))
                        continue
                    q_text = clean_text(q["Question"])
                    if q_text in seen_questions:
                        continue
                    if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                        continue
                    validated_pool.append(q)
                    seen_questions.add(q_text)
                    if len(validated_pool) >= num_required:
                        break

    # Use structural fallback ONLY for Reading Comprehension (RC answers are harder for solver
    # to verify exactly). SC and Analogies have clear right/wrong answers — don't fall back.
    use_fallback = category in ("Reading Comprehension", "ANY")
    if len(validated_pool) < num_required and structural_fallback and use_fallback:
        log(
            f"[Verbal] Using {min(len(structural_fallback), num_required - len(validated_pool))} "
            "structural fallback questions (0 solver votes).",
            level="warning",
        )
        for q in structural_fallback:
            if len(validated_pool) >= num_required:
                break
            q_text = clean_text(q["Question"])
            if q_text in seen_questions:
                continue
            if any(too_similar(q_text, v["Question"]) for v in validated_pool):
                continue
            validated_pool.append(q)
            seen_questions.add(q_text)

    final_questions: List[Dict[str, Any]] = []
    export_rows: List[Dict[str, Any]] = []

    def _build_verbal_mcq_for_q(q):
        dist_result = generate_verbal_distractors(
            question=q["Question"],
            correct_text=q["Answer"],
            context=clean_text(q.get("Context", "")),
            category=clean_text(q.get("Category", "")),
            prompts_dict=prompts_dict,
            subcat=clean_text(q.get("Sub-Category", "")),
            lang=target_language,
        )
        distractors = dist_result["distractors"] if not dist_result.get("skip") and dist_result["distractors"] else []
        return build_verbal_mcq_from_distractors(
            correct=clean_text(q["Answer"]),
            distractors=distractors,
            lang=target_language,
        )

    def _rebuild_verbal_mcq(qd, extra_feedback):
        dist_result = generate_verbal_distractors(
            question=qd["Question"], correct_text=qd["Answer"],
            context=clean_text(qd.get("Context", "")),
            category=clean_text(qd.get("Category", "")),
            prompts_dict=prompts_dict, subcat=clean_text(qd.get("Sub-Category", "")),
            lang=target_language, extra_feedback=extra_feedback,
        )
        distractors = dist_result["distractors"] if not dist_result.get("skip") and dist_result["distractors"] else []
        return build_verbal_mcq_from_distractors(
            correct=clean_text(qd["Answer"]), distractors=distractors, lang=target_language)

    _verbal_pool = validated_pool[:num_required]
    if not _verbal_pool:
        return [], []
    with ThreadPoolExecutor(max_workers=min(len(_verbal_pool), 3)) as executor:
        _verbal_mcq_list = list(executor.map(_build_verbal_mcq_for_q, _verbal_pool))

    merged_list = [{**dict(q), **mcq} for q, mcq in zip(_verbal_pool, _verbal_mcq_list)]

    # Holistic critic on the assembled MCQs; 1-round regeneration for failures.
    verdicts = critique_mcq_batch(merged_list, target_language, "Verbal", category, subcat)
    # Deterministic gate: a question with a placeholder / duplicate option set must
    # never ship — force it into regeneration.
    for i, q2 in enumerate(merged_list):
        if not q2.get("OptionsComplete", True):
            v = verdicts[i]
            v["pass"] = False
            v["target"] = "options"
            v["issues"] = list(v.get("issues", [])) + [
                "fewer than 4 distinct real options (placeholder present)"]
    # Deterministic ambiguity gate: a verbal question where a distractor is ALSO a
    # defensible answer (reviewer's Q18: both ج and د correct) must regenerate.
    if DISTRACTOR_VERIFY:
        with ThreadPoolExecutor(max_workers=min(len(merged_list),
                                                _MAX_CONCURRENT_API_CALLS)) as ex:
            ambiguous = list(ex.map(_verbal_ambiguous, merged_list))
        for i, amb in enumerate(ambiguous):
            if amb:
                v = verdicts[i]
                v["pass"] = False
                v["target"] = "options"
                v["issues"] = list(v.get("issues", [])) + [
                    "more than one option is a defensible correct answer (ambiguous)"]
    fail_idx = [i for i, v in enumerate(verdicts) if not v.get("pass", True)]
    if fail_idx:
        log(f"[critic] Verbal: {len(fail_idx)}/{len(merged_list)} failed, regenerating")
        with ThreadPoolExecutor(max_workers=min(len(fail_idx), _MAX_CONCURRENT_API_CALLS)) as ex:
            regened = list(ex.map(
                lambda i: regen_from_critique(
                    merged_list[i], verdicts[i], target_language, "Verbal",
                    category, subcat, _rebuild_verbal_mcq, system_prompt),
                fail_idx))
        for i, r in zip(fail_idx, regened):
            merged_list[i] = r
    for i in range(len(merged_list)):
        if i not in fail_idx:
            merged_list[i].setdefault("CriticPass", True)
            merged_list[i].setdefault("CriticIssues", "")

    # Never ship a question that still has a placeholder/duplicate option after regen.
    dropped = sum(1 for q2 in merged_list if not _verbal_options_ok(q2))
    if dropped:
        log(f"[Verbal] Dropping {dropped} question(s) with incomplete options after regen",
            level="warning")
    merged_list = [q2 for q2 in merged_list if _verbal_options_ok(q2)]

    for q2 in merged_list:
        final_questions.append(q2)

        export_rows.append({
            "Question": q2.get("Question", ""),
            "Answer": q2.get("Answer", ""),
            "Context": q2.get("Context", ""),
            "CorrectOption": q2.get("CorrectOption", ""),
            "OptionA": q2.get("OptionA", ""),
            "OptionB": q2.get("OptionB", ""),
            "OptionC": q2.get("OptionC", ""),
            "OptionD": q2.get("OptionD", ""),
            "Section": q2.get("Section", ""),
            "Category": q2.get("Category", ""),
            "Sub-Category": q2.get("Sub-Category", ""),
            "Complexity-Level": q2.get("Complexity-Level", ""),
            "Similarity-to-Corpus": q2.get("similarity_to_corpus", None),
            "Votes": q2.get("votes", None),

            "GenRationale_A": q2.get("GenRationale_A", ""),
            "GenRationale_B": q2.get("GenRationale_B", ""),
            "GenRationale_C": q2.get("GenRationale_C", ""),
            "GenRationale_D": q2.get("GenRationale_D", ""),
            "GenMisTag_A": q2.get("GenMisTag_A", ""),
            "GenMisTag_B": q2.get("GenMisTag_B", ""),
            "GenMisTag_C": q2.get("GenMisTag_C", ""),
            "GenMisTag_D": q2.get("GenMisTag_D", ""),

            "CriticPass": q2.get("CriticPass", True),
            "CriticIssues": q2.get("CriticIssues", ""),
        })

    return final_questions, export_rows


# ============================================================
# STREAMLIT QUIZ UI
# ============================================================

def init_quiz_state():
    defaults = {
        "quiz_questions": [],
        "user_answers": {},
        "quiz_started": False,
        "quiz_finished": False,
        "seen_questions": set(),
        "generated_df": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def start_quiz(questions: List[Dict[str, Any]]):
    st.session_state.quiz_questions = questions
    st.session_state.user_answers = {}
    st.session_state.quiz_started = True
    st.session_state.quiz_finished = False

def finish_quiz():
    st.session_state.quiz_finished = True

def apply_rtl_if_arabic(lang: str):
    if lang.lower() == "arabic":
        st.markdown(
            """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
    direction: rtl;
}
[data-testid="stMarkdownContainer"] { text-align: right; }
</style>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(page_title="GAT Question Generator + Quiz", layout="wide")
    init_quiz_state()

    try:
        df_all = load_exam_data(FILE_PATH, SHEETS_TO_LOAD)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.stop()

    # Load category prompts for verbal distractors
    try:
        category_prompts = load_category_prompts(PROMPTS_JSON)
    except Exception as e:
        st.warning(f"Could not load category prompts ({e}). Verbal distractor generation will be limited.")
        category_prompts = {}

    st.sidebar.header("⚙️ Question Setup")

    langs = sorted(df_all["Language"].dropna().astype(str).unique().tolist())
    lang = st.sidebar.selectbox("Language / اللغة", langs)
    apply_rtl_if_arabic(lang)

    # Section selector
    section = st.sidebar.selectbox("Section / القسم", ["Quantitative", "Verbal"])

    # Dynamic title
    if section == "Verbal":
        title = "🎯 مولّد واختبار القدرات اللفظي" if lang.lower() == "arabic" else "🎯 GAT Verbal Question Generator & Quiz"
    else:
        title = "🎯 مولّد واختبار القدرات الكمي" if lang.lower() == "arabic" else "🎯 GAT Quantitative Question Generator & Quiz"
    st.title(title)

    df_lang = df_all[df_all["Language"] == lang]

    def _clean_dropdown(series: pd.Series) -> List[str]:
        """Return sorted unique non-empty, non-nan values from a string series."""
        return sorted([
            v for v in series.dropna().astype(str).str.strip().unique()
            if v and v.lower() not in ("nan", "none", "")
        ])

    if section == "Quantitative":
        df_section = df_lang[df_lang["Section"].astype(str).str.strip().str.lower() == "quantitative"]
        if df_section.empty:
            st.sidebar.error("No quantitative data for this language.")
            st.stop()

        cats = _clean_dropdown(df_section["Category"])
        category = st.sidebar.selectbox("Category / الفئة", ["ANY"] + cats)

        subcat = None
        if category != "ANY":
            subs = _clean_dropdown(
                df_section[df_section["Category"] == category]["Sub-Category"]
            )
            subcat = st.sidebar.selectbox("Sub-Category / الفئة الفرعية", ["ANY"] + subs)
        else:
            st.sidebar.write("Sub-Category: ANY (mixed)")

    else:  # Verbal
        df_section = df_lang[df_lang["Section"].astype(str).str.strip().str.lower() == "verbal"]
        df_section = df_section[df_section["Category"].isin(ALLOWED_VERBAL_CATS)]
        if df_section.empty:
            st.sidebar.error("No verbal data for this language.")
            st.stop()

        cats = _clean_dropdown(df_section["Category"])
        category = st.sidebar.selectbox("Category / الفئة", ["ANY"] + cats)

        subcat = None
        if category != "ANY":
            subs = _clean_dropdown(
                df_section[df_section["Category"] == category]["Sub-Category"]
            )
            subcat = st.sidebar.selectbox("Sub-Category / الفئة الفرعية", ["ANY"] + subs)
        else:
            st.sidebar.write("Sub-Category: ANY (mixed)")

    num_q = st.sidebar.number_input(
        "Number of questions / عدد الأسئلة",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    generate_button = st.sidebar.button("🚀 Generate & Start Quiz / ابدأ الاختبار")

    if generate_button:
        with st.spinner("جارٍ توليد الأسئلة..." if lang.lower() == "arabic" else "Generating questions..."):
            try:
                if section == "Quantitative":
                    questions, export_rows = generate_valid_quant_mcq_questions(
                        df_all,
                        target_language=lang,
                        category=category,
                        subcat=subcat if category != "ANY" else None,
                        num_required=num_q,
                        seen_questions=st.session_state.seen_questions,
                    )
                else:
                    questions, export_rows = generate_valid_verbal_mcq_questions(
                        df_all,
                        target_language=lang,
                        category=category,
                        subcat=subcat if category != "ANY" else None,
                        num_required=num_q,
                        prompts_dict=category_prompts,
                        seen_questions=st.session_state.seen_questions,
                    )
            except Exception as e:
                st.error(f"Error during generation: {e}")
                return

            if not questions:
                msg = (
                    "تعذّر توليد عدد كافٍ من الأسئلة. جرّب عددًا أقل أو غيّر الإعدادات."
                    if lang.lower() == "arabic"
                    else "Could not generate enough valid questions. Try fewer or change filters."
                )
                st.error(msg)
            else:
                st.session_state.generated_df = pd.DataFrame(export_rows)
                start_quiz(questions)

    # Excel export
    if not st.session_state.generated_df.empty and st.session_state.quiz_started:
        buffer = BytesIO()
        st.session_state.generated_df.to_excel(buffer, index=False)
        buffer.seek(0)

        dl_label = (
            "⬇️ تنزيل الأسئلة المولدة (Excel)"
            if lang.lower() == "arabic"
            else "⬇️ Download generated questions (Excel)"
        )
        st.download_button(
            dl_label,
            data=buffer,
            file_name="generated_gat_questions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not st.session_state.quiz_started:
        info_msg = (
            "اختر الإعدادات من القائمة الجانبية ثم اضغط (ابدأ الاختبار)."
            if lang.lower() == "arabic"
            else "Choose your settings in the sidebar and click 'Generate & Start Quiz'."
        )
        st.info(info_msg)
        return

    quiz_questions = st.session_state.quiz_questions
    st.subheader("📝 أجب عن الأسئلة التالية:" if lang.lower() == "arabic" else "📝 Answer the questions:")

    for idx, q in enumerate(quiz_questions):
        st.markdown(f"### سؤال {idx+1}" if lang.lower() == "arabic" else f"### Question {idx+1}")

        # Show passage for Reading Comprehension questions
        context = clean_text(q.get("Context", ""))
        if context:
            passage_label = "النص" if lang.lower() == "arabic" else "Passage"
            with st.expander(passage_label, expanded=True):
                st.write(context)

        st.write(q.get("Question", ""))

        options_labels = []
        mapping = {}

        for opt_label in ["A", "B", "C", "D"]:
            text = q.get(f"Option{opt_label}", "")
            full_label = f"{opt_label}) {text}"
            options_labels.append(full_label)
            mapping[full_label] = opt_label

        prev_answer = st.session_state.user_answers.get(idx)
        default_idx = 0
        if prev_answer:
            for j, lbl in enumerate(options_labels):
                if mapping[lbl] == prev_answer:
                    default_idx = j
                    break

        choice_label = "اختر الإجابة:" if lang.lower() == "arabic" else "Choose your answer:"
        choice = st.radio(choice_label, options_labels, index=default_idx, key=f"q_{idx}")
        st.session_state.user_answers[idx] = mapping[choice]

        st.markdown("---")

    submit_label = "✅ إرسال وعرض النتيجة" if lang.lower() == "arabic" else "✅ Submit & Show Result"
    if not st.session_state.quiz_finished:
        if st.button(submit_label):
            finish_quiz()

    if st.session_state.quiz_finished:
        answers = st.session_state.user_answers
        total = len(quiz_questions)
        correct_count = 0

        results = []
        for idx, q in enumerate(quiz_questions):
            user_opt = answers.get(idx)
            correct_opt = q.get("CorrectOption")
            is_correct = (user_opt == correct_opt)
            if is_correct:
                correct_count += 1

            results.append({
                "index": idx + 1,
                "question": q.get("Question", ""),
                "context": clean_text(q.get("Context", "")),
                "user_opt": user_opt,
                "correct_opt": correct_opt,
                "OptionA": q.get("OptionA", ""),
                "OptionB": q.get("OptionB", ""),
                "OptionC": q.get("OptionC", ""),
                "OptionD": q.get("OptionD", ""),
                "GenRationale_A": q.get("GenRationale_A", ""),
                "GenRationale_B": q.get("GenRationale_B", ""),
                "GenRationale_C": q.get("GenRationale_C", ""),
                "GenRationale_D": q.get("GenRationale_D", ""),
                "GenMisTag_A": q.get("GenMisTag_A", ""),
                "GenMisTag_B": q.get("GenMisTag_B", ""),
                "GenMisTag_C": q.get("GenMisTag_C", ""),
                "GenMisTag_D": q.get("GenMisTag_D", ""),
                "is_correct": is_correct,
            })

        score_pct = (correct_count / max(1, total)) * 100
        if lang.lower() == "arabic":
            st.success(f"نتيجتك: {correct_count} من {total} ({score_pct:.1f}٪)")
        else:
            st.success(f"Your score: {correct_count} / {total} ({score_pct:.1f}%)")

        exp_label = "📊 مراجعة مفصلة للأسئلة" if lang.lower() == "arabic" else "📊 Detailed Review"
        with st.expander(exp_label):
            for r in results:
                st.markdown(f"#### سؤال {r['index']}" if lang.lower() == "arabic" else f"#### Question {r['index']}")

                # Show passage for RC questions in review (no nested expander)
                if r.get("context"):
                    passage_label = "**النص:**" if lang.lower() == "arabic" else "**Passage:**"
                    st.markdown(passage_label)
                    st.info(r["context"])

                st.write(r["question"])

                if lang.lower() == "arabic":
                    st.write(f"- إجابتك: **{r['user_opt']}**")
                    st.write(f"- الإجابة الصحيحة: **{r['correct_opt']}**")
                else:
                    st.write(f"- Your answer: **{r['user_opt']}**")
                    st.write(f"- Correct answer: **{r['correct_opt']}**")

                tab1, tab2 = st.tabs(["Options / الخيارات", "Rationales / المبررات"])

                with tab1:
                    for label, text in {
                        "A": r["OptionA"], "B": r["OptionB"],
                        "C": r["OptionC"], "D": r["OptionD"]
                    }.items():
                        prefix = (
                            "✅" if label == r["correct_opt"]
                            else ("❌" if label == r["user_opt"] and not r["is_correct"] else "")
                        )
                        st.write(f"{prefix} {label}) {text}")

                with tab2:
                    any_rats = False
                    for label in ["A", "B", "C", "D"]:
                        rat = r.get(f"GenRationale_{label}", "")
                        tag = r.get(f"GenMisTag_{label}", "")
                        if rat or tag:
                            any_rats = True
                            if tag:
                                st.write(f"- {label}: **{tag}** — {rat}")
                            else:
                                st.write(f"- {label}: {rat}")
                    if not any_rats:
                        st.write("—")

                st.markdown("---")


if __name__ == "__main__":
    main()
