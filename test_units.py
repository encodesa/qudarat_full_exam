#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic tests for the unit/notation homogeneity layer in app6_final.

These guard the headline bug: a correct answer shipping WITHOUT a unit while the
distractors carry one (Q14), units missing where the stem names them (Q67), and
mixed pi/bare-number notation (Q58). No API calls — pure post-processing logic.
"""
import app6_final as a


def _opts(d):
    return a.enforce_units_on_options(d["correct"], dict(d["options"]),
                                      d.get("question", ""))


def test_q14_correct_gets_unit_from_distractor_majority():
    # Correct answer bare, 3 distractors carry ريالات -> all must carry ريال.
    out = _opts({
        "correct": "٦٠٠",
        "options": {"A": "٥٨٦.٥ ريالات", "B": "٦٠٠",
                    "C": "٤٤٣.٥ ريالات", "D": "٤٣٣.٥ ريالات"},
    })
    assert a.options_unit_homogeneous(out)
    assert all("ريال" in v for v in out.values()), out


def test_q67_unit_seeded_from_question_stem():
    out = _opts({
        "correct": "٦٠",
        "options": {"A": "٦٠", "B": "٥٥", "C": "١٠٠", "D": "٢٠"},
        "question": "سلعة سعرها ٨٠ ريالًا، عليها خصم ٢٥٪. ما هو سعرها بعد الخصم؟",
    })
    assert a.options_unit_homogeneous(out)
    assert all("ريال" in v for v in out.values()), out


def test_q61_area_stays_bare_homogeneous():
    out = _opts({
        "correct": "٤٥",
        "options": {"A": "٥٠", "B": "٤٦", "C": "٤٥", "D": "٩٠"},
    })
    assert a.options_unit_homogeneous(out)
    assert all(not a.extract_unit_from_answer(v) for v in out.values()), out


def test_q58_pi_notation_mismatch_detected():
    out = _opts({
        "correct": "١٠٠ - ٢٥ط",
        "options": {"A": "١٠٠ - ٢٥ط", "B": "١٠٠",
                    "C": "١٠٠ - ٥٠ط", "D": "٢٥ط"},
    })
    # pi expressions must NOT get a stray unit, and the bare 100 among pi options
    # must be flagged as non-homogeneous so the gate forces a rebuild.
    assert not a.options_notation_homogeneous(out), out


def test_q10_already_consistent_unchanged():
    out = _opts({
        "correct": "١٢ سم",
        "options": {"A": "٣ سم", "B": "٢٧ سم", "C": "٦.٧٥ سم", "D": "١٢ سم"},
    })
    assert a.options_unit_homogeneous(out)
    assert all("سم" in v for v in out.values()), out


def test_western_digits_currency():
    out = _opts({
        "correct": "600",
        "options": {"A": "586.5 SR", "B": "600", "C": "443.5 SR", "D": "433.5 SR"},
    })
    assert a.options_unit_homogeneous(out)
    assert all("SR" in v for v in out.values()), out


def test_q3_bare_number_ask_gets_no_stray_percent():
    # Reviewer Q3: "if 30% of a number = 90, what is THE NUMBER?" The givens
    # mention "%", but the ask is for a plain number -> options must stay bare.
    out = _opts({
        "correct": "٣٠٠",
        "options": {"A": "٢٧٠", "B": "٢٧", "C": "٣٠", "D": "٣٠٠"},
        "question": "إذا كان ٣٠٪ من عدد يساوي ٩٠، فما هذا العدد؟",
    })
    assert a.options_unit_homogeneous(out)
    assert all("%" not in v and "٪" not in v for v in out.values()), out
    assert all(not a.extract_unit_from_answer(v) for v in out.values()), out


def test_percent_ask_gets_percent_unit():
    out = _opts({
        "correct": "٥٠",
        "options": {"A": "٥٠", "B": "١١", "C": "٢٢", "D": "١٧"},
        "question": "ما النسبة المئوية للأرز؟",
    })
    assert a.options_unit_homogeneous(out)
    assert all("%" in v for v in out.values()), out


def test_extract_unit_arabic_and_latin():
    assert a.extract_unit_from_answer("٥ سم") == "سم"
    assert a.extract_unit_from_answer("586.5 ريالات") == "ريال"
    assert a.extract_unit_from_answer("12 km") == "km"
    assert a.extract_unit_from_answer("٢٥ط") is None       # pi is notation, not a unit
    assert a.extract_unit_from_answer("٤٥") is None         # bare number


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
