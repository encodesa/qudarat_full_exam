# Qudrat Full Exam Generator

Generates a full sectioned Qudrat (GAT) practice exam — Arabic — with kernel-verified
figures, unit-homogeneous distractors, and per-question explanations, then serves it
in a web exam UI.

## Pipeline

```
exam_blueprint.py        Arabic→English taxonomy; reads Full_Exam/Qudrat_Arabic.xlsx
generate_full_exam.py    orchestrator: deals questions into sections, adds explanations,
                         writes full_exam.json + full_exam.xlsx
generate_visual.py       per-category generation + figure stage; routes Geometry→geo kernel,
                         Data Interpretation→di kernel
app6_final.py            core MCQ engine (generation, solvers, distractors, unit/notation
                         homogeneity, critic)
geo_kernel/measure/render/scenarios + geo_style   deterministic geometry (24 scenarios,
                         answer computed from coordinates)
di_scenarios.py          data-first Data-Interpretation (charts; answer computed from data)
exam_app.py + templates/exam.html   Flask exam UI with end-of-exam review + explanations
```

## Run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

# Generate a 4-section, 96-question exam (best RC quality with 2.5-pro generator):
GEN_MODEL=gemini-2.5-pro GEN_CONCURRENCY=60 python generate_full_exam.py --sections 4 --per-section 24

# Serve it:
python exam_app.py               # http://localhost:8506
```

### Useful env vars
- `GEN_MODEL` — generator (default `gemini-flash-latest`; use `gemini-2.5-pro` for best RC)
- `REVIEW_MODEL` — critic + vision model (default `gemini-2.5-pro`)
- `EXPLAIN_MODEL` — explanation model (default `gemini-2.5-flash`)
- `GEN_CONCURRENCY` — parallel API calls (default 40; raise on a high-rate-limit tier)
- `GEMINI_CALL_TIMEOUT_MS` — per-call timeout (default 150000)
- `GEO_KERNEL_ENABLED`, `DI_KERNEL_ENABLED` — toggle the figure-first kernels (default on)

## Tests

```bash
python test_units.py        # unit/notation homogeneity (no API)
python test_geo_kernel.py
python test_geo_measure.py
```

## Inputs (required at runtime)
- `Qudrat Sample Sheet _ v2 (1).xlsx` — example-question corpus
- `category_prompts_v2.json` — per-category verbal prompts
- `Full_Exam/Qudrat_Arabic.xlsx` — authored blueprint (Arabic taxonomy)
