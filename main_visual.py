"""
GAT Question Review App — VISUAL version — منصة القدرات
Serves image-capable questions from questins_visual/ plus their PNG figures.
Separate from main.py / questins/ (unchanged).
"""
import json, random
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory, abort
import pandas as pd

app = Flask(__name__)
QUESTIONS_FOLDER = Path(__file__).parent / "questins_visual"
IMAGES_FOLDER = QUESTIONS_FOLDER / "images"

_INDEX: dict = {}
_TREE:  dict = {}


def _load():
    global _INDEX, _TREE
    if not QUESTIONS_FOLDER.exists():
        return
    for xlsx in sorted(QUESTIONS_FOLDER.glob("*.xlsx")):
        try:
            df = pd.read_excel(xlsx)
        except Exception:
            continue
        if df.empty:
            continue
        for col in ["Section", "Category", "Sub-Category"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        sec = df["Section"].iloc[0]      if "Section"      in df.columns else "Unknown"
        cat = df["Category"].iloc[0]     if "Category"     in df.columns else "Unknown"
        sub = df["Sub-Category"].iloc[0] if "Sub-Category" in df.columns else "Unknown"
        key = (sec, cat, sub)
        _INDEX[key] = pd.concat([_INDEX[key], df], ignore_index=True) if key in _INDEX else df

    for key in _INDEX:
        _INDEX[key] = (
            _INDEX[key]
            .drop_duplicates(subset=["Question"], keep="first")
            .reset_index(drop=True)
        )

    for sec, cat, sub in _INDEX:
        _TREE.setdefault(sec, {}).setdefault(cat, [])
        if sub not in _TREE[sec][cat]:
            _TREE[sec][cat].append(sub)
    for s in _TREE:
        for c in _TREE[s]:
            _TREE[s][c].sort()


_load()


def _clean(records: list) -> list:
    out = []
    for row in records:
        out.append({
            k: ("" if str(v).strip().lower() in ("nan", "none", "") else str(v))
            for k, v in row.items()
        })
    return out


@app.route("/")
def index():
    return render_template("index_visual.html", tree_json=json.dumps(_TREE))


@app.route("/images/<path:relpath>")
def serve_image(relpath):
    # relpath is "images/<cat>/<file>.png" as stored, strip leading "images/"
    rel = relpath
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    target = (IMAGES_FOLDER / rel).resolve()
    if not str(target).startswith(str(IMAGES_FOLDER.resolve())) or not target.exists():
        abort(404)
    return send_from_directory(IMAGES_FOLDER, rel)


@app.route("/api/count")
def api_count():
    key = (
        request.args.get("section", ""),
        request.args.get("category", ""),
        request.args.get("subcategory", ""),
    )
    return jsonify({"count": len(_INDEX.get(key, []))})


@app.route("/api/questions")
def api_questions():
    key = (
        request.args.get("section", ""),
        request.args.get("category", ""),
        request.args.get("subcategory", ""),
    )
    if key not in _INDEX:
        return jsonify({"questions": [], "count": 0})
    df = _INDEX[key].sample(frac=1, random_state=random.randint(0, 99999))
    questions = _clean(df.to_dict(orient="records"))
    return jsonify({"questions": questions, "count": len(questions)})


if __name__ == "__main__":
    app.run(debug=True, port=8504, host="0.0.0.0")
