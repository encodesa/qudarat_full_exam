"""
Qudrat VISUAL — unified generate + view app (single code).

One Flask app that BOTH generates image-capable questions on demand and
displays them (with their figures) in the browser. Charts/tables are rendered
with matplotlib; geometry/science figures with a Gemini image model.

Run:
  python app_visual.py        # http://127.0.0.1:8505
"""
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

import generate_visual as gv

app = Flask(__name__)
OUT_DIR = gv.OUT_DIR
IMAGES_FOLDER = gv.IMG_DIR


def _clean_row(row: dict) -> dict:
    return {
        k: ("" if str(v).strip().lower() in ("nan", "none", "") else str(v))
        for k, v in row.items()
    }


@app.route("/")
def index():
    return render_template("app_visual.html")


@app.route("/api/options")
def api_options():
    section = request.args.get("section", "Quantitative")
    lang = request.args.get("lang") or None
    try:
        return jsonify({"options": gv.options_for(section, lang)})
    except Exception as e:
        return jsonify({"options": {}, "error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    section = data.get("section", "Quantitative")
    category = data.get("category", "ANY") or "ANY"
    subcat = data.get("subcat", "ANY") or "ANY"
    lang = data.get("lang", "Arabic")
    try:
        num = max(1, min(20, int(data.get("num", 5))))
    except Exception:
        num = 5
    try:
        rows = gv.generate_questions(section, category, subcat, lang, num, save=True)
    except Exception as e:
        gv.gen.log(f"[generate] {e}", level="error")
        return jsonify({"questions": [], "count": 0, "error": str(e)}), 500
    questions = [_clean_row(r) for r in rows]
    return jsonify({"questions": questions, "count": len(questions)})


@app.route("/images/<path:relpath>")
def serve_image(relpath):
    rel = relpath[len("images/"):] if relpath.startswith("images/") else relpath
    target = (IMAGES_FOLDER / rel).resolve()
    if not str(target).startswith(str(IMAGES_FOLDER.resolve())) or not target.exists():
        abort(404)
    return send_from_directory(IMAGES_FOLDER, rel)


if __name__ == "__main__":
    # threaded so the slow generate call doesn't block image requests
    app.run(debug=True, port=8505, host="0.0.0.0", threaded=True)
