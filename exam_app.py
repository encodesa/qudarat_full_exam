"""Sectioned exam interface for the generated full exam.

Serves questins_visual/full_exam.json as a real-paper-style test: one section at
a time, per-section timer, and once a section is finished it locks (no going
back). Figures are served from the existing images folder.

Run:
  python generate_full_exam.py        # build full_exam.json first
  python exam_app.py                  # http://127.0.0.1:8506
"""
import json
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, abort

import generate_visual as gv
import generate_full_exam as gfe

app = Flask(__name__)
IMAGES_FOLDER = gv.IMG_DIR
FULL_JSON = gfe.FULL_JSON

_gen_lock = threading.Lock()
_gen_state = {"running": False, "error": None}


@app.route("/")
def index():
    return render_template("exam.html")


@app.route("/api/exam")
def api_exam():
    if not FULL_JSON.exists():
        return jsonify({"sections": [], "error": "no_exam",
                        "message": "لم يتم توليد اختبار بعد. اضغط على \"توليد اختبار\"."}), 404
    try:
        data = json.loads(Path(FULL_JSON).read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"sections": [], "error": str(e)}), 500
    return jsonify(data)


def _run_generation(sheet, lang, sections, per_section, time_min):
    import sys
    argv = sys.argv
    sys.argv = ["generate_full_exam.py", "--sheet", sheet, "--lang", lang,
                "--sections", str(sections), "--per-section", str(per_section),
                "--time-min", str(time_min)]
    try:
        gfe.main()
        _gen_state["error"] = None
    except Exception as e:
        gv.gen.log(f"[exam-gen] {e}", level="error")
        _gen_state["error"] = str(e)
    finally:
        sys.argv = argv
        _gen_state["running"] = False


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Kick off a full-exam generation in a background thread."""
    if _gen_state["running"]:
        return jsonify({"status": "running"}), 202
    data = request.get_json(force=True) or {}
    sheet = data.get("sheet") or gfe.bp.DEFAULT_SHEET
    lang = data.get("lang") or "Arabic"
    sections = int(data.get("sections") or 5)
    per_section = int(data.get("per_section") or 24)
    time_min = int(data.get("time_min") or 26)
    with _gen_lock:
        if _gen_state["running"]:
            return jsonify({"status": "running"}), 202
        _gen_state["running"] = True
        _gen_state["error"] = None
    t = threading.Thread(target=_run_generation,
                         args=(sheet, lang, sections, per_section, time_min),
                         daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/generate/status")
def api_generate_status():
    return jsonify({"running": _gen_state["running"],
                    "error": _gen_state["error"],
                    "ready": FULL_JSON.exists()})


@app.route("/images/<path:relpath>")
def serve_image(relpath):
    rel = relpath[len("images/"):] if relpath.startswith("images/") else relpath
    target = (IMAGES_FOLDER / rel).resolve()
    if not str(target).startswith(str(IMAGES_FOLDER.resolve())) or not target.exists():
        abort(404)
    return send_from_directory(IMAGES_FOLDER, rel)


if __name__ == "__main__":
    app.run(debug=True, port=8506, host="0.0.0.0", threaded=True)
