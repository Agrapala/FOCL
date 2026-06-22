"""
FLoBC Web Dashboard
====================
Run:  ./venv/bin/python app.py
Open: http://localhost:5000
"""

import os, sys, json, glob, time
import subprocess
from flask import Flask, render_template, Response, jsonify, send_file, abort
from flask_cors import CORS

ROOT = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

app = Flask(__name__)
CORS(app)

# ── Whitelisted scripts ───────────────────────────────────────────────────────

SCRIPTS = {
    "demo": {
        "name":  "Quick Demo",
        "file":  "demo.py",
        "desc":  "4 demos: basic blockchain FL, Byzantine tolerance, sync schemes, tamper detection",
        "time":  "~30 sec",
        "stage": "demo",
    },
    "run_all": {
        "name":  "Run All",
        "file":  "run_all.py",
        "desc":  "Full pipeline: paper experiments + objective proofs + Word document",
        "time":  "~10 min",
        "stage": "full",
    },
    "run_paper_experiments": {
        "name":  "Paper Experiments",
        "file":  "run_paper_experiments.py",
        "desc":  "4 experiments: centralized vs decentralised, T/V ratio, reward-penalty, sync schemes",
        "time":  "~5 min",
        "stage": "experiment",
    },
    "prove_objectives": {
        "name":  "Prove Objectives",
        "file":  "prove_objectives.py",
        "desc":  "Prove all 4 research objectives with metrics, blockchain proof, and charts",
        "time":  "~5 min",
        "stage": "experiment",
    },
    "train_local": {
        "name":  "Train Local Nodes",
        "file":  "train_local_nodes.py",
        "desc":  "Stage 1: train each hospital independently before federation begins",
        "time":  "~3 min",
        "stage": "stage",
    },
    "run_pneumonia": {
        "name":  "Run Pneumonia FL",
        "file":  "run_pneumonia.py",
        "desc":  "Stage 2: all FL experiments on chest X-ray pneumonia dataset",
        "time":  "~5 min",
        "stage": "stage",
    },
    "evaluate_objectives": {
        "name":  "Evaluate Objectives",
        "file":  "evaluate_objectives.py",
        "desc":  "Stage 3: verify research objectives with per-hospital accuracy metrics",
        "time":  "~3 min",
        "stage": "stage",
    },
    "generate_doc": {
        "name":  "Generate Word Doc",
        "file":  "generate_mechanism_doc.py",
        "desc":  "Generate the FLoBC PoCL mechanism Word document",
        "time":  "~10 sec",
        "stage": "demo",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json_path(base, filename):
    path = os.path.realpath(os.path.join(base, filename))
    if not path.startswith(os.path.realpath(base)):
        return None
    if not path.endswith(".json"):
        return None
    return path


def _load_stats():
    stats = {
        "chain_length": None, "chain_valid": None,
        "objectives_met": None, "fl_final_acc": None,
        "charts_count": 0, "results_count": 0,
    }
    proof = os.path.join(ROOT, "results", "blockchain_proof.json")
    if os.path.exists(proof):
        with open(proof) as f:
            d = json.load(f)
        stats["chain_length"] = d.get("chain_length")
        stats["chain_valid"]  = d.get("is_valid")

    obj = os.path.join(ROOT, "results", "objectives_proof.json")
    if os.path.exists(obj):
        with open(obj) as f:
            d = json.load(f)
        keys = ["objective_1", "objective_2", "objective_3", "objective_4"]
        stats["objectives_met"] = sum(1 for k in keys if d.get(k, {}).get("met"))
        log = d.get("fl_accuracy_log", [])
        stats["fl_final_acc"] = round(log[-1], 4) if log else None

    stats["charts_count"]  = len(glob.glob(os.path.join(ROOT, "dashboard", "*.png")))
    stats["results_count"] = len(glob.glob(os.path.join(ROOT, "results", "*.json")))
    return stats


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", scripts=SCRIPTS)


@app.route("/api/stats")
def api_stats():
    return jsonify(_load_stats())


@app.route("/api/run/<script_id>")
def api_run(script_id):
    if script_id not in SCRIPTS:
        abort(404)
    script_file = os.path.join(ROOT, SCRIPTS[script_id]["file"])
    if not os.path.exists(script_file):
        abort(404, description=f"{SCRIPTS[script_id]['file']} not found")

    def generate():
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [PY, script_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
            bufsize=1,
            env=env,
        )
        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                safe = line.replace("\n", " ")
                yield f"data: {safe}\n\n"
            proc.wait()
            yield f"data: __EXIT__:{proc.returncode}\n\n"
        except GeneratorExit:
            proc.terminate()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/blockchain/list")
def api_blockchain_list():
    files = []
    patterns = [
        os.path.join(ROOT, "results", "*.json"),
        os.path.join(ROOT, "results", "New folder", "*.json"),
    ]
    for pattern in patterns:
        for f in sorted(glob.glob(pattern)):
            name = os.path.basename(f)
            rel  = os.path.relpath(f, ROOT)
            try:
                with open(f) as fh:
                    d = json.load(fh)
                if "blocks" in d or "chain_length" in d:
                    files.append({
                        "name":         name,
                        "path":         rel,
                        "chain_length": d.get("chain_length", "?"),
                        "is_valid":     d.get("is_valid"),
                        "size_kb":      round(os.path.getsize(f) / 1024, 1),
                    })
            except Exception:
                pass
    return jsonify(files)


@app.route("/api/blockchain/data")
def api_blockchain_data():
    from flask import request
    rel = request.args.get("file", "")
    path = _safe_json_path(ROOT, rel)
    if not path or not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/results/list")
def api_results_list():
    files = []
    for f in glob.glob(os.path.join(ROOT, "results", "*.json")):
        files.append({
            "name":    os.path.basename(f),
            "size_kb": round(os.path.getsize(f) / 1024, 1),
            "mtime":   os.path.getmtime(f),
        })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(files)


@app.route("/api/results/data")
def api_results_data():
    from flask import request
    filename = request.args.get("file", "")
    path = _safe_json_path(os.path.join(ROOT, "results"), filename)
    if not path or not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/charts/list")
def api_charts_list():
    charts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "dashboard", "*.png"))):
        charts.append(os.path.basename(f))
    return jsonify(charts)


@app.route("/api/charts/<filename>")
def api_chart_image(filename):
    if "/" in filename or "\\" in filename:
        abort(400)
    path = os.path.join(ROOT, "dashboard", filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/api/download/results/<filename>")
def api_download_result(filename):
    path = _safe_json_path(os.path.join(ROOT, "results"), filename)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  FLoBC Web Dashboard")
    print("  Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
