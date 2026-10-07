"""Jobs of the FEM model builder: sweep values, job file, worker process in the background with a log file.

The worker (fem_worker.py) runs in the Python that has hpfem; it is started as its own process that writes its output into <job>/log.txt, so a
rerun of the Streamlit page (any click) neither stops nor loses it: the app finds the process again by its process object / log file.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import fem_geometry as fg

HERE = Path(__file__).resolve().parent


def sweep_values(sweep):
    if sweep["mode"] == "none":
        return []
    n = max(int(sweep["n"]), 1)
    return [float(round(v, 6)) for v in np.linspace(sweep["start"], sweep["stop"], n)]


def wavelength_range(model):
    """(lam_min, lam_max) in nm over which the model is meshed and checked."""
    sw, inc = model["sweep"], model["incidence"]
    if sw["mode"] == "wavelength":
        return float(min(sw["start"], sw["stop"])), float(max(sw["start"], sw["stop"]))
    return float(inc["wavelength_nm"]), float(inc["wavelength_nm"])


def theta_max(model):
    sw, inc = model["sweep"], model["incidence"]
    return float(max(abs(sw["start"]), abs(sw["stop"]))) if sw["mode"] == "theta" else float(abs(inc["theta"]))


def make_job(model, solver, maps, pscan=None):
    sw = dict(model["sweep"])
    sw["values"] = sweep_values(sw)
    job = dict(model=model, layout=fg.layout(model), incidence=model["incidence"], sweep=sw, solver=solver, maps=maps, pscan=pscan)
    return job


def write_job(folder, job):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")


def start_worker(py, repo, folder, threads=0):
    """Starts fem_worker.py in the background; returns the process. Output goes to <folder>/log.txt."""
    folder = Path(folder)
    for f in list(folder.glob("results.json")) + list(folder.glob("maps_*.npz")):
        f.unlink()
    env = {**os.environ, "PYTHONPATH": str(Path(repo) / "python"), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if threads and threads > 0:
        env["OMP_NUM_THREADS"] = str(int(threads))
    log = open(folder / "log.txt", "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen([str(py), str(HERE / "fem_worker.py"), str(folder)], cwd=str(repo), env=env, stdout=log,
                            stderr=subprocess.STDOUT, creationflags=flags)


def read_log(folder, tail=None):
    p = Path(folder) / "log.txt"
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if tail:
        text = "\n".join(text.splitlines()[-tail:])
    return text


def progress(text):
    """(done, total) from the last 'PROGRESS i/n' line, or (0, 0)."""
    m = re.findall(r"PROGRESS (\d+)/(\d+)", text)
    return (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)


def progress_info(text):
    """(fraction 0..1, label) from the PROGRESS lines (sweep points) and the STEP lines (hp-adaptive steps inside a point)."""
    done, total = progress(text)
    if total == 0:
        return 0.0, "Netz wird gelesen, Solver startet …"
    if "\nDONE" in text or text.strip().endswith("DONE"):
        return 1.0, "fertig"
    frac, label = (done - 1) / total, f"Punkt {done} von {total}"
    steps = list(re.finditer(r"STEP (\d+)/(\d+)", text))
    last_prog = [m.start() for m in re.finditer(r"PROGRESS \d+/\d+", text)][-1]
    if steps and steps[-1].start() > last_prog:
        s, S_ = int(steps[-1].group(1)), int(steps[-1].group(2))
        frac += min(s / S_, 1.0) / total
        label += f", hp-Schritt {s} von höchstens {S_}"
    return min(frac, 0.999), label


def log_flags(text):
    return dict(done="DONE" in text.splitlines()[-1:] or "\nDONE" in text or text.strip().endswith("DONE"),
                errors=[l for l in text.splitlines() if l.startswith("ERROR")], warnings=[l for l in text.splitlines() if l.startswith("WARN")])


def default_python():
    for c in ("C:/msys64/ucrt64/bin/python3.exe", "C:/msys64/ucrt64/bin/python.exe", "/ucrt64/bin/python3"):
        if Path(c).exists():
            return c
    return sys.executable


def library_warnings(text):
    """Distinct warnings the hpfem library printed ('[date] [warning] message'), with their counts, for the app to show."""
    counts, order = {}, []
    for line in text.splitlines():
        m = re.match(r"\s*\[[^\]]*\]\s*\[warning\]\s*(.*)", line)
        if m:
            msg = re.sub(r"\b\d+(\.\d+)?\b", "#", m.group(1))        # numbers differ from cell to cell: one entry per kind of warning
            if msg not in counts:
                counts[msg] = [m.group(1), 0]
                order.append(msg)
            counts[msg][1] += 1
    return [(counts[k][0], counts[k][1]) for k in order]
