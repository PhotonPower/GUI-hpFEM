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


def make_job(model, solver, maps, pscan=None, task="scattering", resonance=None):
    """The job file of the worker. task: "scattering" or "resonances" (with resonance = {num_modes, krylov_dimension})."""
    sw = dict(model["sweep"])
    sw["values"] = sweep_values(sw)
    job = dict(model=model, layout=fg.layout(model), incidence=model["incidence"], sweep=sw, solver=solver, maps=maps, pscan=pscan, task=task,
               resonance=resonance or {})
    return job


def write_job(folder, job):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")


def source_has_module(repo):
    """True if <repo>/python/hpfem contains a built extension (_hpfem*.pyd / .so): only then may PYTHONPATH point at the source tree. Without
    it the source package would hide an hpfem installed as a wheel in the chosen Python."""
    d = Path(repo) / "python" / "hpfem" if repo else None
    return bool(d and d.is_dir() and (list(d.glob("_hpfem*.pyd")) or list(d.glob("_hpfem*.so"))))


def worker_env(repo, threads=0):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if source_has_module(repo):
        env["PYTHONPATH"] = str(Path(repo) / "python")
    else:
        env.pop("PYTHONPATH", None)
    if threads and threads > 0:
        env["OMP_NUM_THREADS"] = str(int(threads))
    return env


def _cwd(repo, folder):
    return str(repo) if repo and Path(repo).is_dir() else str(folder)


def start_worker(py, repo, folder, threads=0, script="fem_worker.py"):
    """Starts the worker script (fem_worker.py, or fem_axi_worker.py for bodies of revolution) in the background; returns the process. Output
    goes to <folder>/log.txt."""
    folder = Path(folder)
    for pattern in ("results.json", "maps_*.npz", "tri_*.npz", "hpmesh_*.npz", "mode_*.npz", "field_*.npz", "cancel"):
        for f in folder.glob(pattern):
            f.unlink()
    log = open(folder / "log.txt", "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen([str(py), str(HERE / script), str(folder)], cwd=_cwd(repo, folder), env=worker_env(repo, threads), stdout=log,
                            stderr=subprocess.STDOUT, creationflags=flags)


def start_study(py, repo, folder, threads=0):
    """Starts the job runner of hpfem (python -m hpfem.run, job schema 2) on <folder>/job.json in the background: results.json and the study
    store <name>.study.jsonl go to <folder>, the JSON-lines events to <folder>/log.txt; <folder>/cancel stops it after the current
    evaluation. The study store stays, so a restart with the same job replays the evaluations already done and continues."""
    folder = Path(folder)
    for name in ("results.json", "cancel"):
        if (folder / name).exists():
            (folder / name).unlink()
    log = open(folder / "log.txt", "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen([str(py), "-m", "hpfem.run", str(folder / "job.json"), "--out", str(folder), "--cancel-file", str(folder / "cancel"),
                             "--quiet"], cwd=_cwd(repo, folder), env=worker_env(repo, threads), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags)


def request_cancel(folder):
    """Asks the worker to stop: it polls <folder>/cancel between the phases of a solve (hpfem >= 0.4) and between the sweep points."""
    (Path(folder) / "cancel").write_text("cancel", encoding="utf-8")


def run_check(py, repo, folder, threads=0, timeout=600):
    """Runs the worker with --check (diagnostics of the library, memory estimate; no solve) and returns the dict of check.json, or a dict with
    "error" (and the output of the worker)."""
    folder = Path(folder)
    (folder / "check.json").unlink(missing_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run([str(py), str(HERE / "fem_worker.py"), str(folder), "--check"], cwd=_cwd(repo, folder), env=worker_env(repo, threads),
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
    except Exception as exc:
        return dict(error=str(exc), output="")
    f = folder / "check.json"
    if p.returncode == 0 and f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    return dict(error=f"Prüfung fehlgeschlagen (Code {p.returncode})", output=(p.stdout or "") + (p.stderr or ""))


PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import hpfem
    out["hpfem"] = str(getattr(hpfem, "__version__", "?"))
    out["file"] = str(getattr(hpfem, "__file__", ""))
    try:
        out["version_info"] = hpfem.version_info()
    except Exception:
        pass
    feats = {}
    try:
        import hpfem.grating as g
        feats["grating"] = hasattr(g, "solve")
        feats["jacobian"] = hasattr(g, "jacobian")
        feats["resonances"] = hasattr(g, "resonances")
        feats["jones"] = hasattr(g, "jones")
        feats["emitter"] = hasattr(g, "dipole_emission") and hasattr(g, "emission_cost")
    except Exception:
        feats["grating"] = feats["jacobian"] = feats["resonances"] = feats["jones"] = feats["emitter"] = False
    try:
        import inspect
        import hpfem.dataports as dpt
        feats["dataports"] = hasattr(dpt, "planar_emitter_response")
        feats["offaxis"] = "radii" in inspect.signature(dpt.axisymmetric_emitter_response).parameters
    except Exception:
        feats["dataports"] = feats["offaxis"] = False
    feats["bands"] = all(hasattr(hpfem, n) for n in ("BandStructure2D", "BandStructureSetup2D", "PeriodicPair2D"))
    feats["ports"] = all(hasattr(hpfem, n) for n in ("WaveguidePort", "s_parameters", "PropagatingMode", "subdivide"))
    feats["crystals"] = hasattr(hpfem, "AnisotropicStack") and hasattr(getattr(hpfem, "Material", None), "anisotropic")
    feats["dwr"] = hasattr(hpfem, "conical_dwr_estimate")
    feats["sample"] = hasattr(getattr(hpfem, "ConicalScattering", None), "sample")
    feats["h_field"] = hasattr(getattr(hpfem, "ConicalScattering", None), "h_field")
    feats["absorbed_power"] = hasattr(hpfem, "absorbed_power_by_tag")
    try:
        import hpfem.pv as _pv
        feats["pv"] = hasattr(_pv, "depth_profile")
    except Exception:
        feats["pv"] = False
    feats["progress"] = hasattr(hpfem, "ProgressEvent")
    feats["memory"] = hasattr(hpfem, "estimate_memory")
    try:
        import hpfem.opt  # noqa: F401
        feats["opt"] = True
    except Exception:
        feats["opt"] = False
    try:
        import hpfem.run as hr
        feats["study"] = "optimize" in getattr(hr, "STUDY_TASKS", ())
        feats["opt_mcmc"] = __import__("importlib").util.find_spec("emcee") is not None
    except Exception:
        feats["study"] = feats["opt_mcmc"] = False
    out["features"] = feats
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("PROBE" + json.dumps(out))
"""


def probe_library(py, repo, timeout=120):
    """Version and features of the hpfem that the worker will use (import in the chosen Python with the worker's environment)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run([str(py), "-c", PROBE], cwd=_cwd(repo, HERE), env=worker_env(repo), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, creationflags=flags)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")
    for line in (p.stdout or "").splitlines():
        if line.startswith("PROBE"):
            out = json.loads(line[5:])
            if source_has_module(repo) and not out.get("error"):
                out["older_source"] = older_source_build(py, out.get("hpfem"), flags, timeout)
            return out
    return dict(error=((p.stderr or "") + (p.stdout or "")).strip()[-800:] or f"Code {p.returncode}")


def version_tuple(v):
    """'0.5.0' -> (0, 5, 0); non-numeric parts are ignored."""
    return tuple(int(x) for x in re.findall(r"\d+", str(v or ""))[:3])


def older_source_build(py, source_version, flags=0, timeout=120):
    """If the source build (PYTHONPATH) hides a newer hpfem installed in the chosen Python: {source, installed}, else None."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("PYTHONPATH", None)
    try:
        p = subprocess.run([str(py), "-c", "import hpfem; print('V' + hpfem.__version__)"], cwd=str(HERE), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
    except Exception:
        return None
    for line in (p.stdout or "").splitlines():
        if line.startswith("V"):
            inst = line[1:].strip()
            if version_tuple(inst) > version_tuple(source_version):
                return dict(source=str(source_version), installed=inst)
    return None


def fit_material(py, repo, spec, lam_lo_nm, lam_hi_nm, oscillators=1, count=60, timeout=180):
    """Drude-Lorentz-Fit (fem_disp.fit im Python mit hpfem) an ein Bibliotheks-, Tabellen- oder Drude-Material. Gibt das Ergebnis-Dict
    (spec, max_rel_error, Kurven) zurück; ``ValueError`` mit der Meldung des Fits bei einem Fehler."""
    import tempfile

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "fit.json"
        job.write_text(json.dumps(dict(spec=spec, lam_lo_nm=float(lam_lo_nm), lam_hi_nm=float(lam_hi_nm), oscillators=int(oscillators), count=int(count))),
                       encoding="utf-8")
        try:
            p = subprocess.run([str(py), str(HERE / "fem_disp.py"), "fit", str(job)], cwd=_cwd(repo, HERE), env=worker_env(repo), capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
        except Exception as exc:
            raise ValueError(f"{type(exc).__name__}: {exc}") from exc
    for line in (p.stdout or "").splitlines():
        if line.startswith("FIT"):
            out = json.loads(line[3:])
            if out.get("error"):
                raise ValueError(out["error"])
            return out
    raise ValueError(((p.stderr or "") + (p.stdout or "")).strip()[-600:] or f"Code {p.returncode}")


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
    frac, label = max(done - 1, 0) / total, (f"Punkt {done} von {total}" if done else "Vorbereitung (Resonanzsuche)")
    steps = list(re.finditer(r"STEP (\d+)/(\d+)", text))
    last_prog = [m.start() for m in re.finditer(r"PROGRESS \d+/\d+", text)][-1]
    if steps and steps[-1].start() > last_prog:
        s, S_ = int(steps[-1].group(1)), int(steps[-1].group(2))
        frac += min(s / S_, 1.0) / total
        label += f", hp-Schritt {s} von höchstens {S_}"
    phases = list(re.finditer(r"PHASE (\w+)", text))
    counted = list(re.finditer(r"^PHASE (\w+)(?: [\w.]+)? (\d+)/(\d+)[ \t]*$", text, flags=re.M))          # emitter tasks: "PHASE scan 12/224", "PHASE aperture 0.5 3/32"
    if counted and counted[-1].start() > last_prog and counted[-1].start() >= phases[-1].start():
        name, s, S_ = counted[-1].group(1), int(counted[-1].group(2)), int(counted[-1].group(3))
        frac += min(s / max(S_, 1), 1.0) / total
        label += f" · {PHASE_LABELS.get(name, name)} {s} von {S_}"
    elif phases and phases[-1].start() > last_prog:
        name = phases[-1].group(1)
        label += f" · {PHASE_LABELS.get(name, name)}"
    return min(frac, 0.999), label


PHASE_LABELS = {"assembly": "Assemblierung", "constraints": "Randbedingungen", "factorisation": "Faktorisierung", "factorization": "Faktorisierung",
                "solve": "Lösen", "post": "Auswertung", "eigensolve": "Eigenwertlöser", "done": "Lösung fertig",
                "scan": "Array-Scan (Zellenprobleme)", "emit": "Zellenproblem", "pattern": "Abstrahlmuster (ebene Wellen)", "aperture": "Apertur (ebene Wellen)"}


def estimate_line(text):
    """The memory estimate the worker printed ('ESTIMATE ...'), or ''."""
    m = re.findall(r"^ESTIMATE (.*)$", text, flags=re.M)
    return m[-1] if m else ""


def diagnostics_lines(text):
    """[(severity, code, text)] of the DIAG lines of the log."""
    return [(m.group(1), m.group(2), m.group(3)) for m in re.finditer(r"^DIAG (\w+) (\w+): (.*)$", text, flags=re.M)]


def log_flags(text):
    return dict(done="DONE" in text.splitlines()[-1:] or "\nDONE" in text or text.strip().endswith("DONE"), cancelled="\nCANCELLED" in text,
                errors=[l for l in text.splitlines() if l.startswith("ERROR")], warnings=[l for l in text.splitlines() if l.startswith("WARN")])


def default_python():
    """The Python of the app if it has hpfem installed (wheel, hp-FEM >= 0.4), else MSYS2 (source build), else the Python of the app."""
    import importlib.util
    try:
        if importlib.util.find_spec("hpfem") is not None:
            return sys.executable
    except Exception:
        pass
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
