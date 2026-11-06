"""Studies of hp-FEM (M16, ADR-0012): reading and plotting the study tasks "optimize", "reconstruct" and "uq" of the job runner
(python -m hpfem.run, job schema version 2). No streamlit and no hpfem here.

A study folder holds job.json (the job document), log.txt (the JSON-lines events of the runner on stdout, plus library warnings),
results.json (written at the end) and <name>.study.jsonl (the study store: header, one "evaluation" record per solve, proposals, optimiser
states, remeshes; values in SI). The history is read from the store, so a running, cancelled or resumed study shows every evaluation; the
parameters are converted to job units (lengths in units of "unit", angles in degrees, permittivities as they are) like the runner does.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("optimize", "reconstruct", "uq")
TASK_LABELS = {"optimize": "Optimierung", "reconstruct": "Rekonstruktion (Anpassung an Messwerte)", "uq": "Unsicherheiten und Sensitivität (UQ)"}
STATUS_LABELS = {"ok": "ok", "failed": "fehlgeschlagen", "cancelled": "abgebrochen", "invalid": "ungültig"}

# a cheap lamellar cell (order 2, two configurations): the cell of the runner tests of hp-FEM (python/tests/test_run_study.py); the width
# stays within 180-215 nm, beyond about 220 nm the morphed structured mesh inverts cells (MeshQualityError, the evaluation fails)
TEMPLATE = {
    "version": 2, "name": "lamellar", "problem": "grating", "unit": 1e-9,
    "model": {"period": 400, "y_bottom": -800, "y_top": 950, "background_tag": 1, "slabs": [{"tag": 2, "y0": -800, "y1": 0}],
              "shapes": [{"kind": "rectangle", "tag": 3, "params": {"x": -100, "y": 0, "width": 200, "height": 148}}]},
    "mesh": {"structured": {"nx": 16, "rows": [[-800, 0, 32], [0, 148, 6], [148, 950, 32]]}},
    "materials": {"1": "air", "2": {"n": [1.5, 0.0]}, "3": {"eps": [2.25, 0.0]}},
    "stack": {"incidence": "air", "substrate": {"n": [1.5, 0.0]}, "layers": [], "top": 0},
    "solver": {"order": 2, "pml": {"top": 419, "bottom": 419}, "orders_max": 2},
    "parameters": [{"name": "width", "shape": 0, "field": "width", "bounds": [180, 215]},
                   {"name": "eps", "material": 3, "part": "re", "bounds": [2.0, 2.6]}],
    "configurations": [{"wavelength": 405, "theta_deg": 40, "polarisation": "s", "orders": [["R", 0], ["T", 0]]},
                       {"wavelength": 500, "theta_deg": 20, "polarisation": "p", "orders": [["R", 0]]}],
}
TASK_DEFAULTS = {
    "optimize": {"objective": 0, "maximize": True, "method": "bayesian", "acquisition": "ei", "max_evaluations": 20, "seed": 0},
    "reconstruct": {"synthetic": {"params": {"width": 210, "eps": 2.3}, "noise": 0.002, "seed": 0}, "x0": {"width": 195, "eps": 2.25}},
    "uq": {"inputs": {"width": {"normal": [200, 5]}, "eps": {"uniform": [2.2, 2.4]}}, "propagation": "surrogate", "points": 12, "sobol": True,
           "samples": 10000},
}


def template(task):
    """A complete job document of the given study task on the lamellar template cell."""
    job = copy.deepcopy(TEMPLATE)
    job["task"] = task
    job["name"] = f"lamellar {task}"
    job[task] = copy.deepcopy(TASK_DEFAULTS[task])
    return job


# ----------------------------------------------------------------------------------------------------------------- reading
def _decode(obj):
    """The JSON conventions of the store: {"__complex__": [re, im]}, "inf" / "-inf", null for NaN."""
    if isinstance(obj, dict):
        if set(obj) == {"__complex__"}:
            re_, im_ = obj["__complex__"]
            return complex(re_, im_)
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if obj == "inf":
        return math.inf
    if obj == "-inf":
        return -math.inf
    return obj


def read_store(path):
    """The records of a study store, grouped: header (dict or None), evaluations, states, proposals, remeshes, notes. Lines that are not
    JSON (a truncated last line while the study writes) are skipped."""
    out = {"header": None, "evaluations": [], "states": [], "proposals": [], "remeshes": [], "notes": []}
    p = Path(path)
    if not p.is_file():
        return out
    groups = {"evaluation": "evaluations", "state": "states", "proposal": "proposals", "remesh": "remeshes", "note": "notes"}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = _decode(json.loads(line))
        except ValueError:
            continue
        kind = rec.get("type")
        if kind == "study" and out["header"] is None:
            out["header"] = rec
        elif kind in groups:
            out[groups[kind]].append(rec)
    return out


def parse_events(text):
    """The JSON events of the runner in a log (other lines, e.g. library warnings, are skipped)."""
    events = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    return events


def check_job(job):
    """Errors that keep the runner from starting the job (a list of German texts; empty when the document looks complete). The runner
    checks the rest (materials, mesh, parameter kinds) and reports it in the log."""
    if not isinstance(job, dict):
        return ["Das Dokument ist kein JSON-Objekt."]
    errs = []
    if job.get("version") != 2:
        errs.append("Studien brauchen \"version\": 2.")
    if job.get("task") not in TASKS:
        errs.append(f"\"task\" muss einer von {', '.join(TASKS)} sein (ist {job.get('task')!r}).")
    for key in ("model", "materials", "stack", "parameters", "configurations"):
        if key not in job:
            errs.append(f"Es fehlt \"{key}\".")
    names = [p.get("name") for p in job.get("parameters", []) if isinstance(p, dict)]
    if len(names) != len(set(names)):
        errs.append("Parameternamen müssen eindeutig sein.")
    for i, p in enumerate(job.get("parameters", [])):
        if not isinstance(p, dict) or "name" not in p or "bounds" not in p:
            errs.append(f"parameters[{i}]: braucht \"name\" und \"bounds\".")
        elif not any(k in p for k in ("trapezoid", "field", "material")):
            errs.append(f"parameters[{i}] ({p['name']}): gib \"trapezoid\", \"field\" (mit \"shape\") oder \"material\" an.")
    task = job.get("task")
    if task == "reconstruct":
        block = job.get("reconstruct", {})
        if ("measured" in block) == ("synthetic" in block):
            errs.append("reconstruct: entweder \"measured\" (Messwerte) oder \"synthetic\" (Daten aus dem Modell) angeben.")
    if task == "uq":
        inputs = job.get("uq", {}).get("inputs")
        if not inputs:
            errs.append("uq: \"inputs\" (Verteilungen der unsicheren Parameter) fehlt.")
        else:
            for n in inputs:
                if n not in names:
                    errs.append(f"uq.inputs: unbekannter Parameter {n!r}.")
    return errs


def scales_from_job(job):
    """Factor job unit -> SI per parameter, as the runner uses it: material parameters 1, trapezoid angles degrees, lengths times unit."""
    unit = float(job.get("unit", 1e-9)) if job else 1e-9
    scales = {}
    for p in (job or {}).get("parameters", []):
        if "material" in p:
            scales[p["name"]] = 1.0
        elif p.get("trapezoid") == "angle":
            scales[p["name"]] = math.pi / 180.0
        else:
            scales[p["name"]] = unit
    return scales


def load_study(folder):
    """Everything of a study folder: job, results, store, events, log text (missing parts are None / empty)."""
    folder = Path(folder)

    def _json(name):
        p = folder / name
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
        except ValueError:
            return None

    job, results = _json("job.json"), _json("results.json")
    store_path = None
    if results and results.get("study") and Path(results["study"]).is_file():
        store_path = Path(results["study"])
    else:
        found = sorted(folder.glob("*.study.jsonl"), key=lambda q: q.stat().st_mtime)
        store_path = found[-1] if found else None
    log_path = folder / "log.txt"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return {"folder": str(folder), "job": job, "results": results, "store": read_store(store_path) if store_path else read_store(""),
            "store_path": None if store_path is None else str(store_path), "events": parse_events(log), "log": log}


def info(study):
    """task, parameter names, scales (job unit -> SI), bounds (job units), observables, objective index and sense of a loaded study."""
    job, res, header = study.get("job") or {}, study.get("results") or {}, study["store"].get("header") or {}
    task = res.get("task") or job.get("task")
    if res.get("parameters"):
        names = list(res["parameters"])
        scales = {n: float(v["scale"]) for n, v in res["parameters"].items()}
    elif job.get("parameters"):
        scales = scales_from_job(job)
        names = list(scales)
    else:  # only the store: SI values
        names = [p["name"] for p in header.get("space", {}).get("parameters", [])]
        scales = {n: 1.0 for n in names}
    bounds = {p["name"]: tuple(float(b) for b in p["bounds"]) for p in job.get("parameters", []) if "bounds" in p}
    if not bounds:
        for p in header.get("space", {}).get("parameters", []):
            if "lower" in p and p["name"] in scales:
                bounds[p["name"]] = (p["lower"] / scales[p["name"]], p["upper"] / scales[p["name"]])
    observables = list(res.get("observables") or header.get("evaluator", {}).get("observables") or [])
    spec = job.get("optimize", {}) if task == "optimize" else {}
    objective = (res.get("optimize") or {}).get("objective", spec.get("objective", 0))
    if isinstance(objective, str):
        objective = observables.index(objective) if objective in observables else 0
    return {"task": task, "names": names, "scales": scales, "bounds": bounds, "observables": observables, "objective": int(objective or 0),
            "maximize": bool(spec.get("maximize", False)), "name": res.get("job") or job.get("name") or header.get("evaluator", {}).get("name")}


def history(study):
    """One row per evaluation in the order of the store: Nr, the parameters in job units, the observables, status, cache, time (s), the error
    of a failed evaluation (e.g. the mesh-quality guard of the morph). Falls
    back to the points of results.json or the evaluation events when there is no store."""
    inf = info(study)
    names, scales, obs = inf["names"], inf["scales"], inf["observables"]
    rows = []
    recs = study["store"]["evaluations"]
    if recs:
        for i, r in enumerate(recs):
            params = r.get("params") or {}
            error = r.get("error") if isinstance(r.get("error"), str) else (r.get("meta") or {}).get("error")
            rows.append(({"Nr": i} | {n: (params[n] / scales.get(n, 1.0) if params.get(n) is not None else np.nan) for n in names},
                         r.get("values"), r.get("status", "ok"), False, r.get("cost"), error))
    else:
        pts = (study.get("results") or {}).get("points") or [e for e in study.get("events", []) if e.get("event") == "evaluation"]
        for i, e in enumerate(pts):
            params = e.get("params") or {}
            rows.append(({"Nr": e.get("index", i)} | {n: params.get(n, np.nan) for n in names}, e.get("values"), e.get("status", "ok"),
                         bool(e.get("cached")), e.get("seconds"), None))
    out = []
    for base, values, status, cached, seconds, error in rows:
        values = list(values) if values is not None else []
        if not obs:
            obs = [f"Größe {k}" for k in range(len(values))]
        row = dict(base)
        for k, name in enumerate(obs):
            v = values[k] if k < len(values) else None
            row[name] = float(np.real(v)) if v is not None else np.nan
        row["Status"] = STATUS_LABELS.get(status, status)
        row["aus Cache"] = cached
        row["Zeit (s)"] = np.nan if seconds is None else float(seconds)
        row["Fehler"] = error or ""
        out.append(row)
    cols = ["Nr", *names, *obs, "Status", "aus Cache", "Zeit (s)", "Fehler"]
    return pd.DataFrame(out, columns=cols)


def progress(study):
    """(fraction 0..1 or None, text) of a study, from the events of the log and the job."""
    job = study.get("job") or {}
    task = job.get("task")
    events = study.get("events", [])
    n = sum(1 for e in events if e.get("event") == "evaluation")
    if any(e.get("event") == "cancelled" for e in events):  # the runner emits "cancelled" and then "done"
        return None, f"abgebrochen nach {n} Auswertungen"
    if any(e.get("event") == "done" for e in events):
        return 1.0, f"fertig: {n} Auswertungen"
    total = None
    spec = job.get(task, {}) if task else {}
    if task == "optimize":
        total = spec.get("max_evaluations") or (30 if spec.get("method") in ("bayesian", "bo") else None)
    elif task == "uq" and spec.get("propagation") == "surrogate" and spec.get("points"):
        total = int(spec["points"])
    if not events:
        return 0.0, "Start …"
    if total:
        return min(n / float(total), 0.999), f"{n} von höchstens {total} Auswertungen"
    return None, f"{n} Auswertungen"


# ----------------------------------------------------------------------------------------------------------------- analysis
def best_so_far(values, maximize=False):
    """Running best of a sequence (NaN entries, e.g. failed evaluations, keep the previous best)."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    best = np.nan
    for i, x in enumerate(v):
        if np.isfinite(x) and (not np.isfinite(best) or (x > best if maximize else x < best)):
            best = x
        out[i] = best
    return out


def pareto_mask(points, maximize):
    """True for the non-dominated rows of points (n x k) with the sense per column (maximize: list of bool); rows with NaN are never
    on the front. A row is dominated if another is at least as good in every column and better in one."""
    y = np.asarray(points, dtype=float)
    sign = np.where(np.asarray(maximize, dtype=bool), -1.0, 1.0)  # minimise sign * y
    z = y * sign
    ok = np.all(np.isfinite(z), axis=1)
    mask = np.zeros(len(z), dtype=bool)
    for i in np.flatnonzero(ok):
        others = z[ok]
        dominated = np.any(np.all(others <= z[i], axis=1) & np.any(others < z[i], axis=1))
        mask[i] = not dominated
    return mask


def pareto_front(df, x, y, max_x, max_y):
    """Rows of the history on the Pareto front of the observables x and y, sorted along x."""
    m = pareto_mask(df[[x, y]].to_numpy(), [max_x, max_y])
    return df[m].sort_values(x)


# ----------------------------------------------------------------------------------------------------------------- tables
def optimize_dataframe(study):
    """The optimum: parameters and observables (rows), with the reference values of the job."""
    res = study.get("results") or {}
    o = res.get("optimize")
    if not o:
        return pd.DataFrame()
    inf = info(study)
    ref = {n: v.get("reference") for n, v in (res.get("parameters") or {}).items()}
    rows = [{"Größe": n, "Optimum": o["params"].get(n), "Referenz": ref.get(n), "Art": "Parameter"} for n in inf["names"]]
    for k, name in enumerate(inf["observables"]):
        v = o.get("values")[k] if o.get("values") else None
        rows.append({"Größe": name, "Optimum": v, "Referenz": None, "Art": "Zielgröße" if k == inf["objective"] else "Messgröße"})
    return pd.DataFrame(rows)


def reconstruct_dataframe(study):
    """Fitted parameters with the Laplace standard errors and, if sampled, the posterior mean, std and 95 % interval."""
    rec = (study.get("results") or {}).get("reconstruct")
    if not rec:
        return pd.DataFrame()
    ref = {n: v.get("reference") for n, v in ((study.get("results") or {}).get("parameters") or {}).items()}
    truth = ((study.get("job") or {}).get("reconstruct", {}).get("synthetic") or {}).get("params", {})
    post = rec.get("posterior") or {}
    pnames = post.get("names", [])
    rows = []
    for n in rec.get("names", list(rec.get("params", {}))):
        row = {"Parameter": n, "Anpassung": rec["params"].get(n), "σ (Laplace)": rec.get("std", {}).get(n), "Referenz": ref.get(n)}
        if truth:
            row["wahr (synthetisch)"] = truth.get(n)
        if n in pnames:
            j = pnames.index(n)
            q = post.get("quantiles", {})
            row["Posterior Mittel"] = post["mean"][j]
            row["Posterior σ"] = post["std"][j]
            if "0.025" in q and "0.975" in q:
                row["95 % (Posterior)"] = f"{q['0.025'][j]:.6g} … {q['0.975'][j]:.6g}"
        rows.append(row)
    return pd.DataFrame(rows)


def residual_dataframe(study):
    """Measured and fitted values per observable, with the residual."""
    res = study.get("results") or {}
    rec = res.get("reconstruct")
    if not rec or rec.get("values") is None:
        return pd.DataFrame()
    obs = res.get("observables") or [f"Größe {k}" for k in range(len(rec["values"]))]
    measured = rec.get("measured") or (study.get("job") or {}).get("reconstruct", {}).get("measured")
    rows = []
    for k, name in enumerate(obs):
        rows.append({"Messgröße": name, "gemessen": measured[k] if measured else None, "angepasst": rec["values"][k],
                     "Residuum": rec["residual"][k] if rec.get("residual") else None})
    return pd.DataFrame(rows)


def uq_dataframe(study):
    """Mean (or the value of the linearisation), standard deviation and the quantiles per observable."""
    u = (study.get("results") or {}).get("uq")
    if not u:
        return pd.DataFrame()
    obs = u.get("observables", [])
    rows = []
    for k, name in enumerate(obs):
        row = {"Messgröße": name}
        if "mean" in u:
            row["Mittel"] = u["mean"][k]
        if "values" in u:
            row["Wert (linear)"] = u["values"][k]
        row["σ"] = u["std"][k]
        for q, v in sorted(u.get("quantiles", {}).items(), key=lambda t: float(t[0])):
            row[f"{float(q) * 100:g} %"] = v[k]
        if "surrogate_std" in u:
            row["σ Ersatzmodell"] = u["surrogate_std"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def sobol_dataframe(study):
    """Sobol' indices (first order and total) per observable and input; for the linear propagation the variance shares."""
    u = (study.get("results") or {}).get("uq") or {}
    s = u.get("sobol")
    if not s:
        return pd.DataFrame()
    rows = []
    for k, name in enumerate(u.get("observables", [])):
        for j, inp in enumerate(u.get("inputs", [])):
            row = {"Messgröße": name, "Eingang": inp, "S₁ (erste Ordnung)": s["first"][k][j], "S_T (total)": s["total"][k][j]}
            if "first_conf" in s:
                row["± S₁"] = s["first_conf"][k][j]
                row["± S_T"] = s["total_conf"][k][j]
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------- figures
def _empty(text):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    return fig


def fig_history(df, observable, maximize=False, optimum=None):
    """Values of one observable over the evaluations, with the running best (minimum or maximum)."""
    import matplotlib.pyplot as plt

    if df.empty or observable not in df:
        return _empty("noch keine Auswertungen")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ok = df["Status"] == "ok"
    ax.plot(df["Nr"][ok], df[observable][ok], "o", ms=4, color="#1f77b4", label="Auswertung")
    if (~ok).any():
        ax.plot(df["Nr"][~ok], np.full((~ok).sum(), np.nanmin(df[observable]) if ok.any() else 0.0), "x", color="#d62728",
                label="fehlgeschlagen")
    ax.step(df["Nr"], best_so_far(df[observable].where(ok), maximize), where="post", color="#ff7f0e",
            label="bisher bestes (" + ("Maximum" if maximize else "Minimum") + ")")
    if optimum is not None:
        ax.axhline(optimum, color="0.3", lw=0.8, ls="--", label="Ergebnis")
    ax.set_xlabel("Auswertung Nr.")
    ax.set_ylabel(observable)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_parameters(df, names, bounds=None, best=None):
    """The parameters of every evaluation (one panel each), with the bounds and the result."""
    import matplotlib.pyplot as plt

    if df.empty or not names:
        return _empty("noch keine Auswertungen")
    fig, axs = plt.subplots(len(names), 1, figsize=(8, 1.9 * len(names) + 0.6), sharex=True, squeeze=False)
    for ax, n in zip(axs[:, 0], names, strict=True):
        ax.plot(df["Nr"], df[n], "o-", ms=3, lw=0.6, color="#2ca02c")
        for b in (bounds or {}).get(n, ()):
            ax.axhline(b, color="0.5", lw=0.6, ls=":")
        if best and best.get(n) is not None:
            ax.axhline(best[n], color="#ff7f0e", lw=1.0, ls="--")
        ax.set_ylabel(n)
        ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("Auswertung Nr.")
    fig.tight_layout()
    return fig


def fig_scatter(df, x, y, color=None):
    """Two parameters (or a parameter and an observable) of all evaluations, coloured by a third column."""
    import matplotlib.pyplot as plt

    if df.empty:
        return _empty("noch keine Auswertungen")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    if color and color in df:
        sc = ax.scatter(df[x], df[y], c=df[color], cmap="viridis", s=28, edgecolor="k", lw=0.3)
        fig.colorbar(sc, ax=ax, label=color)
    else:
        ax.scatter(df[x], df[y], s=28, color="#1f77b4")
    for nr, a, b in zip(df["Nr"], df[x], df[y], strict=True):
        ax.annotate(str(nr), (a, b), fontsize=6, xytext=(2, 2), textcoords="offset points", color="0.35")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_pareto(df, x, y, max_x, max_y):
    """All evaluations in the plane of two observables, the Pareto front (non-dominated evaluations) highlighted and joined."""
    import matplotlib.pyplot as plt

    if df.empty or x not in df or y not in df:
        return _empty("noch keine Auswertungen")
    front = pareto_front(df[df["Status"] == "ok"], x, y, max_x, max_y)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(df[x], df[y], s=22, color="0.65", label="Auswertungen")
    ax.plot(front[x], front[y], "o-", color="#d62728", ms=6, lw=1.2, label=f"Pareto-Front ({len(front)})")
    for nr, a, b in zip(front["Nr"], front[x], front[y], strict=True):
        ax.annotate(str(nr), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"{x} ({'max' if max_x else 'min'})")
    ax.set_ylabel(f"{y} ({'max' if max_y else 'min'})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_sobol(study):
    """Sobol' bars: first-order and total index per input, one panel per observable (with the confidence of the estimate)."""
    import matplotlib.pyplot as plt

    u = (study.get("results") or {}).get("uq") or {}
    s = u.get("sobol")
    if not s:
        return _empty("keine Sobol'-Indizes (uq.sobol = true)")
    obs, inputs = u.get("observables", []), u.get("inputs", [])
    first, total = np.asarray(s["first"], dtype=float), np.asarray(s["total"], dtype=float)
    fconf = np.asarray(s.get("first_conf", np.zeros_like(first)), dtype=float)
    tconf = np.asarray(s.get("total_conf", np.zeros_like(total)), dtype=float)
    ncol = min(len(obs), 3)
    nrow = int(np.ceil(len(obs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
    xs = np.arange(len(inputs))
    for k, name in enumerate(obs):
        ax = axs[k // ncol, k % ncol]
        ax.bar(xs - 0.2, first[k], 0.4, yerr=fconf[k] if fconf.any() else None, color="#1f77b4", capsize=2, label="S₁ erste Ordnung")
        ax.bar(xs + 0.2, total[k], 0.4, yerr=tconf[k] if tconf.any() else None, color="#ff7f0e", capsize=2, label="S_T total")
        ax.set_xticks(xs, inputs, rotation=20, ha="right", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
    for k in range(len(obs), nrow * ncol):
        axs[k // ncol, k % ncol].axis("off")
    axs[0, 0].legend(fontsize=7)
    if u.get("propagation") == "linear":
        fig.suptitle("Varianzanteile der Linearisierung (erste Ordnung = total)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_uq(study):
    """Spread of every observable: the 2.5-97.5 % and 16-84 % intervals and the median (Monte Carlo on the surrogate), or value +- 1 and
    2 standard deviations of the linearisation."""
    import matplotlib.pyplot as plt

    u = (study.get("results") or {}).get("uq") or {}
    if not u:
        return _empty("kein UQ-Ergebnis")
    obs = u.get("observables", [])
    fig, axs = plt.subplots(1, len(obs), figsize=(2.6 * len(obs) + 0.8, 3.4), squeeze=False)
    q = u.get("quantiles", {})
    for k, (ax, name) in enumerate(zip(axs[0], obs, strict=True)):
        if q:
            lo2, lo1, med, hi1, hi2 = (q[key][k] for key in ("0.025", "0.16", "0.5", "0.84", "0.975"))
        else:
            c, s = u["values"][k], u["std"][k]
            lo2, lo1, med, hi1, hi2 = c - 2 * s, c - s, c, c + s, c + 2 * s
        ax.vlines(0, lo2, hi2, color="#1f77b4", lw=2)
        ax.vlines(0, lo1, hi1, color="#1f77b4", lw=7)
        ax.plot(0, med, "o", color="white", mec="#1f77b4", ms=7)
        if "mean" in u:
            ax.plot(0.15, u["mean"][k], "d", color="#ff7f0e", ms=5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(name, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.ticklabel_format(axis="y", useOffset=False)
    fig.suptitle("Streuung: Strich 95 %, Balken 68 %, Kreis Median" + (" (Raute: Mittel)" if "mean" in u else " (linear: ±σ, ±2σ)"), fontsize=9)
    fig.tight_layout()
    return fig


def fig_posterior(study):
    """Marginals of the reconstructed parameters: the Laplace Gaussian (fit +- sigma) and, if sampled, the posterior (emcee) as a Gaussian
    of its mean and std with the 68 % and 95 % quantile intervals; the synthetic truth as a line."""
    import matplotlib.pyplot as plt

    res = study.get("results") or {}
    rec = res.get("reconstruct")
    if not rec:
        return _empty("kein Rekonstruktionsergebnis")
    names = rec.get("names", list(rec.get("params", {})))
    post = rec.get("posterior") or {}
    pnames = post.get("names", [])
    truth = ((study.get("job") or {}).get("reconstruct", {}).get("synthetic") or {}).get("params", {})
    fig, axs = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.2), squeeze=False)
    for ax, n in zip(axs[0], names, strict=True):
        mu, sd = rec["params"][n], rec.get("std", {}).get(n)
        spans = [mu]
        if sd and np.isfinite(sd) and sd > 0:
            spans += [mu - 4 * sd, mu + 4 * sd]
        if n in pnames:
            j = pnames.index(n)
            spans += [post["mean"][j] - 4 * post["std"][j], post["mean"][j] + 4 * post["std"][j]]
        lo, hi = min(spans), max(spans)
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        x = np.linspace(lo, hi, 400)
        if sd and np.isfinite(sd) and sd > 0:
            ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color="#1f77b4", label="Laplace")
        ax.axvline(mu, color="#1f77b4", lw=0.8, ls="--")
        if n in pnames:
            j = pnames.index(n)
            m, s = post["mean"][j], post["std"][j]
            if s > 0:
                ax.plot(x, np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)), color="#ff7f0e", label="Posterior (emcee)")
            qs = post.get("quantiles", {})
            if "0.025" in qs and "0.975" in qs:
                ax.axvspan(qs["0.025"][j], qs["0.975"][j], color="#ff7f0e", alpha=0.10)
            if "0.16" in qs and "0.84" in qs:
                ax.axvspan(qs["0.16"][j], qs["0.84"][j], color="#ff7f0e", alpha=0.18)
        if n in truth:
            ax.axvline(truth[n], color="k", lw=1.2, label="wahr")
        ax.set_xlabel(n)
        ax.set_yticks([])
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_correlation(study):
    """Correlation matrix of the Laplace approximation."""
    import matplotlib.pyplot as plt

    rec = (study.get("results") or {}).get("reconstruct")
    if not rec or not rec.get("correlation"):
        return _empty("keine Korrelationen")
    c = np.asarray(rec["correlation"], dtype=float)
    names = rec.get("names", [])
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2.2, 1.0 * len(names) + 1.6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(c[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
