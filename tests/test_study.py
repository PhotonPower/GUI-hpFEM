"""Study view (hp-FEM M16 S9b): reading study folders of python -m hpfem.run, history, Pareto front, tables and figures. Pure Python, no hpfem;
the folders mimic the files the runner writes (results.json, <name>.study.jsonl, log.txt with JSON-lines events)."""
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import fem_study as fs

OBS = ["s 405nm 40deg: R0", "s 405nm 40deg: T0", "p 500nm 20deg: R0"]
PARAMS = {"width": {"scale": 1e-9, "reference": 200.0}, "eps": {"scale": 1.0, "reference": 2.25}}


def write_study(folder, task, block, evaluations, job_extra=None, truncated=False, results=True):
    """A study folder: job.json, the store (values SI), log.txt with events, results.json."""
    job = fs.template(task)
    job["name"] = "demo"
    job[task] = job_extra or job[task]
    (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
    store = folder / "demo.study.jsonl"
    header = {"type": "study", "schema": 1, "space": {"parameters": [
        {"kind": "continuous", "name": "width", "lower": 1.7e-7, "upper": 2.3e-7},
        {"kind": "continuous", "name": "eps", "lower": 2.0, "upper": 2.6}]}, "evaluator": {"name": "demo", "observables": OBS}}
    lines = [json.dumps(header), json.dumps({"type": "state", "method": "x", "iteration": 0, "evaluations": 0, "state": {}})]
    errors = {"failed": "MeshQualityError: Morph: 2 inverted cells"}
    events = [{"event": "start", "job": "demo", "points": None}]
    for i, (w, e, values, status) in enumerate(evaluations):
        vals = [v if np.isfinite(v) else None for v in values]
        lines.append(json.dumps({"type": "evaluation", "params": {"width": w * 1e-9, "eps": e}, "values": vals, "status": status,
                                 "cost": 0.5, "fidelity": {}, "jacobian": None, "meta": {"error": errors[status]} if status in errors else {}}))
        events.append({"event": "evaluation", "index": i, "params": {"width": w, "eps": e}, "values": vals, "status": status,
                       "cached": False, "seconds": 0.5})
    text = "\n".join(lines) + "\n"
    if truncated:
        text += '{"type": "evaluation", "params": {"wid'
    store.write_text(text, encoding="utf-8")
    log = "\n".join(json.dumps(e) for e in events) + "\nsome library warning\n"
    if results:
        res = {"version": 2, "job": "demo", "task": task, "study": str(store), "parameters": PARAMS, "observables": OBS, "cancelled": False,
               "points": [e for e in events if e["event"] == "evaluation"], task: block}
        (folder / "results.json").write_text(json.dumps(res), encoding="utf-8")
        log += json.dumps({"event": "done", "results": str(folder / "results.json")}) + "\n"
    (folder / "log.txt").write_text(log, encoding="utf-8")
    return folder


EVALS = [(183.0, 2.28, [0.035, 0.82, 0.0075], "ok"), (193.5, 2.17, [0.030, 0.85, 0.0070], "ok"),
         (219.0, 2.52, [0.060, 0.78, 0.0091], "ok"), (178.5, 2.32, [np.nan, np.nan, np.nan], "failed"),
         (215.4, 2.60, [0.066, 0.785, 0.0095], "ok")]


def test_history_from_the_store_in_job_units(tmp_path):
    block = {"method": "bayesian", "objective": 0, "params": {"width": 215.4, "eps": 2.6}, "value": 0.066, "values": [0.066, 0.785, 0.0095],
             "success": True, "message": "max", "evaluations": 5, "new_evaluations": 5, "cache_hits": 0}
    st_ = fs.load_study(write_study(tmp_path, "optimize", block, EVALS, truncated=True))
    inf = fs.info(st_)
    assert inf["task"] == "optimize" and inf["names"] == ["width", "eps"] and inf["observables"] == OBS
    assert inf["maximize"] and inf["objective"] == 0 and inf["bounds"]["width"] == (180.0, 215.0)
    df = fs.history(st_)
    assert len(df) == 5  # the truncated last line is skipped
    assert df["width"].tolist() == pytest.approx([e[0] for e in EVALS])  # SI -> nm
    assert df["Status"].tolist()[3] == "fehlgeschlagen" and np.isnan(df[OBS[0]][3])
    assert df["Fehler"][3].startswith("MeshQualityError") and df["Fehler"][0] == ""
    assert fs.progress(st_) == (1.0, "fertig: 5 Auswertungen")
    opt = fs.optimize_dataframe(st_)
    assert opt.set_index("Größe").loc["width", "Optimum"] == pytest.approx(215.4)
    assert opt.set_index("Größe").loc[OBS[0], "Art"] == "Zielgröße"
    for fig in (fs.fig_history(df, OBS[0], True, 0.066), fs.fig_parameters(df, inf["names"], inf["bounds"], block["params"]),
                fs.fig_scatter(df, "width", "eps", OBS[0])):
        assert fig is not None


def test_running_study_without_results_and_progress(tmp_path):
    st_ = fs.load_study(write_study(tmp_path, "optimize", {}, EVALS[:2], job_extra={"method": "L-BFGS-B", "max_evaluations": 8},
                                    results=False))
    assert st_["results"] is None
    inf = fs.info(st_)
    assert inf["names"] == ["width", "eps"] and inf["observables"] == OBS  # from the job and the store header
    df = fs.history(st_)
    assert df["width"].tolist() == pytest.approx([183.0, 193.5])
    frac, label = fs.progress(st_)
    assert frac == pytest.approx(2 / 8) and "2 von höchstens 8" in label


def test_best_so_far_and_pareto():
    assert fs.best_so_far([3, np.nan, 1, 2, 0.5], maximize=False) == pytest.approx([3, 3, 1, 1, 0.5])
    assert fs.best_so_far([1, 3, 2], maximize=True) == pytest.approx([1, 3, 3])
    pts = np.array([[1.0, 5.0], [2.0, 4.0], [1.5, 4.5], [2.0, 2.0], [0.5, 1.0], [np.nan, 9.0], [2.0, 4.0]])
    # maximise both: (2,4) twice (neither dominates the other), (1.5,4.5) and (1,5) are the front; (2,2) and (0.5,1) are dominated
    assert fs.pareto_mask(pts, [True, True]).tolist() == [True, True, True, False, False, False, True]
    # minimise the first, maximise the second: (0.5, 1) is best in x, (1, 5) in y; everything else is dominated by (1, 5)
    assert fs.pareto_mask(pts, [False, True]).tolist() == [True, False, False, False, True, False, False]


def test_pareto_front_on_the_history(tmp_path):
    st_ = fs.load_study(write_study(tmp_path, "optimize", {}, EVALS, results=False))
    df = fs.history(st_)
    front = fs.pareto_front(df[df["Status"] == "ok"], OBS[0], OBS[1], True, True)
    # max R0 and max T0: the evaluations 1 (best T), 0 (between) and 4 (best R) trade off; 2 is dominated by 4
    assert front["Nr"].tolist() == [1, 0, 4]
    assert fs.fig_pareto(df, OBS[0], OBS[1], True, True) is not None


def test_reconstruct_tables_and_posterior(tmp_path):
    block = {"measured": [0.054, 0.826, 0.0091], "params": {"width": 214.1, "eps": 2.344}, "std": {"width": 4.07, "eps": 0.0133},
             "correlation": [[1.0, 0.256], [0.256, 1.0]], "names": ["width", "eps"], "chi2_red": 0.066, "cost": 0.033, "at_bounds": [],
             "warnings": [], "iterations": 7, "evaluations": 7, "success": True, "message": "converged", "values": [0.0543, 0.8261, 0.0086],
             "residual": [6e-5, 1e-5, -5e-4],
             "posterior": {"names": ["width", "eps"], "mean": [214.9, 2.343], "std": [4.2, 0.0139],
                           "quantiles": {"0.025": [207.2, 2.314], "0.16": [210.4, 2.329], "0.5": [214.8, 2.343], "0.84": [219.0, 2.357],
                                         "0.975": [223.1, 2.371]}, "acceptance": 0.6, "autocorr": [19.3, 18.0], "walkers": 8, "steps": 300}}
    extra = {"synthetic": {"params": {"width": 210, "eps": 2.3}, "noise": 0.002}}
    st_ = fs.load_study(write_study(tmp_path, "reconstruct", block, EVALS[:3], job_extra=extra))
    df = fs.reconstruct_dataframe(st_).set_index("Parameter")
    assert df.loc["width", "σ (Laplace)"] == pytest.approx(4.07) and df.loc["eps", "wahr (synthetisch)"] == pytest.approx(2.3)
    assert df.loc["width", "Posterior σ"] == pytest.approx(4.2) and df.loc["width", "95 % (Posterior)"] == "207.2 … 223.1"
    r = fs.residual_dataframe(st_)
    assert r["gemessen"].tolist() == pytest.approx(block["measured"]) and r["Messgröße"].tolist() == OBS
    assert fs.fig_posterior(st_) is not None and fs.fig_correlation(st_) is not None
    del block["posterior"]
    st_ = fs.load_study(write_study(tmp_path, "reconstruct", block, EVALS[:3], job_extra=extra))
    assert "Posterior σ" not in fs.reconstruct_dataframe(st_) and fs.fig_posterior(st_) is not None


def test_uq_tables_and_sobol(tmp_path):
    q = {"0.025": [0.038, 0.812, 0.0076], "0.16": [0.041, 0.817, 0.0077], "0.5": [0.045, 0.827, 0.0079], "0.84": [0.049, 0.836, 0.0081],
         "0.975": [0.053, 0.841, 0.0083]}
    block = {"propagation": "surrogate", "inputs": ["width", "eps"], "observables": OBS, "mean": [0.045, 0.827, 0.0079], "std": [0.0039, 0.0085, 0.00019],
             "quantiles": q, "clipped": 1, "surrogate_std": [1e-5, 1e-5, 2e-6],
             "sobol": {"first": [[0.42, 0.58], [0.08, 0.91], [0.79, 0.20]], "total": [[0.43, 0.59], [0.09, 0.92], [0.80, 0.21]],
                       "first_conf": [[0.02, 0.02]] * 3, "total_conf": [[0.02, 0.02]] * 3, "samples": 1024}}
    st_ = fs.load_study(write_study(tmp_path, "uq", block, EVALS[:3]))
    u = fs.uq_dataframe(st_).set_index("Messgröße")
    assert u.loc[OBS[1], "σ"] == pytest.approx(0.0085) and u.loc[OBS[0], "97.5 %"] == pytest.approx(0.053)
    s = fs.sobol_dataframe(st_)
    assert len(s) == 6 and s[(s["Messgröße"] == OBS[1]) & (s["Eingang"] == "eps")]["S₁ (erste Ordnung)"].item() == pytest.approx(0.91)
    fig = fs.fig_sobol(st_)
    assert len([a for a in fig.axes if a.get_visible() and a.axison]) == 3
    assert fs.fig_uq(st_) is not None
    # the linear propagation: values +- std, variance shares as indices
    lin = {"propagation": "linear", "inputs": ["width"], "observables": OBS, "values": [0.045, 0.827, 0.0079], "std": [0.004, 0.008, 0.0002],
           "covariance": [], "jacobian": [], "contributions": [[1.0], [1.0], [1.0]], "sobol": {"first": [[1.0]] * 3, "total": [[1.0]] * 3}}
    st_ = fs.load_study(write_study(tmp_path, "uq", lin, EVALS[:1]))
    assert "Wert (linear)" in fs.uq_dataframe(st_) and fs.fig_uq(st_) is not None and fs.fig_sobol(st_) is not None


def test_templates_and_job_checks():
    for task in fs.TASKS:
        assert fs.check_job(fs.template(task)) == []
    job = fs.template("uq")
    job["uq"]["inputs"]["height"] = {"normal": [100, 1]}
    assert any("height" in e for e in fs.check_job(job))
    job = fs.template("reconstruct")
    job["reconstruct"]["measured"] = [0.1, 0.8, 0.01]
    assert any("entweder" in e for e in fs.check_job(job))
    job = fs.template("optimize")
    job["version"] = 1
    job["parameters"].append({"name": "width", "bounds": [1, 2], "field": "x", "shape": 0})
    errs = fs.check_job(job)
    assert any("version" in e for e in errs) and any("eindeutig" in e for e in errs)
    assert fs.check_job([1]) == ["Das Dokument ist kein JSON-Objekt."]
    assert fs.scales_from_job({"unit": 1e-6, "parameters": [{"name": "a", "trapezoid": "angle"}, {"name": "w", "field": "width"},
                                                            {"name": "e", "material": 3}]}) == pytest.approx({"a": np.pi / 180, "w": 1e-6, "e": 1.0})


def test_store_decoding(tmp_path):
    p = tmp_path / "x.study.jsonl"
    p.write_text('{"type": "study"}\n{"type": "evaluation", "values": [{"__complex__": [1, 2]}, "inf", null]}\nnot json\n'
                 '{"type": "remesh", "mesh_id": 1}\n', encoding="utf-8")
    s = fs.read_store(p)
    assert s["evaluations"][0]["values"][0] == 1 + 2j and s["evaluations"][0]["values"][1] == np.inf and len(s["remeshes"]) == 1
    assert fs.parse_events('{"event": "start"}\n[warning] x\n{"event": "do') == [{"event": "start"}]
