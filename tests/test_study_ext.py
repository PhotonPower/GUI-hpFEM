"""Erweiterte Studien: Nebenbedingungen, Mehrziel (Pareto), Mehrstufen; Studien-Worker (kurze Rechnungen; die BoTorch-Läufe nur mit BoTorch)."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_run as fr  # noqa: E402
import fem_study as fs  # noqa: E402


def job_with(**spec):
    job = fs.template("optimize")
    job["optimize"].update(spec)
    return job


def synthetic_study(job, rows, results=None):
    """A loaded study (fs.load_study layout) with SI parameters in the store: rows = (width_nm, eps, [values])."""
    recs = [{"params": {"width": w * 1e-9, "eps": e}, "values": v, "status": "ok", "cost": 1.0} for w, e, v in rows]
    return {"folder": "x", "job": job, "results": results, "store": {"header": {}, "evaluations": recs}, "store_path": None, "events": [], "log": ""}


def test_labels_match_the_library_names():
    assert fs.observable_labels(fs.template("optimize")) == ["s 405nm 40deg: R0", "s 405nm 40deg: T0", "p 500nm 20deg: R0"]
    j = fs.template("optimize")
    j["configurations"][1]["phi_deg"] = 30
    assert fs.observable_labels(j)[2] == "p 500nm 20deg/30deg: R0"


def test_check_job_extended_keys():
    ok = fs.template_constrained()
    assert fs.check_job(ok) == []
    assert fs.check_job(fs.template("optimize")) == []
    bad = job_with(constraints=[{"observable": "gibt es nicht", "upper": 1}])
    assert any("keine der Messgrößen" in e for e in fs.check_job(bad))
    assert any("nur mit method" in e for e in fs.check_job(job_with(method="L-BFGS-B", constraints=[{"observable": 0, "upper": 1}])))
    assert any("unbekannter Parameter" in e for e in fs.check_job(job_with(linear_constraints=[{"coefficients": {"nope": 1}, "upper": 1}])))
    assert any("größer als upper" in e for e in fs.check_job(job_with(linear_constraints=[{"coefficients": {"width": 1}, "lower": 5, "upper": 1}])))
    assert any("mindestens zwei" in e for e in fs.check_job(job_with(method="pareto", objectives=[0])))
    assert fs.check_job(job_with(method="pareto", objectives=[0, 2], maximize=[True, False], ref_point=[0.0, 1.0])) == []
    assert any("ref_point" in e for e in fs.check_job(job_with(method="pareto", objectives=[0, 2], ref_point=[0.0])))
    assert any("Stufen" in e for e in fs.check_job(job_with(method="multi_fidelity", fidelities=[{"order": 2}])))
    assert any("optimize.method" in e for e in fs.check_job(job_with(method="quark")))


def test_apply_extended_roundtrip_and_errors():
    job = fs.template("optimize")
    labels = fs.observable_labels(job)
    new = fs.apply_extended(job, method="bayesian", max_evaluations=12, outcome=[(labels[1], None, 0.5)], linear=[("width: 1, eps: 20", 230.0, None)])
    spec = new["optimize"]
    assert spec["constraints"] == [{"observable": labels[1], "name": "g0", "upper": 0.5}]
    assert spec["linear_constraints"][0]["coefficients"] == {"width": 1.0, "eps": 20.0} and spec["linear_constraints"][0]["lower"] == 230.0
    assert spec["max_evaluations"] == 12 and fs.check_job(new) == [] and fr.study_needs_worker(new)
    assert "constraints" not in job["optimize"]                                  # the input stays unchanged
    again = fs.apply_extended(new, method="bayesian", outcome=[], linear=[])    # the form without rows removes the old keys
    assert "constraints" not in again["optimize"] and not fr.study_needs_worker(again)
    p = fs.apply_extended(job, method="pareto", objectives=[labels[0], labels[2]], maximize=[True, False], ref_point="0, 1", outcome=[])
    assert p["optimize"]["objectives"] == [labels[0], labels[2]] and p["optimize"]["maximize"] == [True, False] and p["optimize"]["ref_point"] == [0.0, 1.0]
    m = fs.apply_extended(job, method="multi_fidelity", fidelities="2, 3, 4")
    assert m["optimize"]["fidelities"] == [{"order": 2}, {"order": 3}, {"order": 4}]
    for kw, msg in ((dict(method="bayesian", outcome=[("x", 0, 1)]), "keine Messgröße"), (dict(method="bayesian", outcome=[(labels[0], None, None)]), "Grenze"),
                    (dict(method="bayesian", linear=[("width 1", 0, 1)]), "Parameter: Zahl"), (dict(method="pareto", objectives=[labels[0]], maximize=[True]), "mindestens zwei")):
        with pytest.raises(ValueError, match=msg):
            fs.apply_extended(job, **kw)


def test_history_marks_constraint_violations():
    job = fs.template_constrained()                      # R(p, 500 nm) <= 0.9 and width in [185, 210]
    rows = [(190.0, 2.2, [0.30, 0.50, 0.70]), (200.0, 2.3, [0.40, 0.40, 0.95]), (215.0, 2.3, [0.50, 0.30, 0.60]), (205.0, 2.4, [0.20, 0.60, 0.80])]
    study = synthetic_study(job, rows)
    df = fs.history(study)
    assert list(df["Zulässig"]) == [True, False, False, True]
    viol = fs.constraint_violations(job, df, fs.info(study)["observables"] or fs.observable_labels(job))
    assert {k: list(v) for k, v in viol.items()} == {"R500 <= 0.9": [False, True, False, False], "Breite 185-210 nm": [False, False, True, False]}
    assert "Zulässig" not in fs.history(synthetic_study(fs.template("optimize"), rows)).columns       # old jobs: unchanged columns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obs = fs.observable_labels(job)
    plt.close(fs.fig_history(df, obs[0], True))
    plt.close(fs.fig_pareto(df, obs[0], obs[1], True, True, ref=(0.0, 0.0)))
    inf = fs.info(synthetic_study(job, rows))
    front = fs.history_front(df, inf, [0, 1], [True, True])
    assert list(front["Nr"]) == [3, 0] or list(front["Nr"]) == [0, 3]            # only feasible, non-dominated rows
    assert set(front["Nr"]) <= {0, 3}


def test_pareto_results_tables():
    job = job_with(method="pareto", objectives=["s 405nm 40deg: R0", "p 500nm 20deg: R0"], maximize=[True, False], max_evaluations=None)
    obs = fs.observable_labels(job)
    res = {"task": "optimize", "parameters": {"width": {"scale": 1e-9, "reference": 200.0}, "eps": {"scale": 1.0, "reference": 2.25}}, "observables": obs,
           "optimize": {"method": "pareto", "objective": obs[0], "objectives": [obs[0], obs[2]], "maximize": [True, False], "ref_point": [0.0, 1.0],
                        "front": [{"params": {"width": 190.0, "eps": 2.2}, "values": [0.3, 0.5, 0.7], "objectives": [0.3, 0.7]},
                                  {"params": {"width": 205.0, "eps": 2.4}, "values": [0.5, 0.4, 0.9], "objectives": [0.5, 0.9]}],
                        "params": {"width": 190.0, "eps": 2.2}, "value": 0.3, "values": [0.3, 0.5, 0.7], "success": True, "evaluations": 9,
                        "new_evaluations": 9, "cache_hits": 0, "message": ""}}
    study = synthetic_study(job, [(190.0, 2.2, [0.3, 0.5, 0.7]), (205.0, 2.4, [0.5, 0.4, 0.9]), (200.0, 2.3, [0.2, 0.6, 0.95])], res)
    inf = fs.info(study)
    assert inf["method"] == "pareto" and inf["objectives"] == [0, 2] and inf["maximize_list"] == [True, False]
    fd = fs.front_dataframe(study)
    assert len(fd) == 2 and list(fd["width"]) == [190.0, 205.0] and fd[obs[2]].iloc[1] == pytest.approx(0.9)
    df = fs.optimize_dataframe(study)
    assert set(df[df["Art"] == "Zielgröße"]["Größe"]) == {obs[0], obs[2]}
    assert fs.progress(dict(study, events=[{"event": "evaluation"}] * 5))[1].endswith("40 Auswertungen")        # default budget of the multi-objective methods


def test_study_needs_worker_and_start_choice(tmp_path, monkeypatch):
    assert not fr.study_needs_worker(fs.template("optimize")) and not fr.study_needs_worker(fs.template("uq"))
    assert fr.study_needs_worker(fs.template_constrained()) and fr.study_needs_worker(job_with(method="pareto", objectives=[0, 1]))
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace())
    for job in (fs.template("optimize"), fs.template_constrained()):
        (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
        fr.start_study(sys.executable, None, tmp_path)
    assert calls[0][1:3] == ["-m", "hpfem.run"] and Path(calls[1][1]).name == "fem_study_worker.py"      # only extended jobs use the worker


def test_worker_conversion_and_missing_botorch(monkeypatch):
    opt = pytest.importorskip("hpfem.opt")
    pytest.importorskip("hpfem.run")
    import fem_study_worker as sw
    from hpfem.run import JobError
    spec = fs.template_constrained()["optimize"]
    lin = sw.linear_constraints(spec, {"width": 1e-9, "eps": 1.0})
    assert lin[0].coefficients == {"width": pytest.approx(1e9)} and lin[0].lower == 185 and lin[0].upper == 210
    oc = sw.outcome_constraints(spec, fs.observable_labels(fs.template("optimize")))
    assert isinstance(oc[0], opt.OutcomeConstraint) and oc[0].upper == 0.9 and oc[0].lower == -np.inf
    with pytest.raises(JobError, match="not one of"):
        sw.outcome_constraints({"constraints": [{"observable": "x", "upper": 1}]}, ["a"])
    assert sw.is_extended(spec) and not sw.is_extended({"method": "bayesian"}) and sw.is_extended({"method": "pareto"})
    space = opt.DesignSpace([opt.Continuous("width", 1.8e-7, 2.15e-7), opt.Continuous("eps", 2.0, 2.6)])
    study = SimpleNamespace(space=space, observables=["a", "b"], note=lambda *a, **k: None, cache_hits=0)
    model = SimpleNamespace(scales={"width": 1e-9, "eps": 1.0})
    monkeypatch.setattr(sw, "has_botorch", lambda: False)
    for method, extra in (("pareto", {"objectives": ["a", "b"]}), ("multi_fidelity", {"fidelities": [{"order": 2}, {"order": 3}]})):
        with pytest.raises(JobError, match=r"hpfem\[opt-bo\]"):
            sw.run_extended(study, model, {"method": method, **extra}, lambda *a: {})
    assert len(study.space.constraints) == 0                                     # no linear constraint was given


def test_worker_short_run_with_constraints(tmp_path):
    pytest.importorskip("hpfem.run")
    job = fs.template_constrained()
    job["name"] = "kurz"
    job["optimize"]["max_evaluations"] = 6                                       # the initial design only: about 10 s
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, str(HERE / "fem_study_worker.py"), str(tmp_path / "job.json"), "--out", str(tmp_path), "--quiet"],
                       capture_output=True, text=True, timeout=110)
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-600:]
    study = fs.load_study(tmp_path)
    o = study["results"]["optimize"]
    assert o["method"] == "bayesian" and o["evaluations"] == 6 and o["constraints"][0]["upper"] == 0.9 and "feasible" in o
    df = fs.history(study)
    assert len(df) == 6 and "Zulässig" in df
    assert df["width"].between(185 - 1e-6, 210 + 1e-6).all()                      # the linear constraint holds for every evaluation
    assert all(e["event"] != "error" for e in fs.parse_events(study["log"]))


def test_worker_unknown_option_reports_error_event(tmp_path):
    pytest.importorskip("hpfem.run")
    job = job_with(method="bayesian", constraints=[{"observable": "x", "upper": 1}], max_evaluations=2)
    job["name"] = "fehler"
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, str(HERE / "fem_study_worker.py"), str(tmp_path / "job.json"), "--out", str(tmp_path), "--quiet"],
                       capture_output=True, text=True, timeout=110)
    assert p.returncode == 1
    errs = [e for e in fs.parse_events(p.stdout) if e.get("event") == "error"]
    assert errs and "optimize.constraints" in errs[0]["text"]


def test_worker_short_pareto_run_with_botorch(tmp_path):
    pytest.importorskip("botorch")
    pytest.importorskip("hpfem.run")
    job = fs.template("optimize")
    labels = fs.observable_labels(job)
    job = fs.apply_extended(job, method="pareto", max_evaluations=7, objectives=[labels[0], labels[2]], maximize=[True, False],
                            linear=[("width: 1", 185, 210)])
    job["name"] = "pareto-kurz"
    job["optimize"].update(raw_samples=16, num_restarts=1)                      # one BoTorch step after the six initial points, about 30 s
    assert fs.check_job(job) == []
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, str(HERE / "fem_study_worker.py"), str(tmp_path / "job.json"), "--out", str(tmp_path), "--quiet"],
                       capture_output=True, text=True, timeout=115)
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-600:]
    study = fs.load_study(tmp_path)
    o = study["results"]["optimize"]
    assert o["method"] == "pareto" and o["objectives"] == [labels[0], labels[2]] and o["maximize"] == [True, False] and o["evaluations"] == 7
    front = fs.front_dataframe(study)
    assert 1 <= len(front) <= 7 and front["width"].between(185 - 1e-6, 210 + 1e-6).all()
    hist = fs.history(study)
    ref = fs.history_front(hist, fs.info(study))
    assert len(ref) == len(front)                                                # the front of the result equals the one recomputed from the history
