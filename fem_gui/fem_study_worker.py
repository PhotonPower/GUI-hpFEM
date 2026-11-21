"""Worker of the extended studies (hp-FEM 0.5): runs in the Python that has hpfem, called by the app instead of ``python -m hpfem.run``.

    python fem_study_worker.py <job.json> --out DIR [--cancel-file F] [--threads N] [--quiet]

The job runner of hpfem (hpfem.run, schema 2) runs a study with ONE objective and no constraints. This worker is the runner itself (same arguments,
same JSON-lines events on stdout, same results.json and study store) with one function replaced: the optimisation step of the task "optimize"
(hpfem.run._optimize). Jobs without the extended keys go to the original function unchanged. Extended keys of the block ``optimize`` (values in job units
like the parameters):

* ``constraints``: list of ``{"observable": name or index, "lower": x, "upper": y, "name": ..}``: unknown constraints on the observables
  (hpfem.opt.OutcomeConstraint, Gaussian process per constraint, expected improvement times the probability of feasibility). method ``bayesian`` only.
* ``linear_constraints``: list of ``{"coefficients": {parameter: c, ..}, "lower": x, "upper": y, "name": ..}``: ``lower <= sum c * parameter <= upper`` on
  the parameters in job units (hpfem.opt.LinearConstraint on the design space, all methods).
* ``method`` ``pareto``: multi-objective Bayesian optimisation (hpfem.opt.pareto_optimize, BoTorch, extra ``opt-bo``) with ``objectives`` (names or
  indices), ``maximize`` (bool or list), ``ref_point`` (objective units, optional), ``max_evaluations``, ``n_initial``, ``seed``.
* ``method`` ``multi_fidelity``: hpfem.opt.multi_fidelity_optimize (BoTorch) with ``fidelities`` (e.g. ``[{"order": 2}, {"order": 3}]``, cheapest first),
  ``costs`` and ``max_cost`` (optional), ``num_fantasies``. Both BoTorch methods take ``raw_samples`` and ``num_restarts`` (effort of the
  acquisition optimisation).

``results["optimize"]`` keeps the keys of the runner (method, objective, params, value, values, success, message, evaluations, new_evaluations, cache_hits)
and adds ``constraints`` / ``linear_constraints`` as given, for ``pareto`` ``objectives``, ``maximize``, ``ref_point`` and ``front``
(list of ``{"params": job units, "values": all observables, "objectives": the objective values}``).
"""
from __future__ import annotations

import importlib.util
import sys

INF = float("inf")
EXTENDED_KEYS = ("constraints", "linear_constraints")
EXTENDED_METHODS = ("pareto", "multi_fidelity")


def is_extended(spec) -> bool:
    """True if the block ``optimize`` of a job needs this worker."""
    spec = spec or {}
    return bool(spec.get("constraints") or spec.get("linear_constraints") or spec.get("method") in EXTENDED_METHODS)


def _bound(v, default):
    return default if v is None else float(v)


def outcome_constraints(spec, observables):
    from hpfem import opt
    from hpfem.run import JobError

    out = []
    for i, c in enumerate(spec.get("constraints") or []):
        where = f"optimize.constraints[{i}]"
        obs = c.get("observable", 0)
        if isinstance(obs, str) and obs not in observables:
            raise JobError(f"{where}.observable {obs!r}: not one of {observables}")
        if isinstance(obs, int) and not 0 <= obs < len(observables):
            raise JobError(f"{where}.observable {obs}: the study has {len(observables)} observables")
        lo, hi = _bound(c.get("lower"), -INF), _bound(c.get("upper"), INF)
        if lo > hi:
            raise JobError(f"{where}: lower > upper")
        out.append(opt.OutcomeConstraint(obs, lo, hi, str(c.get("name") or f"g{i}")))
    return out


def linear_constraints(spec, scales):
    """hpfem.opt.LinearConstraint in SI from the job-unit constraints (a coefficient on a parameter in job units becomes c / scale on the SI value)."""
    from hpfem import opt
    from hpfem.run import JobError

    out = []
    for i, c in enumerate(spec.get("linear_constraints") or []):
        where = f"optimize.linear_constraints[{i}]"
        coef = c.get("coefficients") or {}
        if not coef:
            raise JobError(f"{where}: 'coefficients' missing")
        unknown = sorted(set(coef) - set(scales))
        if unknown:
            raise JobError(f"{where}: unknown parameters {unknown}")
        lo, hi = _bound(c.get("lower"), -INF), _bound(c.get("upper"), INF)
        if lo > hi:
            raise JobError(f"{where}: lower > upper")
        out.append(opt.LinearConstraint({n: float(v) / scales[n] for n, v in coef.items()}, lo, hi, str(c.get("name") or f"c{i}")))
    return out


def has_botorch() -> bool:
    return importlib.util.find_spec("botorch") is not None and importlib.util.find_spec("torch") is not None


def _need_botorch(what):
    from hpfem.run import JobError

    if not has_botorch():
        raise JobError(f"{what} braucht BoTorch im Python der Rechnung (pip install \"hpfem[opt-bo]\")")


def _values(evaluation):
    from hpfem.run import _to_list

    return None if evaluation is None or evaluation.values is None else _to_list(evaluation.values)


def single_result(method, objective, result, model, spec):
    from hpfem.run import _to_list  # noqa: F401

    out = {"method": method, "objective": objective, "params": model.to_job(result.params), "value": float(result.value),
           "values": _values(result.evaluation), "success": bool(result.success), "message": str(result.message),
           "evaluations": int(result.evaluations), "new_evaluations": int(result.new_evaluations), "cache_hits": int(result.cache_hits)}
    if hasattr(result, "feasible"):
        out["feasible"] = bool(result.feasible)
    for key in ("constraints", "linear_constraints"):
        if spec.get(key):
            out[key] = spec[key]
    return out


def _flags(value, n):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            from hpfem.run import JobError

            raise JobError(f"optimize.maximize: {len(value)} entries for {n} objectives")
        return [bool(v) for v in value]
    return [bool(value)] * n


def _acq_options(spec, fantasies=False):
    """Effort of the BoTorch acquisition optimisation per iteration (``raw_samples``, ``num_restarts``, multi-fidelity: ``num_fantasies``); the library
    defaults (512, 10, 64) take minutes per step on a many-core machine, smaller values make quick tests."""
    out = {}
    for key in ("raw_samples", "num_restarts") + (("num_fantasies",) if fantasies else ()):
        if spec.get(key) is not None:
            out[key] = int(spec[key])
    return out


def run_extended(study, model, spec, original):
    """The optimisation step with the extended keys; ``original`` is hpfem.run._optimize."""
    from hpfem import opt
    from hpfem.run import JobError, _get, _objective

    method = str(_get(spec, "method", "L-BFGS-B"))
    lin = linear_constraints(spec, model.scales)
    if lin:                                            # the constraints of the design space reach every optimiser through study.space
        study.space = opt.DesignSpace(list(study.space.parameters), list(study.space.constraints) + lin)
        study.note("linear constraints of the GUI: " + ", ".join(c.name for c in lin))
    out_cons = outcome_constraints(spec, study.observables)
    if method not in EXTENDED_METHODS and not out_cons:
        result = original(study, model, spec)
        result["linear_constraints"] = spec.get("linear_constraints")
        return result
    seed = int(_get(spec, "seed", 0))
    max_eval = _get(spec, "max_evaluations")
    if method in ("bayesian", "bo"):
        objective = _objective(spec, study.observables)
        res = opt.bayesian_optimize(study, objective, acquisition=str(_get(spec, "acquisition", "ei")), use_gradients=bool(_get(spec, "use_gradients", False)),
                                    constraints=out_cons, maximize=bool(_get(spec, "maximize", False)), max_evaluations=int(max_eval or 30), seed=seed)
        return single_result(method, objective, res, model, spec)
    if out_cons:
        raise JobError("optimize.constraints (Nebenbedingungen an Messgrößen) gehen nur mit method \"bayesian\"")
    if method == "pareto":
        _need_botorch("Mehrzieloptimierung (method \"pareto\")")
        objs = list(_get(spec, "objectives", []))
        if len(objs) < 2:
            raise JobError("optimize.objectives: mindestens zwei Zielgrößen angeben")
        for o in objs:
            _objective({"objective": o}, study.observables)
        maximize = _flags(_get(spec, "maximize", False), len(objs))
        ref = _get(spec, "ref_point")
        if ref is not None and len(ref) != len(objs):
            raise JobError("optimize.ref_point: ein Wert je Zielgröße")
        res = opt.pareto_optimize(study, objs, maximize=maximize, ref_point=None if ref is None else [float(v) for v in ref],
                                  max_evaluations=int(max_eval or 40), n_initial=_get(spec, "n_initial"), seed=seed,
                                  **_acq_options(spec))
        front = [{"params": model.to_job(p), "values": _values(e), "objectives": [float(v) for v in row]}
                 for p, e, row in zip(res.front.params, res.front.evaluations, res.front.values)]
        names = [study.observables[o] if isinstance(o, int) else o for o in objs]
        first = objs[0]
        return {"method": "pareto", "objective": first, "objectives": names, "maximize": maximize, "ref_point": ref, "front": front,
                "params": front[0]["params"] if front else {}, "value": front[0]["objectives"][0] if front else float("nan"),
                "values": front[0]["values"] if front else None, "success": bool(front), "message": str(res.message),
                "evaluations": int(res.evaluations), "new_evaluations": int(res.new_evaluations), "cache_hits": int(study.cache_hits)}
    if method == "multi_fidelity":
        _need_botorch("Mehrstufen-Optimierung (method \"multi_fidelity\")")
        fids = _get(spec, "fidelities")
        if not fids or len(fids) < 2:
            raise JobError("optimize.fidelities: mindestens zwei Stufen, z. B. [{\"order\": 2}, {\"order\": 3}], billigste zuerst")
        objective = _objective(spec, study.observables)
        res = opt.multi_fidelity_optimize(study, objective, fidelities=[dict(f) for f in fids], costs=_get(spec, "costs"),
                                          maximize=bool(_get(spec, "maximize", False)), max_evaluations=int(max_eval or 40),
                                          max_cost=_get(spec, "max_cost"), seed=seed, **_acq_options(spec, fantasies=True))
        out = single_result("multi_fidelity", objective, res, model, spec)
        out["fidelities"] = fids
        return out
    raise JobError(f"optimize.method {method!r}")


def install():
    """Replaces hpfem.run._optimize (hpfem 0.5.x)."""
    import hpfem
    import hpfem.run as run

    if not str(getattr(hpfem, "__version__", "")).startswith("0.5"):
        raise RuntimeError(f"fem_study_worker.py ist für hpfem 0.5.x gebaut (gefunden: {getattr(hpfem, '__version__', '?')}); bitte hpfem.run direkt benutzen")
    if not hasattr(run, "_optimize"):
        raise RuntimeError("hpfem.run._optimize nicht gefunden: diese hpfem-Version passt nicht zum Studien-Worker")
    original = run._optimize

    def patched(study, model, spec):
        if not is_extended(spec):
            return original(study, model, spec)
        return run_extended(study, model, spec, original)

    run._optimize = patched
    return run


def main(argv=None) -> int:
    try:
        run = install()
    except Exception as exc:                           # the GUI reads the error event like the one of the runner
        import json
        sys.stdout.write(json.dumps({"event": "error", "type": type(exc).__name__, "text": str(exc)}) + "\n")
        sys.stdout.flush()
        return 1
    return run.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
