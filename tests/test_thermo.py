"""Heat and mechanics (2D): models, validation, closed-form checks of the mechanics worker, wire chain optics -> heat -> stress, the app."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_thermo as ft  # noqa: E402

PY = "/home/itofrenner/hpfemenv/bin/python"
pytestmark = pytest.mark.skipif(not Path(PY).exists(), reason="hpfem environment not available")


def run_worker(model, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(ft.normalise(copy.deepcopy(model))), encoding="utf-8")
    r = subprocess.run([PY, str(HERE / "fem_thermo_worker.py"), str(folder)], capture_output=True, text=True, timeout=240)
    res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    return r, res


def preset(sub):
    return copy.deepcopy(next(v for k, v in ft.presets().items() if sub in k))


# --------------------------------------------------------------------------------------------------------------------- model
def test_presets_validate_and_are_json():
    for name, m in ft.presets().items():
        assert not [t for lv, t in ft.validate(ft.normalise(m)) if lv == "error"], name
        json.dumps(m)
    assert {m["case"] for m in ft.presets().values()} == {"wire", "block", "axi"}


def test_validation_errors():
    m = preset("Goldnanodraht in Glas: Erwärmung")
    m["wire"]["outer_factor"] = 3
    assert any("ganzzahlig" in t for lv, t in ft.validate(m) if lv == "error")
    m = preset("Goldnanodraht in Glas: Erwärmung")
    m["materials"]["Glas"] = {"type": "index", "n": 1.5, "k": 0.1}
    assert any("verlustfrei" in t for lv, t in ft.validate(m) if lv == "error")
    m = preset("Block unter Zug")
    m["mech"]["bc"]["left"]["kind"] = m["mech"]["bc"]["bottom"]["kind"] = "free"
    assert any("Gleichgewicht" in t for lv, t in ft.validate(m) if lv == "error")
    m = preset("Block unter Zug")
    m["props"]["Stahl-ähnlich"]["nu"] = 0.6
    assert any("ν" in t for lv, t in ft.validate(m) if lv == "error")
    m = preset("selbstkonsistent")
    m["props"]["Gold"].update(deps_re=0.0, deps_im=0.0)
    assert any("ohne dε/dT" in t for lv, t in ft.validate(m) if lv == "warn")
    m = preset("Goldnanodraht in Glas: Erwärmung")
    m["materials"]["Glas"] = {"type": "library", "name": "water"}
    m["mech"]["on"] = True
    assert any("Fluid" in t for lv, t in ft.validate(m) if lv == "warn")


def test_analytic_reference_values_and_limits():
    m = preset("Block unter Zug")
    (lab1, ux, _), (lab2, uy, _) = ft.analytic_checks(m)
    E, nu = 200e9, 0.3
    assert ux == pytest.approx((1 - nu**2) * 100e6 * 4e-6 / E) and uy == pytest.approx(-nu * (1 + nu) * 100e6 * 2e-6 / E)
    m["mech"]["dT_b"] = 80.0                                  # a gradient has no textbook value here
    assert ft.analytic_checks(m) == []
    assert ft.analytic_checks(preset("Goldnanodraht in Glas: Erwärmung")) == []


def test_line_source_estimate_scales():
    m = preset("Goldnanodraht in Glas: Erwärmung")
    a = ft.heat_line_estimate(m, 100.0)
    assert ft.heat_line_estimate(m, 200.0) == pytest.approx(2 * a)
    m["props"]["Glas"]["kappa"] *= 2
    assert ft.heat_line_estimate(m, 100.0) == pytest.approx(a / 2)


def test_figures_render(tmp_path):
    for m in ft.presets().values():
        fig = ft.fig_preview(m)
        fig.clear()


# --------------------------------------------------------------------------------------------------------------------- mechanics worker vs closed forms
@pytest.mark.parametrize("sub,mod", [
    ("Block unter Zug", {}),
    ("Block unter Zug", {"mech.dT_a": 40.0, "mech.dT_b": 40.0, "mech.shrink_pct": 0.05}),          # traction + uniform temperature + shrinkage
    ("Zylinder unter Axialdruck", {}),
    ("Zylinder unter Axialdruck", {"mech.dT_a": 30.0, "mech.dT_b": 30.0}),
])
def test_block_and_cylinder_match_closed_form(tmp_path, sub, mod):
    m = preset(sub)
    for k, v in mod.items():
        d, last = m, k.split(".")
        for kk in last[:-1]:
            d = d[kk]
        d[last[-1]] = v
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-400:]
    checks = ft.analytic_checks(m)
    assert checks
    for lab, exp, key in checks:
        assert res["probes"][key] == pytest.approx(exp, rel=1e-7), lab
    f = np.load(tmp_path / "fields.npz")
    assert np.isfinite(f["svm"]).sum() > 100 and float(f["umag"].max()) > 0
    assert f["mesh_disp"].shape == f["mesh_vertices"].shape


def test_clamped_heating_gives_stress_and_free_gives_none(tmp_path):
    m = preset("Block eingespannt")
    r, res = run_worker(m, tmp_path / "a")
    assert r.returncode == 0
    assert res["scalars"]["Größte Von-Mises-Spannung (MPa)"] > 50.0                    # E alpha dT ~ 120 MPa scale
    for s in ft.SIDE_TAGS:                                                              # free expansion: no stress
        m["mech"]["bc"][s] = {"kind": "free", "value": 0.0}
    r, res = run_worker(m, tmp_path / "b")
    assert r.returncode == 0
    assert res["scalars"]["Größte Von-Mises-Spannung (MPa)"] < 1e-3                    # remove_rigid_modes: free body


def test_worker_reports_solver_error(tmp_path):
    m = preset("Block unter Zug")
    for s in ft.SIDE_TAGS:                                                              # unbalanced load on a free body (validate would refuse it)
        m["mech"]["bc"][s] = {"kind": "free", "value": 0.0}
    m["mech"]["bc"]["right"] = {"kind": "traction", "value": 10.0}
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 1 and res.get("error") and "ERROR" in r.stdout


# --------------------------------------------------------------------------------------------------------------------- wire chain
def small_wire(**kw):
    m = preset("Goldnanodraht in Glas: Erwärmung und Wärmespannung")
    m["wire"].update(half_factor=3, outer_factor=6, cells_per_radius=1)
    m["order"] = 2
    m["grid"] = 40
    m.update(kw)
    return m


def test_wire_heat_matches_line_source_and_stress_is_finite(tmp_path):
    m = small_wire()
    m["wire"].update(half_factor=4, outer_factor=8, cells_per_radius=2)
    m["order"] = 3
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-600:]
    sc = res["scalars"]
    p, dt = sc["Absorbierte Leistung (W/m)"], sc["Temperaturanstieg max. (K)"]
    assert p > 10 and dt > 5
    assert dt == pytest.approx(ft.heat_line_estimate(m, p), rel=0.05)                      # line source in a square heat sink
    assert sc["  davon im Draht (W/m)"] == pytest.approx(p, rel=1e-6)                       # glass is lossless
    f = np.load(tmp_path / "fields.npz")
    assert np.nanmin(f["T"]) >= 300.0 - 1.0 and np.isfinite(f["svm"]).any() and f["Eabs"].shape == f["T"].shape
    assert sc["Größte Von-Mises-Spannung (MPa)"] > 1.0
    for key in ft.FIELDS:                                                                   # every map and the other figures render
        if key in f.files:
            ft.fig_field(f, key).clear()
    ft.fig_deformed(f, 1e3).clear()
    ft.fig_line(f, "T", "y", 0.3).clear()
    ft.fig_history([3.0, 0.3, 0.03]).clear()


def test_wire_selfconsistent_loop_converges_and_changes_the_result(tmp_path):
    m = small_wire()
    m["mech"]["on"] = False
    r0, res0 = run_worker(m, tmp_path / "plain")
    ms = copy.deepcopy(m)
    ms["thermal"]["selfconsistent"] = True
    ms["props"]["Gold"].update(deps_re=-0.05, deps_im=0.0)
    r1, res1 = run_worker(ms, tmp_path / "sc")
    assert r0.returncode == r1.returncode == 0, r1.stdout[-500:]
    assert res1["converged"] and res1["scalars"]["Iterationen"] > 1 and len(res1["history"]) == res1["scalars"]["Iterationen"]
    assert res1["history"][-1] < 1e-3 < res1["history"][0]
    assert res1["scalars"]["Absorbierte Leistung (W/m)"] != pytest.approx(res0["scalars"]["Absorbierte Leistung (W/m)"], rel=1e-3)
    assert "Dehnungsenergie (J/m)" not in res1["scalars"]
    mz = copy.deepcopy(ms)
    mz["props"]["Gold"].update(deps_re=0.0, deps_im=0.0)                                   # no coefficient: one pass equals the plain run
    r2, res2 = run_worker(mz, tmp_path / "zero")
    assert res2["scalars"]["Iterationen"] == 1
    assert res2["scalars"]["Absorbierte Leistung (W/m)"] == pytest.approx(res0["scalars"]["Absorbierte Leistung (W/m)"], rel=5e-3)   # observed: 8e-4
    assert res2["scalars"]["Maximale Temperatur (K)"] == pytest.approx(res0["scalars"]["Maximale Temperatur (K)"], rel=5e-3)


# --------------------------------------------------------------------------------------------------------------------- app
def _app(tmp_path):
    from streamlit.testing.v1 import AppTest

    import fem_app as fa  # noqa: F401  (path only)

    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=120)
    at.session_state["app_mode"] = "thermo"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    return at


def test_app_all_presets_without_exception(tmp_path):
    at = _app(tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    for name in ft.presets():
        at.selectbox(key="th_preset").select(name).run()
        [b for b in at.button if b.key == "th_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])
        assert at.session_state["th_model"]["name"] == name


def test_app_shows_results(tmp_path):
    m = preset("Zylinder unter Axialdruck")
    run_worker(m, tmp_path / "thermo")
    at = _app(tmp_path)
    at.session_state["th_run_model"] = m
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("geschlossene" in str(x.value) for x in at.markdown)
