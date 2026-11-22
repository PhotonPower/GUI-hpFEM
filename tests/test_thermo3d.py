"""Heat and mechanics (3D): models, validation, closed-form checks of the worker (block), sphere chain optics -> heat -> stress, estimates, figures, the app."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_thermo3d as t3  # noqa: E402

PY = "/home/itofrenner/hpfemenv/bin/python"
pytestmark = pytest.mark.skipif(not Path(PY).exists(), reason="hpfem environment not available")


def run_worker(model, folder, *extra, max_gb=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(t3.job(t3.normalise(copy.deepcopy(model)), max_gb)), encoding="utf-8")
    r = subprocess.run([PY, str(HERE / "fem_thermo3d_worker.py"), str(folder), *extra], capture_output=True, text=True, timeout=240)
    res = json.loads((folder / "results.json").read_text(encoding="utf-8")) if (folder / "results.json").is_file() else None
    return r, res


def preset(sub):
    return copy.deepcopy(next(v for k, v in t3.presets().items() if sub in k))


# --------------------------------------------------------------------------------------------------------------------- model
def test_presets_validate_and_are_json():
    for name, m in t3.presets().items():
        assert not [t for lv, t in t3.validate(t3.normalise(m)) if lv == "error"], name
        json.dumps(m)
    assert {m["case"] for m in t3.presets().values()} == {"particle", "block"}


def test_validation_errors():
    m = preset("Goldkugel in Glas: Erwärmung")
    m["particle"]["outer_factor"] = 3
    assert any("ganzzahlig" in t for lv, t in t3.validate(m) if lv == "error")
    m = preset("Goldkugel in Glas: Erwärmung")
    m["materials"]["Glas"] = {"type": "index", "n": 1.5, "k": 0.1}
    assert any("verlustfrei" in t for lv, t in t3.validate(m) if lv == "error")
    m = preset("Quader unter Zug")
    m["mech"]["bc"]["xmin"]["kind"] = m["mech"]["bc"]["ymin"]["kind"] = m["mech"]["bc"]["zmin"]["kind"] = "free"
    assert any("Gleichgewicht" in t for lv, t in t3.validate(m) if lv == "error")
    m = preset("Wärmequelle")
    m["thermal"]["sinks"] = {s: False for s in t3.SIDE_TAGS}
    assert any("Wärmesenke" in t for lv, t in t3.validate(m) if lv == "error")
    m = preset("selbstkonsistent")
    assert not any("ohne dε/dT" in t for lv, t in t3.validate(m))             # the preset carries a coefficient (used_materials covers the sphere)
    m["props"]["Gold"].update(deps_re=0.0, deps_im=0.0)
    assert any("ohne dε/dT" in t for lv, t in t3.validate(m) if lv == "warn")
    m = preset("Quader unter Zug")
    m["props"]["Stahl-ähnlich"]["nu"] = 0.6
    assert any("ν" in t for lv, t in t3.validate(m) if lv == "error")


def test_closed_forms_and_limits():
    m = preset("Quader unter Zug")
    (_, ux, _), (_, uy, _), (_, uz, _) = t3.analytic_checks(m)
    E, nu, s = 200e9, 0.3, 100e6
    assert ux == pytest.approx(s * 4e-6 / E) and uy == pytest.approx(-nu * s * 2e-6 / E) and uz == pytest.approx(-nu * s * 1e-6 / E)
    m["mech"]["dT_b"] = 80.0                                   # a gradient has no textbook value here
    assert t3.analytic_checks(m) == []
    assert t3.analytic_checks(preset("Goldkugel in Glas: Erwärmung")) == []
    h = preset("Wärmequelle")
    (_, tm, key), = t3.heat_checks(h)
    assert key == "T_mid" and tm == pytest.approx(300.0 + 1e15 * (4e-6) ** 2 / (8 * 50.0))
    h["thermal"]["sinks"]["xmax"] = False
    (_, te, key), = t3.heat_checks(h)
    assert key == "T_end" and te == pytest.approx(300.0 + 1e15 * (4e-6) ** 2 / (2 * 50.0))
    h["thermal"]["sinks"]["ymax"] = True
    assert t3.heat_checks(h) == []


def test_estimates_scale():
    a, b = t3.mech_estimate(1000), t3.mech_estimate(4000)
    assert b["dofs"] == 4 * a["dofs"] and b["seconds"] > 4 * a["seconds"] and b["bytes"] > a["bytes"]
    m = preset("Goldkugel in Glas: Erwärmung")
    assert t3.particle_heat_estimate(m, 33e-6) == pytest.approx(33e-6 / (4 * np.pi * 1.4) * (1 / 50e-9 - 1 / (2 * 0.6606785 * 250e-9)))
    assert np.isnan(t3.particle_heat_estimate(m, 0.0))
    g = t3.available_gb()
    assert g is None or g > 0


def test_mirror_signs_and_shapes():
    sys.path.insert(0, str(HERE))
    import fem_thermo3d_worker as w

    xs, ys, zs = np.array([0.5, 1.5]), np.array([0.5, 1.5, 2.5]), np.array([0.5])
    f = {k: np.arange(6, dtype=float).reshape(2, 3, 1) + 1.0 for k in ("T", "ux", "uy", "sxy", "uz")}
    axes, out = w.mirror([xs, ys, zs], f, dict(quarter=True))
    assert [len(a) for a in axes] == [4, 6, 1] and np.allclose(axes[0], [-1.5, -0.5, 0.5, 1.5])
    assert out["T"].shape == (4, 6, 1)
    assert np.allclose(out["T"], out["T"][::-1, :, :]) and np.allclose(out["T"], out["T"][:, ::-1, :])                   # even
    assert np.allclose(out["ux"], -out["ux"][::-1, :, :]) and np.allclose(out["ux"], out["ux"][:, ::-1, :])             # odd in x only
    assert np.allclose(out["uy"], out["uy"][::-1, :, :]) and np.allclose(out["uy"], -out["uy"][:, ::-1, :])             # odd in y only
    assert np.allclose(out["sxy"], -out["sxy"][::-1, :, :]) and np.allclose(out["sxy"], -out["sxy"][:, ::-1, :])        # odd in both
    same = w.mirror([xs, ys, zs], f, dict(quarter=False))
    assert same[1] is f


# --------------------------------------------------------------------------------------------------------------------- block worker
@pytest.mark.parametrize("variant", ["tension", "heating", "shrink"])
def test_block_matches_closed_form(tmp_path, variant):
    m = preset("Quader unter Zug")
    if variant == "heating":
        m["mech"].update(dT_a=50.0, dT_b=50.0)
        m["mech"]["bc"]["xmax"] = {"kind": "free", "value": 0.0}
    if variant == "shrink":
        m["mech"].update(shrink_a_pct=1.0, shrink_b_pct=1.0)
    checks = t3.analytic_checks(m)
    assert checks
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-500:]
    for _, exp, key in checks:
        assert res["probes"][key] == pytest.approx(exp, rel=1e-8, abs=1e-18)
    assert res["scalars"]["Größte Von-Mises-Spannung auf dem Raster (MPa)"] == pytest.approx(0.0 if variant == "heating" else 100.0, abs=1e-6)   # the shrink case keeps the 100 MPa pull


def test_block_free_heating_gives_no_stress_and_clamped_does(tmp_path):
    m = preset("Quader eingespannt")
    r, res = run_worker(m, tmp_path / "clamped")
    assert r.returncode == 0 and res["scalars"]["Größte Von-Mises-Spannung auf dem Raster (MPa)"] > 50.0
    m["mech"]["bc"] = {s: {"kind": "free", "value": 0.0} for s in t3.SIDE_TAGS}
    r, res = run_worker(m, tmp_path / "free")
    assert r.returncode == 0 and res["scalars"]["Größte Von-Mises-Spannung auf dem Raster (MPa)"] < 1e-3                 # free body: remove_rigid_modes


def test_block_heat_source_matches_the_slab_solution_and_drives_stress(tmp_path):
    m = preset("Wärmequelle")
    (_, exp, _), = t3.heat_checks(m)
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-500:]
    assert res["probes"]["T_mid"] == pytest.approx(exp, rel=1e-8)
    assert res["probes"]["T_end"] == pytest.approx(300.0, abs=1e-6)                          # sink face
    assert res["scalars"]["Wärmeleistung im Quader (W)"] == pytest.approx(1e15 * 4e-6 * 2e-6 * 1e-6, rel=1e-9)
    assert res["scalars"]["Größte Von-Mises-Spannung auf dem Raster (MPa)"] > 1.0
    f = np.load(tmp_path / "fields3d.npz")
    assert f["T"].shape == (len(f["x"]), len(f["y"]), len(f["z"])) and np.nanmax(f["T"]) == pytest.approx(exp, rel=0.05)
    for key in t3.FIELDS:                                                                   # every map and the other figures render
        if key in f.files:
            for axis in "xyz":
                t3.fig_slice(f, key, axis, 0.4).clear()
    t3.fig_line(f, "T", "x", 0.3, 0.6).clear()
    for axis in "xyz":
        t3.fig_deformed(f, axis, 0.5, 1e3).clear()
    t3.fig_history([3.0, 0.3, 0.03]).clear()
    t3.fig_preview(m).clear()
    import matplotlib.pyplot as plt

    plt.close("all")


def test_worker_estimate_limit_and_errors(tmp_path):
    m = preset("Quader unter Zug")
    r, res = run_worker(m, tmp_path / "est", "estimate")
    est = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("ESTIMATE"))[len("ESTIMATE"):])
    assert est["cells"] == 96 and est["h1_dofs"] == 225 and est["mech_dofs"] == 675 and est["bytes"] > 0
    r, res = run_worker(m, tmp_path / "lim", max_gb=1e-6)                                     # the memory limit stops the run before solving
    assert r.returncode == 1 and "exceeds the limit" in r.stdout and res["error"]
    for s in t3.SIDE_TAGS:                                                                   # unbalanced load on a free body (validate would refuse it)
        m["mech"]["bc"][s] = {"kind": "free", "value": 0.0}
    m["mech"]["bc"]["xmax"] = {"kind": "traction", "value": 10.0}
    r, res = run_worker(m, tmp_path / "err")
    assert r.returncode == 1 and res.get("error") and "ERROR" in r.stdout


def test_particle_estimate_reports_optics_and_mechanics(tmp_path):
    r, _ = run_worker(preset("Goldkugel in Glas: Erwärmung und Wärmespannung"), tmp_path, "estimate")
    est = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("ESTIMATE"))[len("ESTIMATE"):])
    assert est["nd_dofs"] > est["h1_dofs"] > 0 and est["optics_bytes"] > 1e6 and "SparseLU" in est["optics_text"] and est["bytes"] >= est["optics_bytes"]


# --------------------------------------------------------------------------------------------------------------------- sphere chain
def test_sphere_chain_heat_matches_the_estimate_and_fields_are_symmetric(tmp_path):
    m = preset("Goldkugel in Glas: Erwärmung und Wärmespannung")
    m["grid"] = 18
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-800:]
    sc = res["scalars"]
    p, dt = sc["Absorbierte Leistung (W)"], sc["Temperaturanstieg im Kugelmittelpunkt (K)"]
    assert p > 1e-6 and dt > 5
    assert dt == pytest.approx(t3.particle_heat_estimate(m, p), rel=0.10)                    # observed: -3.7 % on the coarse mesh
    assert sc["  davon in der Kugel (W)"] == pytest.approx(p, rel=1e-6)                       # glass is lossless
    assert sc["Größte Von-Mises-Spannung auf dem Raster (MPa)"] > 1.0
    f = np.load(tmp_path / "fields3d.npz")
    assert float(f["sphere_r"]) == 50.0 and str(f["unit"]) == "nm"
    assert np.allclose(f["x"], -f["x"][::-1]) and np.allclose(f["y"], -f["y"][::-1])           # the quarter is mirrored to the full box
    for key in ("T", "Eabs", "umag", "svm"):
        a = f[key]
        assert np.allclose(a, a[::-1, :, :], equal_nan=True) and np.allclose(a, a[:, ::-1, :], equal_nan=True)
    assert np.allclose(f["ux"], -f["ux"][::-1, :, :], equal_nan=True)
    assert np.nanmax(f["T"]) >= 300.0 + dt * 0.9 and np.nanmin(f["T"]) >= 300.0 - 0.5
    for axis in "xyz":
        t3.fig_slice(f, "T", axis, 0.5).clear()
    t3.fig_slice(f, "Eabs", "y", 0.5).clear()


def test_sphere_selfconsistent_loop_runs_and_is_reported(tmp_path):
    m = preset("selbstkonsistent")
    m["order"] = 1                                                                            # quick: ~1 800 optical unknowns
    m["thermal"].update(max_iter=3, tolerance=1e-2)
    m["props"]["Gold"].update(deps_re=-0.05, deps_im=0.0)
    r, res = run_worker(m, tmp_path)
    assert r.returncode == 0 and res["finished"], r.stdout[-800:]
    sc = res["scalars"]
    assert sc["Iterationen"] >= 2 and len(res["history"]) == sc["Iterationen"] and res["history"][-1] < res["history"][0]
    assert "Dehnungsenergie (J)" not in sc and "Schleife konvergiert" in sc


# --------------------------------------------------------------------------------------------------------------------- app
def _app(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=120)
    at.session_state["app_mode"] = "thermo"
    at.session_state["th_dim"] = "3D"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    return at


def test_app_all_presets_without_exception(tmp_path):
    at = _app(tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    for name in t3.presets():
        at.selectbox(key="t3_preset").select(name).run()
        [b for b in at.button if b.key == "t3_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])
        assert at.session_state["th3_model"]["name"] == name


def test_app_two_d_still_default_and_dimension_switch(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=120)
    at.session_state["app_mode"] = "thermo"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception and at.radio(key="th_dim").value == "2D" and "th_model" in at.session_state
    at.radio(key="th_dim").set_value("3D").run()
    assert not at.exception and "th3_model" in at.session_state


def test_app_shows_results(tmp_path):
    m = preset("Quader unter Zug")
    run_worker(m, tmp_path / "thermo3d")
    at = _app(tmp_path)
    at.session_state["th3_run_model"] = m
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("geschlossene" in str(x.value) for x in at.markdown)
