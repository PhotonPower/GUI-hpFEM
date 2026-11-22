"""Time domain (TimeDomain2D/3D): model, estimates, signals, spectra, validation, short worker runs (pulse transit time, energy), app."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_time as ft  # noqa: E402

STRIP = "Pulsausbreitung im Streifen"
SPHERE = "3D: dielektrische Kugel (n = 2) im PEC-Kasten"


def errors(m):
    return [t for lv, t in ft.validate(m) if lv == "error"]


def run_worker(m, folder):
    """Mesh (2D), job, worker in this process with two threads (the machine is shared); returns the results."""
    hpfem = pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    import fem_time_worker as tw

    hpfem.set_num_threads(2)
    folder.mkdir(parents=True, exist_ok=True)
    ft.build_mesh(m, folder)
    (folder / "job.json").write_text(json.dumps(ft.job(m)), encoding="utf-8")
    assert tw.main([str(folder)]) == 0
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def strip_model(**time_kw):
    """Vacuum strip 4 um x 0.3 um, PEC top and bottom, absorbing ends, E_y pulse from x = 1 um; probes at 1.6 and 2.8 um (coarse, p = 2)."""
    m = copy.deepcopy(ft.presets()[STRIP])
    m["mesh"].update(order=2, cells_per_wavelength=3.0)
    m["width_nm"] = 300.0                                                                     # only the TEM mode propagates: the E_y modes n >= 2 have cutoffs above 1000 THz
    m["source"].update(x_nm=1000.0, width_fs=3.0)
    m["probes"] = [[1600.0, 0.0], [2800.0, 0.0]]
    m["time"].update(steps_per_period=30, frames=10, **time_kw)
    return m


# ------------------------------------------------------------------------------------------------------------------- pure
def test_presets_validate_and_job_is_json():
    for name, m in ft.presets().items():
        assert not errors(m), (name, errors(m))
        d = json.loads(json.dumps(ft.job(m)))
        assert d["steps"] == ft.estimate(m)["steps"] > 100 and d["dt"] > 0 and d["dim"] == m["dim"]
        assert (d["probes"] and len(d["probes"][0]) == (3 if m["dim"] == "3d" else 2))
    assert len(ft.presets()) >= 5


def test_estimate_numbers():
    m = ft.default_model()
    e = ft.estimate(m)
    assert e["f0_THz"] == pytest.approx(599.585, rel=1e-5)                                     # c / 500 nm
    assert e["f_max_THz"] == pytest.approx(e["f0_THz"] + 3 / (2 * math.pi * 3.0) * 1e3, rel=1e-9)
    assert e["dt_fs"] == pytest.approx(1e3 / e["f_max_THz"] / 40, rel=1e-9)                    # 40 steps per period of f_max
    assert e["t0_fs"] == 12.0 and e["steps"] == math.ceil(e["t_end_fs"] / e["dt_fs"])
    m["time"].update(dt_fs=0.05, t_end_fs=10.0, f_max_THz=700.0)
    e = ft.estimate(m)
    assert e["dt_fs"] == 0.05 and e["steps"] == 200 and e["lam_min_nm"] == pytest.approx(ft.C0 / 700e12 / 1e-9)
    g = copy.deepcopy(ft.default_model())
    g["source"]["kind"] = "gaussian"
    assert ft.estimate(g)["f_max_THz"] == pytest.approx(4 / (2 * math.pi * 3.0) * 1e3)


def test_signal_derivatives_match_finite_differences():
    omega, t0, tau = 2 * math.pi * 6e14, 12e-15, 3e-15
    t = np.array([5e-15, 10e-15, 12.4e-15, 17e-15, 30e-15])
    h = 1e-19
    for kind in ft.SIGNALS:
        num = (ft.signal_value(kind, omega, t0, tau, t + h) - ft.signal_value(kind, omega, t0, tau, t - h)) / (2 * h)
        assert np.allclose(ft.signal_derivative(kind, omega, t0, tau, t), num, rtol=2e-4, atol=1e-3 * np.max(np.abs(num))), kind


def test_signal_formulas_equal_the_library_pulses():
    hpfem = pytest.importorskip("hpfem")
    omega, t0, tau = 2 * math.pi * 6e14, 12e-15, 3e-15
    for kind, lib in (("modulated", hpfem.modulated_gaussian(omega, t0, tau)), ("gaussian", hpfem.gaussian_pulse(t0, tau))):
        for t in (4e-15, 11e-15, 13.3e-15, 20e-15):
            assert lib.derivative(t) == pytest.approx(float(ft.signal_derivative(kind, omega, t0, tau, t)), rel=1e-9, abs=1e-6)


def test_spectrum_peak_of_a_modulated_pulse_is_the_carrier():
    m = ft.default_model()
    est = ft.estimate(m)
    omega, tau = 2 * math.pi * est["f0_THz"] * 1e12, 3e-15
    t = np.arange(0, 60e-15, est["dt_fs"] * 1e-15)
    g = ft.signal_derivative("modulated", omega, est["t0_fs"] * 1e-15, tau, t)
    f, a = ft.spectrum(t / 1e-15, g)
    assert ft.peak_frequency(f, a, 100, 1500) == pytest.approx(est["f0_THz"], rel=0.01)
    f, a = ft.spectrum(t / 1e-15, np.sin(2 * math.pi * 250e12 * t), "hann")                  # a pure tone, windowed
    assert ft.peak_frequency(f, a) == pytest.approx(250.0, abs=1e3 / 60 / 2)
    assert ft.arrival_time(t / 1e-15, np.exp(-(t - 25e-15) ** 2 / (2 * (3e-15) ** 2))) == pytest.approx(25.0, abs=1e-3)


def test_validation():
    m = ft.default_model()
    m["materials"]["Glas"] = {"type": "index", "n": 1.5, "k": 0.1}
    m["background"] = "Glas"
    assert any("verlustbehaftet" in t for t in errors(m))
    m["loss_as_sigma"] = True
    assert not errors(m)
    e, sg = ft.eps_sigma(m, "Glas")
    assert e == pytest.approx(1.5**2 - 0.1**2) and sg == pytest.approx(ft.EPS0 * 2 * math.pi * ft.C0 / 500e-9 * 2 * 1.5 * 0.1)
    m = ft.default_model()
    m["materials"]["Au"] = {"type": "library", "name": "Au"}
    m["background"] = "Au"
    assert any("Re ε" in t for t in errors(m))
    m = ft.default_model()
    m["source"]["x_nm"] = 1e6
    m["probes"] = [[10.0, 5000.0]]
    assert sum("außerhalb" in t for t in errors(m)) == 2
    m = ft.default_model()
    m["time"]["t_end_fs"] = 1e5
    assert any("Zeitschritte" in t for t in errors(m))
    m = ft.default_model()
    m["time"].update(gamma=0.6, beta=0.25)
    assert any("β ≥" in t for t in errors(m))
    m["time"].update(gamma=0.6, beta=0.3025)
    assert not errors(m) and any("γ > 1/2" in t for lv, t in ft.validate(m) if lv == "warn")
    m = ft.default_model()
    m["time"].update(steps_per_period=5)
    assert any("Phasenfehler" in t for lv, t in ft.validate(m) if lv == "warn")
    m = ft.default_model()
    m["materials"]["Kristall"] = {"type": "crystal", "kind": "uniaxial", "n_o": 1.6, "k_o": 0.0, "n_e": 1.5, "k_e": 0.0, "theta_deg": 0.0, "phi_deg": 0.0}
    m["shapes"] = [{"type": "disc", "material": "Kristall", "cx": 1000.0, "cy": 0.0, "r": 200.0}]
    assert any("Kristalle" in t for t in errors(m))


def test_job_boundaries_and_materials():
    m = ft.presets()[STRIP]
    d = ft.job(m)
    assert d["pec_tags"] == [ft.TAG_BOTTOM, ft.TAG_TOP] and d["absorbing_tags"] == [ft.TAG_LEFT, ft.TAG_RIGHT]
    lossy = ft.presets()["Leitfähiger Block (σ = 1·10⁵ S/m): Dämpfung des Pulses"]
    dl = ft.job(lossy)
    assert dl["materials"]["Leiter"]["sigma"] == 1e5 and dl["materials"]["Leiter"]["eps"] == 2.0 and dl["material_tags"]["Leiter"] == 3
    d3 = ft.job(ft.presets()[SPHERE])
    assert d3["pec_tags"] == [1, 2, 3, 4, 5, 6] and d3["sphere"]["outer"] == pytest.approx(600e-9) and d3["source"]["pol"] == [0.0, 0.0, 1.0]


def test_figures_without_results_build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (STRIP, SPHERE):
        fig = ft.fig_preview(ft.presets()[name])
        plt.close(fig)
        fig = ft.fig_source(ft.presets()[name])
        plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------- worker runs
def test_pulse_transit_time_and_absorbing_boundary(tmp_path):
    m = strip_model()
    res = run_worker(m, tmp_path / "strip")
    t, e = np.asarray(res["t_fs"]), np.asarray(res["energy"])
    assert len(t) == res["meta"]["steps"] + 1 and res["meta"]["dim"] == "2d"
    p1, p2 = res["probes"][0], res["probes"][1]
    t1, t2 = ft.arrival_time(t, p1["ey"]), ft.arrival_time(t, p2["ey"])
    expected = 1200e-9 / ft.C0 / 1e-15                                                         # TEM mode of the strip: speed c
    assert t2 - t1 == pytest.approx(expected, rel=0.03), (t1, t2, expected)
    assert max(np.max(np.abs(p1["ex"])), np.max(np.abs(p2["ex"]))) < 0.08 * np.max(np.abs(p1["ey"]))     # E_y dominates (E_x only from the near field of the finite source)
    te = np.asarray(res["t_energy_fs"])
    assert e.max() > 0 and e[-1] < 0.03 * e.max() and te[-1] == pytest.approx(t[-1])           # the pulse has left through the absorbing ends
    f, a = ft.spectrum(t, p2["ey"])
    assert ft.peak_frequency(f, a, 300, 1000) == pytest.approx(res["meta"]["estimate"]["f0_THz"], rel=0.06)     # carrier
    fr = np.load(tmp_path / "strip" / "field_frames.npz")
    assert fr["ey"].shape[1] == len(fr["points"]) and fr["ey"].shape[0] == len(fr["times_fs"]) >= 10 and np.max(np.abs(fr["ey"])) > 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = dict(fr)
    for q in ("abs", "ey"):
        plt.close(ft.fig_frame(frames, 3, q, res["meta"], None, m))
    plt.close(ft.fig_probes(res))
    plt.close(ft.fig_energy(res, log=True))
    plt.close(ft.fig_spectrum(res, [0, 1], "ey", "hann", "transfer", "lam"))
    assert len(ft.probe_dataframe(res)) == len(t) and len(ft.energy_dataframe(res)) == len(te)


def test_closed_cavity_conserves_energy_and_conductivity_damps(tmp_path):
    m = strip_model(t_end_fs=45.0)
    m["boundary"] = {k: "pec" for k in ft.SIDES_2D}
    res = run_worker(m, tmp_path / "pec")
    te, e = np.asarray(res["t_energy_fs"]), np.asarray(res["energy"])
    late = te > 12.0 + 6 * 3.0 + 1.0                                                           # the source has ended (t0 + 6 tau)
    assert late.sum() > 20 and (e[late].max() - e[late].min()) / e[late].max() < 1e-4
    m["sigma"] = {"Vakuum": 2.0e3}                                                             # a weakly conducting background damps the energy
    res2 = run_worker(m, tmp_path / "sigma")
    e2, te2 = np.asarray(res2["energy"]), np.asarray(res2["t_energy_fs"])
    late2 = te2 > 19.0
    assert np.all(np.diff(e2[late2]) <= 1e-9 * e2.max()) and e2[-1] < 0.9 * e2[late2][0]


def test_3d_sphere_cavity_runs_and_conserves_energy(tmp_path):
    m = copy.deepcopy(ft.presets()[SPHERE])
    m["mesh"]["order"] = 1
    m["time"].update(t_end_fs=16.0, frames=4)
    m["grid"] = 21
    m["probes"] = [[250.0, 0.0, 0.0], [0.0, 250.0, 100.0]]
    res = run_worker(m, tmp_path / "ball")
    te, e = np.asarray(res["t_energy_fs"]), np.asarray(res["energy"])
    assert res["meta"]["dim"] == "3d" and res["meta"]["dofs"] < 1000 and "ez" in res["probes"][0]
    late = te > 4.0 + 6 * 1.0 + 0.5
    assert late.sum() > 10 and (e[late].max() - e[late].min()) / e[late].max() < 1e-4        # PEC box: lossless
    fr = np.load(tmp_path / "ball" / "field_frames.npz")
    assert fr["ez"].shape[1:] == (21, 21) and np.nanmax(np.abs(fr["ez"])) > 0 and np.isnan(fr["ez"]).sum() >= 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close(ft.fig_frame(dict(fr), 2, "abs", res["meta"], None, None))


def test_3d_dof_limit_is_reported(tmp_path):
    hpfem = pytest.importorskip("hpfem")
    import fem_time_worker as tw

    hpfem.set_num_threads(2)
    m = copy.deepcopy(ft.presets()[SPHERE])
    d = ft.job(m)
    d["max_dofs"] = 100
    folder = tmp_path / "limit"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps(d), encoding="utf-8")
    assert tw.main([str(folder)]) == 1


# ------------------------------------------------------------------------------------------------------------------- app
def test_app_time_mode_and_results(tmp_path):
    from streamlit.testing.v1 import AppTest

    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    run_worker(strip_model(), tmp_path / "time")
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=180)
    at.session_state["app_mode"] = "time"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Zeitbereich" in str(t.value) for t in at.title)
    assert at.dataframe, "Tabelle der Materialien oder der Ergebnisse fehlt"
    at.selectbox(key="tm_preset").select(SPHERE).run()
    [b for b in at.button if b.key == "tm_preset_load"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["tm_model"]["dim"] == "3d"
