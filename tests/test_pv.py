"""Photovoltaik-Auswertung (fem_pv): reine Tests gegen analytische Fälle, ein kleiner Worker-Lauf, die App."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

import fem_pv as pv


def test_blackbody_scaling_and_shape():
    lam = np.linspace(280.0, 4000.0, 3001)
    I = pv.blackbody_irradiance(lam, total=1000.0)
    assert np.sum(0.5 * (I[1:] + I[:-1]) * np.diff(lam)) == pytest.approx(1000.0, rel=1e-4)
    assert lam[np.argmax(I)] == pytest.approx(502.0, abs=15.0)                 # Wien: 2.898e-3 / 5778 K = 501.5 nm (per wavelength)
    assert pv.blackbody_irradiance(500.0, total=500.0) == pytest.approx(0.5 * pv.blackbody_irradiance(500.0, total=1000.0))


def test_band_widths_and_weights_match_hpfem_rule():
    lam = np.array([400.0, 500.0, 700.0, 800.0])
    w = pv.band_widths(lam)
    assert w == pytest.approx([100.0, 150.0, 150.0, 100.0])                    # midpoints, end bands doubled half-widths
    assert pv.band_widths([600.0]) == pytest.approx([1.0])
    spec = (np.array([300.0, 900.0]), np.array([2.0, 2.0]))
    assert pv.band_weights(lam, spec) == pytest.approx(2.0 * w)
    assert pv.band_weights([1000.0], spec) == pytest.approx([0.0])             # outside: 0
    assert pv.coverage([500.0, 1000.0], spec) == pytest.approx(0.5)


def test_jsc_flat_spectrum_and_limit():
    lam = np.linspace(400.0, 800.0, 41)
    spec = (np.array([300.0, 900.0]), np.array([1.0, 1.0]))                    # 1 W/(m² nm)
    # A = 1: J = q ∫ I λ/(hc) dλ = q I / (2 h c) (λ2² − λ1²); the end bands have the full spacing (hpfem rule), so the bands span 395..805 nm
    exact = pv.Q_E * 1.0 / (pv.H_PLANCK * pv.C0) * 0.5 * (805e-9**2 - 395e-9**2) * 1e9 * 0.1
    assert pv.jsc_limit(lam, spec) == pytest.approx(exact, rel=2e-3)
    assert pv.jsc(lam, np.zeros(len(lam)), spec) == 0.0
    half = np.where(lam >= 600.0, 1.0, 0.0)                                    # absorbs only above 600 nm
    assert 0.0 < pv.jsc(lam, half, spec) < pv.jsc_limit(lam, spec)
    # one photon per electron: J = q Φ for a single band of 1 W/m² at 500 nm
    one = pv.jsc([500.0], [1.0], (np.array([400.0, 600.0]), np.array([1.0, 1.0])))
    assert one == pytest.approx(pv.Q_E * 1.0 * 500e-9 / (pv.H_PLANCK * pv.C0) * 0.1, rel=1e-12)


def test_spectral_profile_and_integral():
    lam = np.array([500.0, 600.0, 700.0])
    y = np.linspace(0.0, 1000.0, 11)
    g = np.array([np.full(11, 2.0), np.full(11, 3.0), np.full(11, 5.0)])      # 1/(m³ s) per W/m²
    spec = (np.array([400.0, 800.0]), np.array([0.5, 0.5]))
    w = pv.band_weights(lam, spec)
    prof = pv.spectral_profile(lam, g, spec)
    assert prof == pytest.approx(np.full(11, float(w @ np.array([2.0, 3.0, 5.0]))))
    assert pv.integrated_generation(y, np.ones(11)) == pytest.approx(11 * 100e-9)


def test_parse_spectrum_csv():
    lam, irr = pv.parse_spectrum_csv("# AM\nlambda;irr\n400;1,5\n500;2,0\n600;1,0\n")
    assert lam == pytest.approx([400, 500, 600]) and irr == pytest.approx([1.5, 2.0, 1.0])
    lam, irr = pv.parse_spectrum_csv("400, 1.0\n500, 2.0")
    assert irr == pytest.approx([1.0, 2.0])
    for bad in ("400 1\n", "500 1\n400 1\n", "400 1\n500 -1\n"):
        with pytest.raises(ValueError):
            pv.parse_spectrum_csv(bad)


def test_collect_tables_figures():
    pts = [dict(lam_nm=600.0, pv=dict(absorber="Si", A_abs=0.5, y_nm=[0.0, 1.0], g_per_intensity=[1.0, 2.0])),
           dict(lam_nm=500.0, pv=dict(absorber="Si", A_abs=0.7, y_nm=[0.0, 1.0], g_per_intensity=[3.0, 4.0])),
           dict(lam_nm=550.0, error="x"), dict(lam_nm=700.0)]
    lam, a, y, g = pv.collect(pts)
    assert lam.tolist() == [500.0, 600.0] and a.tolist() == [0.7, 0.5] and g.tolist() == [[3.0, 4.0], [1.0, 2.0]]
    assert pv.collect([dict(lam_nm=1.0)]) is None
    spec = (np.array([400.0, 800.0]), np.array([1.0, 1.0]))
    tab = pv.spectrum_table(lam, a, spec)
    assert list(tab["λ (nm)"]) == [500.0, 600.0] and tab["J Beitrag (mA/cm²)"].sum() == pytest.approx(pv.jsc(lam, a, spec))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fig in (pv.fig_absorption(lam, a, spec, "Si"), pv.fig_absorption(lam, a), pv.fig_profile(y, g[0])):
        plt.close(fig)
    assert list(pv.profile_table(y, g[0]).columns) == ["y (nm)", "G (1/(m³ s))"]


def _hpfem_ok():
    try:
        import gmsh  # noqa: F401
        import hpfem.pv  # noqa: F401
        return hasattr(__import__("hpfem"), "absorbed_power_by_tag")
    except Exception:
        return False


@pytest.mark.skipif(not _hpfem_ok(), reason="braucht hpfem 0.5 mit pv und Gmsh")
def test_worker_small_run_generation_profile(tmp_path):
    """Two wavelengths on a coarse mesh (a few seconds): the PV post-processing gives a profile whose integral reproduces the exact absorption."""
    import fem_geometry as fg
    import fem_run as fr
    from pathlib import Path

    name = next(k for k in fg.presets() if k.startswith("Si-Lamellengitter"))
    m = copy.deepcopy(fg.presets()[name])
    m["sweep"] = dict(m["sweep"], mode="wavelength", start=500.0, stop=650.0, n=2)
    ms = dict(cells_per_wavelength=2.0, skin_cells=1.0, interface_factor=0.7, curved=True, pml_cells_per_wavelength=4.0, lam_min_nm=500.0, lam_max_nm=650.0)
    fg.build_mesh(m, ms, tmp_path)
    solver = dict(order=2, backend="AUTO", pml_target=1e-3, pml_angle_cap_deg=80.0, orders_max=2, order_points=64, engine="grating", jacobian=False,
                  scalar="auto", check=False, jones=False, pv=dict(absorber="Si", bins=24))
    fr.write_job(tmp_path, fr.make_job(m, solver, dict(enabled=False, indices=[], res_nm=5.0, triangulate=False, subdiv=3, hs=False)))
    worker = Path(fr.__file__).parent / "fem_worker.py"
    p = subprocess.run([sys.executable, str(worker), str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert "DONE" in p.stdout, p.stdout[-800:] + p.stderr[-800:]
    pts = json.loads((tmp_path / "results.json").read_text())["points"]
    assert len(pts) == 2 and all("pv" in q for q in pts), [q.get("error") for q in pts]
    for q in pts:
        d = q["pv"]
        assert d["absorber"] == "Si" and len(d["y_nm"]) == 24 == len(d["g_per_intensity"])
        assert d["A_abs"] == pytest.approx(q["absorbed"]["by_material"]["Si"])
        assert min(d["g_per_intensity"]) >= 0.0 and max(d["g_per_intensity"]) > 0.0
        assert d["check_ratio"] == pytest.approx(1.0, abs=0.08)
    lam, a, y, g = pv.collect(pts)
    assert pv.jsc(lam, a, (np.array([400.0, 800.0]), np.array([1.0, 1.0]))) > 0.0
