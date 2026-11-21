"""Aufgabe „Hohlraum“ im Rotationskörper-Modus: Modell, Prüfung, analytische Moden, Netz, Rechnung (mit hpfem) und Oberfläche."""
import copy
import json

import numpy as np
import pytest

import fem_axi as fa

CAV = "Zylinderhohlraum R = 1 µm, h = 1,5 µm (leer, PEC; Test gegen die analytischen Moden)"


def cav_model():
    m = copy.deepcopy(fa.presets()[CAV])
    m.pop("task_hint")
    return fa.complete(m)


def errors(m):
    return [t for lvl, t in fa.validate(m, "cavity") if lvl == "error"]


def uni(n_o, n_e, theta=0.0):
    return {"type": "crystal", "kind": "uniaxial", "n_o": n_o, "k_o": 0.0, "n_e": n_e, "k_e": 0.0, "theta_deg": theta, "phi_deg": 0.0}


# ------------------------------------------------------------------------------------------------------------------- model
def test_template_and_defaults():
    assert fa.presets()[CAV]["task_hint"] == "cavity"
    m = cav_model()
    assert not errors(m) and m["cavity"]["orders"] == [0, 1, 2] and fa.cavity_orders(m) == [0, 1, 2]
    old = {k: v for k, v in fa.default_model().items() if k != "cavity"}
    assert fa.complete(old)["cavity"]["radius_nm"] > 0                     # models of older versions get the keys
    assert fa.load_model(json.dumps(old))["cavity"]["num_modes"] >= 1
    assert fa.wavelength_range(m, "cavity") == (1500.0, 1500.0)


def test_validate_cavity():
    m = cav_model()
    m["materials"]["Metall"] = {"type": "index", "n": 0.2, "k": 3.0}
    m["background"] = "Metall"
    assert any("verlustbehaftet" in t for t in errors(m))
    m = cav_model()
    m["parts"] = [dict(type="cylinder", material="Glas", r_inner=0.0, radius=1200.0, z_bottom=100.0, height=200.0)]
    assert any("heraus" in t for t in errors(m))
    m["parts"][0]["radius"] = 500.0
    assert not errors(m) and any("analytische" in t for lvl, t in fa.validate(m, "cavity") if lvl == "warning")
    m = cav_model()
    m["cavity"]["orders"] = []
    assert errors(m)
    m = cav_model()
    m["cavity"]["radius_nm"] = 0.0
    assert errors(m)
    m = cav_model()
    m["materials"]["Luft"] = uni(1.5, 1.7)
    assert not errors(m)                                                    # crystals fill the cavity (axis along z)
    m["materials"]["Luft"] = uni(1.5, 1.7, theta=40.0)
    assert any("nicht diagonal" in t for t in errors(m))                    # a tilted optic axis breaks the rotational symmetry


def test_analytic_modes_of_the_empty_cylinder():
    j01 = 2.404825557695773
    a = fa.cavity_analytic(1000.0, 1500.0, 1.0, [0, 1, 2], count=0)
    tm010 = next(d for d in a if (d["family"], d["m"], d["n"], d["p"]) == ("TM", 0, 1, 0))
    assert tm010["lam_nm"] == pytest.approx(2 * np.pi * 1000.0 / j01, rel=1e-12)
    te111 = next(d for d in a if (d["family"], d["m"], d["n"], d["p"]) == ("TE", 1, 1, 1))
    assert te111["lam_nm"] == pytest.approx(2 * np.pi / np.sqrt((1.8411837813406593 / 1000.0) ** 2 + (np.pi / 1500.0) ** 2), rel=1e-12)
    assert all(d["p"] >= 1 for d in a if d["family"] == "TE")               # TE has no p = 0
    assert [d["lam_nm"] for d in a] == sorted((d["lam_nm"] for d in a), reverse=True)
    n = 1.5
    a2 = fa.cavity_analytic(1000.0, 1500.0, n, [0], count=0)
    assert next(d for d in a2 if d["family"] == "TM")["lam_nm"] == pytest.approx(n * tm010["lam_nm"])


def test_figures():
    import matplotlib.pyplot as plt

    m = cav_model()
    plt.close(fa.cavity_preview_figure(m))
    modes = [dict(m=0, k=0, f_THz=114.7, lam_nm=2613.0), dict(m=1, k=0, f_THz=133.0, lam_nm=2253.0)]
    plt.close(fa.fig_cavity_modes(modes, fa.cavity_analytic(1000.0, 1500.0, 1.0, [0, 1], count=4)))
    plt.close(fa.fig_cavity_modes(modes))
    assert fa.cavity_frame(dict(points=[dict(modes=modes), dict(error="x")])) == modes


# ------------------------------------------------------------------------------------------------------ mesh and worker
def run_cavity(tmp_path, m, order=3, cpw=3.0, fields=True):
    pytest.importorskip("gmsh")
    pytest.importorskip("hpfem")
    import fem_axi_worker as fw

    ms = dict(cells_per_wavelength=cpw, skin_cells=1.0, interface_factor=0.7, curved=False)
    try:
        stats = fa.build_cavity_mesh(m, ms, tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"gmsh unavailable: {exc}")
    (tmp_path / "job.json").write_text(json.dumps(dict(model=m, task="cavity", solver=dict(order=order, backend="AUTO"), subdivisions=2, fields=fields)))
    assert fw.main([str(tmp_path)]) == 0
    return stats, json.loads((tmp_path / "results.json").read_text())


def tri_area(p):
    a, b, c = p
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def test_mesh_tags_and_geometry(tmp_path):
    pytest.importorskip("gmsh")
    m = cav_model()
    try:
        stats = fa.build_cavity_mesh(m, dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=False), tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"gmsh unavailable: {exc}")
    d = np.load(tmp_path / "mesh_plot.npz")
    assert stats["task"] == "cavity" and stats["cells"] > 0
    assert d["xy"][:, 0].min() == pytest.approx(0.0) and d["xy"][:, 0].max() == pytest.approx(1000.0)
    assert d["xy"][:, 1].min() == pytest.approx(0.0) and d["xy"][:, 1].max() == pytest.approx(1500.0)
    area = sum(tri_area(d["xy"][t]) for t in d["tri"])
    assert area == pytest.approx(1000.0 * 1500.0, rel=1e-9)                # the cells fill the meridian rectangle
    m["parts"] = [dict(type="cylinder", material="Glas", r_inner=0.0, radius=400.0, z_bottom=500.0, height=300.0)]
    fa.build_cavity_mesh(m, dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=False), tmp_path / "ins")
    d2 = np.load(tmp_path / "ins" / "mesh_plot.npz")
    names = [str(n) for n in d2["names"]]
    glass = sum(tri_area(d2["xy"][t]) for t, mt in zip(d2["tri"], d2["mat"]) if names[mt] == "Glas")
    assert glass == pytest.approx(400.0 * 300.0, rel=1e-9)


def test_worker_modes_match_the_analytic_cylinder(tmp_path):
    m = cav_model()
    stats, res = run_cavity(tmp_path, m)
    assert res["cavity"]["orders"] == [0, 1, 2] and len(res["points"]) == 3
    an = fa.cavity_analytic(1000.0, 1500.0, np.sqrt(1.000293**2), [0, 1, 2], count=0)
    worst = 0.0
    for pt in res["points"]:
        assert len(pt["modes"]) == 4
        lam = [md["lam_nm"] for md in pt["modes"]]
        assert lam == sorted(lam, reverse=True)                              # k0 ascending = lambda descending
        for md in pt["modes"]:
            assert md["f_THz"] == pytest.approx(299792458.0 / (md["lam_nm"] * 1e-9) / 1e12, rel=1e-9)
            near = min((a for a in an if a["m"] == md["m"]), key=lambda a: abs(a["lam_nm"] - md["lam_nm"]))
            worst = max(worst, abs(md["lam_nm"] / near["lam_nm"] - 1.0))
    assert worst < 2e-3                                                      # coarse mesh, p = 3: measured ~7e-4 (degenerate TE0 / TM1 pairs)
    tm010 = res["points"][0]["modes"][0]
    assert tm010["lam_nm"] == pytest.approx(2 * np.pi * 1000.0 / 2.404825557695773 * 1.000293, rel=1e-5)
    assert (tmp_path / "mode_0_0.npz").is_file() and (tmp_path / "mode_2_3.npz").is_file()
    d = fa.load_field(tmp_path / "mode_0_0.npz")                             # TM010: E_z ~ J0(k r), no E_r, no E_phi
    from scipy.special import j0
    ez = np.abs(d["E"][:, 2])
    assert np.nanmax(np.abs(ez / ez.max() - np.abs(j0(2.404825557695773 * d["points_nm"][:, 0] / 1000.0)))) < 0.02
    assert np.nanmax(np.abs(d["E"][:, 0])) < 0.02 and int(d["m"]) == 0
    assert d["points_nm"][:, 0].max() == pytest.approx(1000.0) and d["points_nm"][:, 1].max() == pytest.approx(1500.0)


def test_worker_dielectric_filling_and_uniaxial_crystal(tmp_path):
    n = 1.5
    m = cav_model()
    m["materials"]["Luft"] = {"type": "index", "n": n, "k": 0.0}
    m["cavity"].update(orders=[0], num_modes=1, lam_ref_nm=2000.0)
    _, res = run_cavity(tmp_path / "diel", m, cpw=3.0, fields=False)
    assert res["points"][0]["modes"][0]["lam_nm"] == pytest.approx(n * 2 * np.pi * 1000.0 / 2.404825557695773, rel=1e-5)   # lambda scales with n
    m["materials"]["Luft"] = uni(1.5, 1.7)                                   # axis along z: TM010 sees eps_z = n_e^2
    _, res = run_cavity(tmp_path / "uni", m, cpw=3.0, fields=False)
    assert res["points"][0]["modes"][0]["lam_nm"] == pytest.approx(1.7 * 2 * np.pi * 1000.0 / 2.404825557695773, rel=1e-5)


# ------------------------------------------------------------------------------------------------------------------- app
def test_app_cavity_preset_without_exception():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(fa.__file__).with_name("fem_app.py")), default_timeout=120)
    at.session_state["app_mode"] = "axi"
    at.run()
    assert not at.exception
    at.selectbox(key="ax_preset_choice").select(CAV).run()
    [b for b in at.button if b.key == "ax_load_preset"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["ax_task"] == "cavity" and at.session_state["ax_model"]["cavity"]["radius_nm"] == 1000.0
    assert "task_hint" not in at.session_state["ax_model"]
    assert not [e for e in at.error if "Fehler" in str(e.value)]


def test_app_shows_cavity_results(tmp_path):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    m = cav_model()
    m["cavity"].update(orders=[0, 1], num_modes=2)
    run_cavity(tmp_path / "run", m)
    at = AppTest.from_file(str(Path(fa.__file__).with_name("fem_app.py")), default_timeout=120)
    at.session_state["app_mode"] = "axi"
    at.run()
    at.session_state["ax_view_dir"] = str(tmp_path / "run")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Hohlraum" in str(c.value) for c in at.caption)
    assert at.dataframe, "Tabelle der Moden fehlt"
