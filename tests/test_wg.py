"""Waveguides and ports (hp-FEM 0.5): model, validation, geometry helpers, mesh (Gmsh, skip-able), tables and figures from synthetic results."""
import copy
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import fem_wg as fw


def test_presets_valid_and_job_is_json():
    for name, m in fw.presets().items():
        assert [t for lv, t in fw.validate(m) if lv == "error"] == [], name
        json.dumps(fw.job(m))
    assert fw.job(fw.default_model())["wavelengths_nm"] == [1550.0]


def test_sweep_list():
    assert fw.sweep_list({"mode": "wavelength", "start": 1500.0, "stop": 1600.0, "n": 3}) == [1500.0, 1550.0, 1600.0]


def test_validation_errors():
    m = fw.default_model()
    m["materials"]["Kern"] = {"type": "crystal", "kind": "uniaxial", "n_o": 1.6, "k_o": 0.0, "n_e": 1.5, "k_e": 0.0, "theta_deg": 0.0, "phi_deg": 0.0}
    assert any("Kristall" in t for _, t in fw.validate(m))
    m = fw.default_model()
    m["ports"]["left"]["enabled"] = m["ports"]["right"]["enabled"] = False
    assert any("Port" in t for lv, t in fw.validate(m) if lv == "error")
    m = fw.default_model()
    m["shapes"][0]["w"] = 3000.0
    assert any("Wand" in t for _, t in fw.validate(m))
    m = fw.default_model()
    m["materials"]["Kern"] = {"type": "index", "n": 2.4, "k": 0.1}
    assert any("verlustbehaftet" in t for _, t in fw.validate(m))
    m["task"] = "modes"
    assert any(lv == "error" and "verlustfrei" in t for lv, t in fw.validate(m))
    m = fw.default_model()
    m["shapes"].append({"type": "ring", "material": "Kern", "cx": 100.0, "cy": 0.0, "r": 100.0, "w": 300.0})
    assert any("0 < w < 2r" in t for _, t in fw.validate(m))
    m = fw.default_model()
    m["shapes"][0]["material"] = "gibtsnicht"
    assert any("fehlt" in t for _, t in fw.validate(m))


def test_shapes_inside_and_priority():
    m = fw.default_model()
    m["shapes"].append({"type": "disc", "material": "Silica", "cx": 1500.0, "cy": 0.0, "r": 100.0})
    assert fw.material_at(m, 500.0, 0.0) == "Kern"
    assert fw.material_at(m, 500.0, 500.0) == "Silica"
    assert fw.material_at(m, 1500.0, 0.0) == "Silica"                      # the later disc covers the strip
    ring = {"type": "ring", "material": "Kern", "cx": 0.0, "cy": 0.0, "r": 1000.0, "w": 200.0}
    assert fw.inside(ring, 1000.0, 0.0) and not fw.inside(ring, 0.0, 0.0) and not fw.inside(ring, 1200.0, 0.0)
    poly = {"type": "poly", "material": "Kern", "points": [[0, 0], [100, 0], [0, 100]]}
    assert fw.inside(poly, 20.0, 20.0) and not fw.inside(poly, 80.0, 80.0)
    assert fw.shape_bounds(poly) == (0.0, 100.0, 0.0, 100.0)


def test_mesh_tags_and_surface_material(tmp_path):
    pytest.importorskip("gmsh")
    m = fw.default_model()
    m["shapes"].append({"type": "ring", "material": "Silica", "cx": 1500.0, "cy": 0.0, "r": 150.0, "w": 60.0})
    st = fw.build_mesh(m, 1550.0, tmp_path)
    assert st["cells"] > 100
    text = (tmp_path / "mesh.msh").read_text()
    assert "$PhysicalNames" in text
    for name in ("left", "right", "bottom", "top", "Silica", "Kern"):
        assert f'"{name}"' in text


def test_core_in_the_middle_is_not_swallowed(tmp_path):
    """A core fully inside the box leaves a cladding surface with a hole: its centre of mass lies in the core (regression)."""
    gmsh = pytest.importorskip("gmsh")
    m = fw.presets()["Querschnitt: Streifen 450 × 220 nm"]
    fw.build_mesh(m, 1550.0, tmp_path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "mesh.msh"))
        names = {gmsh.model.getPhysicalName(2, t): t for _, t in gmsh.model.getPhysicalGroups(2)}
        assert set(names) == {"Silica", "Kern"}
        areas = {}
        for n, t in names.items():
            tot = 0.0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(2, t):
                tot += sum(gmsh.model.mesh.getElementQualities(gmsh.model.mesh.getElements(2, ent)[1][0], "volume"))
            areas[n] = tot
        assert areas["Kern"] == pytest.approx(450.0 * 220.0, rel=1e-6)
    finally:
        gmsh.finalize()


def _ports_point(i, lam):
    ch = [dict(port=0, side="left", mode=0, beta=1.0), dict(port=1, side="right", mode=0, beta=1.0)]
    s = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)
    return dict(index=i, lam_nm=lam, channels=ch, S_re=s.real.tolist(), S_im=s.imag.tolist(), excitation=ch[0], R=0.01, T=0.81, loss=0.18,
                port_modes={"left": [dict(n_eff=2.0, propagating=True)], "right": [dict(n_eff=2.0, propagating=True)]})


def test_tables_and_figures():
    pts = [_ports_point(0, 1500.0), _ports_point(1, 1550.0), dict(index=2, lam_nm=1600.0, error="x")]
    df = fw.ports_dataframe(pts)
    assert list(df["λ (nm)"]) == [1500.0, 1550.0] and df["T"].iloc[0] == pytest.approx(0.81)
    sm = fw.s_matrix_frame(pts[0])
    assert sm.shape == (2, 2) and sm.iloc[1, 0] == pytest.approx(0.81)
    assert fw.s_matrix_frame(pts[0], "phase").shape == (2, 2)
    for fig in (fw.fig_spectrum(pts), fw.fig_neff(pts), fw.fig_preview(fw.default_model()), fw.fig_preview(fw.presets()["Querschnitt: Streifen 450 × 220 nm"])):
        assert fig is not None


def test_field_figures():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    simp = np.array([[0, 1, 2], [1, 3, 2]])
    npz = dict(points=pts, simplices=simp, values=(np.ones((4, 2)) * (1 + 1j)))
    for q in ("abs", "re_x", "re_y"):
        assert fw.fig_field(npz, q, "t", copy.deepcopy(fw.default_model())) is not None
    mode = dict(points=pts, simplices=simp, et_0=np.ones((4, 2), complex), ez_0=np.ones(4, complex))
    for q in ("abs_t", "abs_z", "ex", "ey"):
        assert fw.fig_mode(mode, 0, q, "t") is not None


# ------------------------------------------------------------------------------------------------------------ with hpfem (skipped without it)
def _run(model, tmp_path):
    pytest.importorskip("gmsh")
    pytest.importorskip("hpfem")
    import fem_wg_worker as ww

    fw.build_mesh(model, min(fw.sweep_list(model["sweep"])), tmp_path)
    (tmp_path / "job.json").write_text(json.dumps(fw.job(model)), encoding="utf-8")
    assert ww.main([str(tmp_path)]) == 0
    return json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))


def test_straight_waveguide_transmits_and_matches_the_slab_dispersion(tmp_path):
    import scipy.optimize

    m = fw.default_model()
    m["field"] = "first"
    res = _run(m, tmp_path)
    p = res["points"][0]
    assert p["T"] == pytest.approx(1.0, abs=1e-3) and p["R"] < 1e-3 and abs(p["loss"]) < 1e-3
    k0, d, n_core, n_clad = 2 * np.pi / 1550.0, 450.0, 2.4, 1.44

    def f(n):
        kappa, gamma = k0 * np.sqrt(n_core**2 - n**2), k0 * np.sqrt(n**2 - n_clad**2)
        return kappa * np.tan(kappa * d / 2) - (n_core / n_clad) ** 2 * gamma

    n_lo = np.sqrt(n_core**2 - (np.pi / (k0 * d)) ** 2)                      # the fundamental mode lies on the first branch of the tangent
    n_eff = scipy.optimize.brentq(f, n_lo + 1e-6, n_core - 1e-9)
    assert p["port_modes"]["left"][0]["n_eff"] == pytest.approx(n_eff, rel=2e-4)
    assert p["channels"][0]["side"] == "left" and len(p["S_re"]) == len(p["channels"])
    s = np.array(p["S_re"]) + 1j * np.array(p["S_im"])
    assert np.allclose(s, s.T, atol=1e-6)                                   # reciprocity
    assert np.allclose(s.conj().T @ s, np.eye(len(s)), atol=1e-2)           # lossless: unitary
    npz = np.load(tmp_path / p["field"])
    assert npz["values"].shape[1] == 2 and npz["points"].shape[0] == npz["values"].shape[0]


def test_step_has_a_small_reflection_and_conserves_power(tmp_path):
    m = fw.presets()["Stufe: Streifen 450 → 900 nm"]
    m["field"] = "none"
    p = _run(m, tmp_path)["points"][0]
    assert 1e-4 < p["R"] < 0.05 and abs(p["loss"]) < 1e-3


def test_cross_section_modes_are_sorted_and_bounded(tmp_path):
    m = copy.deepcopy(fw.presets()["Querschnitt: Streifen 450 × 220 nm"])
    m["mesh"]["cells_per_wavelength"] = 5.0
    m["modes"]["num"] = 2
    p = _run(m, tmp_path)["points"][0]
    n = p["n_eff"]
    assert 2.2 < n[0] < 2.5 and 1.44 < n[1] < n[0]                            # 450 x 220 nm silicon wire in silica, TE0 about 2.35
    npz = np.load(tmp_path / p["modes_file"])
    assert npz["et_0"].shape[1] == 2 and npz["ez_0"].shape[0] == npz["points"].shape[0]
