"""Rotationskörper (fem_axi): Modell, Vorlagen, Prüfungen, Layout, Netz, Abbildungen; ohne hpfem."""
import copy
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fem_axi as fa  # noqa: E402


def test_presets_valid_and_serialisable():
    for name, m in fa.presets().items():
        json.dumps(m)
        for task in ("resonance", "emitter"):
            errs = [t for lvl, t in fa.validate(m, task) if lvl == "error"]
            assert not errs, (name, task, errs)
        lay = fa.layout(m)
        assert lay["r_out"] > lay["r_in"] > lay["r_struct"] and lay["z_top"] > lay["z_hi"] > lay["z_lo"] > lay["z_bot"]
        assert fa.load_model(json.dumps(m))["kind"] == "axi"


def test_dbr_pillar_layers_share_interfaces():
    parts, centre, height = fa.dbr_pillar(3, 4, 500.0, 940.0, n_hi=3.53, n_lo=2.95)
    assert len(parts) == 2 * 3 + 2 * 4 + 1
    for a, b in zip(parts, parts[1:]):
        assert a["z_bottom"] + a["height"] == pytest.approx(b["z_bottom"], abs=1e-9)
    cav = parts[2 * 4]
    assert cav["z_bottom"] < centre < cav["z_bottom"] + cav["height"]
    assert height == pytest.approx(parts[-1]["z_bottom"] + parts[-1]["height"])


def test_material_at_and_source_box_check():
    m = fa.presets()[next(k for k in fa.presets() if k.startswith("Mikrosäule") and "schnell" in k)]
    z = m["emitter"]["z_nm"]
    assert fa.material_at(m, 0.0, z) == "GaAs"
    assert fa.material_at(m, 5000.0, z) == "Luft"
    assert fa.material_at(m, 5000.0, -10.0) == "GaAs"                    # substrate
    bad = copy.deepcopy(m)
    bad["emitter"]["sigma_nm"] = 60.0                                     # box 240 nm reaches into the mirrors
    assert any("Kasten" in t for lvl, t in fa.validate(bad, "emitter") if lvl == "error")


def test_validation_errors():
    m = fa.default_model()
    m["parts"].append(dict(type="torus", material="Glas", r_center=100.0, z_center=0.0, radius=200.0))
    m["resonance"]["m"] = -1
    errs = [t for lvl, t in fa.validate(m) if lvl == "error"]
    assert any("Ringradius" in t for t in errs) and any("m muss" in t for t in errs)
    m = fa.default_model()
    m["materials"]["Au"] = {"type": "library", "name": "air"}
    m["background"] = "Glas"
    m["materials"]["Glas"] = {"type": "index", "n": 1.5, "k": 0.1}
    assert any("verlustfrei" in t for lvl, t in fa.validate(m) if lvl == "error")


def test_part_polygons():
    for kind in fa.PART_TYPES:
        q = fa.part_polygon(fa.new_part(kind, "Glas"))
        assert q.shape[1] == 2 and len(q) >= 3 and q[:, 0].min() >= -1e-9


def test_mesh_has_axis_wall_and_source_box(tmp_path):
    pytest.importorskip("gmsh")
    m = fa.presets()[next(k for k in fa.presets() if k.startswith("Dielektrische Kugel"))]
    ms = dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=False, pml_cells_per_wavelength=4.0, lam_min_nm=900, lam_max_nm=1100)
    try:
        stats = fa.build_mesh(m, ms, tmp_path, "emitter")
    except Exception as exc:                                                # Gmsh without display libraries on some CI images
        pytest.skip(f"gmsh unavailable: {exc}")
    text = (tmp_path / "mesh.msh").read_text()
    names = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0]
    assert f'1 {fa.TAG_AXIS} "axis"' in names and f'1 {fa.TAG_WALL} "wall"' in names
    assert "Emitterkasten" in names and stats["source_box_nm"] > 0 and stats["cells"] > 50


def test_figures():
    import matplotlib.pyplot as plt
    for m in fa.presets().values():
        plt.close(fa.preview_figure(m))
    modes = [dict(k=0, lam_nm=930.0, Q=170.0), dict(k=1, lam_nm=945.0, Q=-3.0)]
    plt.close(fa.fig_modes(modes, 940.0))
    pts = [dict(lam_nm=920.0 + i, purcell=1.0 + i, purcell_radiative=0.5, beta_top=0.2, beta_bottom=0.3) for i in range(3)]
    plt.close(fa.fig_purcell(pts, dict(lam_nm=921.0, Q=170.0)))
    pts_ = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = dict(points_nm=pts_, simplices=np.array([[0, 1, 2], [1, 3, 2]]), E=np.ones((4, 3), complex), m=1, lam_nm=930.0)
    for q in fa.AXI_QUANTITIES:
        plt.close(fa.fig_axi_field(d, fa.default_model(), q))


def test_scattering_settings_and_validation():
    m = fa.presets()[next(k for k in fa.presets() if k.startswith("Gold-Nanokugel in Wasser"))]
    assert not [t for lvl, t in fa.validate(m, "scattering") if lvl == "error"]
    assert fa.wavelength_range(m, "scattering") == (450.0, 700.0)
    lay = fa.layout(m)
    assert lay["r_struct"] < lay["r_plane"] < lay["r_in"]
    pillar = fa.presets()[next(k for k in fa.presets() if "schnell" in k)]
    assert any("Substrat" in t for lvl, t in fa.validate(pillar, "scattering") if lvl == "error")
    old = fa.default_model()
    del old["scattering"]
    del old["emitter"]["modal"]
    new = fa.complete(old)
    assert new["scattering"]["sweep"]["start"] < new["resonance"]["wavelength_nm"] < new["scattering"]["sweep"]["stop"]
    assert new["emitter"]["modal"] is False


def test_mie_reference():
    pytest.importorskip("scipy")
    import fem_axi_worker as fw
    small = fw.mie(500.0, 5.0, 1.5, 1.0)                                 # Rayleigh limit: sigma_sca = 8 pi / 3 k^4 a^6 ((m^2-1)/(m^2+2))^2
    k, a = 2 * np.pi / 500e-9, 5e-9
    rayleigh = 8 * np.pi / 3 * k ** 4 * a ** 6 * ((1.5 ** 2 - 1) / (1.5 ** 2 + 2)) ** 2
    assert small["sigma_sca"] == pytest.approx(rayleigh, rel=1e-3)
    assert abs(small["sigma_abs"]) < 1e-6 * small["sigma_sca"]
    lossy = fw.mie(530.0, 40.0, complex(0.5, 2.2), 1.33)
    assert lossy["sigma_abs"] > 0 and lossy["sigma_ext"] > lossy["sigma_sca"]


def test_scattering_and_modal_figures():
    import matplotlib.pyplot as plt
    pts = [dict(lam_nm=500.0 + 10 * i, sigma_sca=1e-14 * (i + 1), sigma_abs=2e-14, sigma_ext=1e-14 * (i + 3),
                mie=dict(sigma_sca=1e-14 * (i + 1), sigma_abs=2e-14, sigma_ext=1e-14 * (i + 3))) for i in range(4)]
    plt.close(fa.fig_cross_sections(pts, np.pi * 40.0 ** 2, 0.0, "S"))
    th = np.linspace(0, np.pi, 19).tolist()
    pt = dict(lam_nm=530.0, theta_deg=30.0, pattern_theta=th, **{f"pattern_{k}": np.ones(19).tolist() for k in ("xz_0", "xz_pi", "yz_0", "yz_pi")})
    plt.close(fa.fig_pattern(pt))
    pts_ = np.array([[0.0, -50.0], [50.0, -50.0], [0.0, 50.0], [50.0, 50.0]])
    d = dict(points_nm=pts_, simplices=np.array([[0, 1, 2], [1, 3, 2]]), E_right=np.zeros((4, 3), complex), E_left=np.zeros((4, 3), complex),
             lam_nm=530.0, theta_deg=30.0, pol="P", k_bg=2 * np.pi / 530e-9)
    xz, E = fa.scatter_field(d, "tot")
    assert np.allclose(np.abs(E[:, 0]), np.cos(np.radians(30.0))) and np.allclose(np.abs(E[:, 2]), np.sin(np.radians(30.0)))
    for q in fa.SCATTER_QUANTITIES:
        plt.close(fa.fig_scatter_field(d, fa.default_model(), q))
    modal = [dict(lam_nm=930.0 + i, purcell=1.5, modal=dict(total=1.5, mode=1.0, background=0.5)) for i in range(3)]
    plt.close(fa.fig_modal(modal, dict(Q=170.0)))


def test_radially_infinite_layers():
    P = fa.presets()
    mem = P[next(k for k in P if k.startswith("GaAs-Membran"))]
    assert fa.material_at(mem, 0.0, 100.0) == "GaAs" and fa.material_at(mem, 1e5, 100.0) == "GaAs"   # through the PML
    assert fa.material_at(mem, 0.0, 300.0) == "Luft"
    assert fa.base_material(mem, 100.0) == "GaAs"
    lay = fa.layout(mem)
    assert lay["z_plane_bottom"] < 0.0 and lay["z_plane_top"] > 200.0
    assert not [t for lvl, t in fa.validate(mem, "emitter") if lvl == "error"]
    assert any("Schichten" in t for lvl, t in fa.validate(mem, "scattering") if lvl == "error")
    pil = P[next(k for k in P if k.startswith("Mikrosäule auf planarem"))]
    assert len(pil["layers"]) == 20 and fa.material_at(pil, 1e5, pil["layers"][0]["height"] / 2) == "GaAs"
    assert fa.material_at(pil, 0.0, pil["emitter"]["z_nm"]) == "GaAs"
    old = fa.default_model()
    del old["layers"]
    assert fa.complete(old)["layers"] == []


def test_layer_mesh(tmp_path):
    pytest.importorskip("gmsh")
    mem = fa.presets()[next(k for k in fa.presets() if k.startswith("GaAs-Membran"))]
    ms = dict(cells_per_wavelength=2.0, skin_cells=1.0, interface_factor=0.7, curved=False, pml_cells_per_wavelength=3.0, lam_min_nm=900, lam_max_nm=1000)
    try:
        stats = fa.build_mesh(mem, ms, tmp_path, "emitter")
    except Exception as exc:
        pytest.skip(f"gmsh unavailable: {exc}")
    d = np.load(tmp_path / "mesh_plot.npz")
    xy, tri, mat = d["xy"], d["tri"], d["mat"]
    gaas = list(d["names"]).index("GaAs")
    lay = fa.layout(mem)
    assert xy[tri[mat == gaas]].reshape(-1, 2)[:, 0].max() == pytest.approx(lay["r_out"])   # the layer reaches the outer wall
    assert stats["source_box_nm"] > 0


# ------------------------------------------------------------------------------------------------ far field of emitters (without hpfem)
def _larmor_x(theta, p_total=2.0):
    """Intensity [W/sr] of an x-directed dipole of power p_total at the azimuths phi_k of fem_axi.FAR_PHIS: 3 P / (8 pi) (1 - sin^2 theta cos^2 phi)."""
    phis = [2 * np.pi * k / fa.FAR_PHIS for k in range(fa.FAR_PHIS)]
    return np.array([3 * p_total / (8 * np.pi) * (1 - np.sin(theta) ** 2 * np.cos(phi) ** 2) for phi in phis])


def test_far_pattern_metrics_isotropic_and_larmor():
    th = np.linspace(0.0, np.pi, 721)
    iso = np.full_like(th, 1.0 / (4 * np.pi))
    r = fa.far_pattern_metrics(th, iso, 1.0, 1.0)
    assert r["axis_gain"] == pytest.approx(1.0)
    assert r["cone_fraction"][-1] == pytest.approx(1.0, abs=1e-5)
    assert r["fraction_within_deg"]["30"] == pytest.approx((1 - np.cos(np.radians(30))) / 2, abs=1e-4)
    assert r["half_angle_fwhm_deg"] is None
    inten = _larmor_x(th)
    r = fa.far_pattern_metrics(th, inten.mean(axis=0), 2.0, 2.0)
    assert r["axis_gain"] == pytest.approx(1.5, abs=1e-6)
    assert r["cone_fraction"][-1] == pytest.approx(1.0, abs=1e-5)


def test_emitter_far_final_planes_and_balance():
    import fem_axi_worker as fw

    th = np.linspace(0.0, np.pi, 361)
    raw = dict(theta=th, inten=_larmor_x(th), p_far=2.0, orders=[1, -1])
    out = fw.emitter_far_final(raw, 2.0, 2.0, False)
    assert out["balance"] == pytest.approx(1.0)
    assert np.allclose(out["gain_xz_0"], 1.5 * np.cos(th) ** 2, atol=1e-9)          # E plane (phi = 0): dipole axis in the plane
    assert np.allclose(out["gain_yz_0"], 1.5, atol=1e-9)                              # H plane: uniform
    assert np.allclose(out["gain_xz_pi"], out["gain_xz_0"]) and np.allclose(out["gain_yz_pi"], out["gain_yz_0"])
    assert out["axis_gain"] == pytest.approx(1.5, abs=1e-6)
    json.dumps(out)
    pt = dict(lam_nm=450.0, far=out)
    import matplotlib.pyplot as plt
    for log in (False, True):
        fig = fa.fig_far_pattern(pt, log=log)
        plt.close(fig)


def test_box_in_background_lens_and_substrate():
    base = fa.default_model()
    base["parts"] = [dict(type="sphere", material="Glas", z_center=0.0, radius=500.0)]
    base["emitter"]["z_nm"] = 700.0
    ok, why = fa.box_in_background(base)
    assert ok, why
    sub = copy.deepcopy(base)
    sub["substrate"] = "Glas"
    assert not fa.box_in_background(sub)[0]
    lay = base.copy()
    lay["layers"] = [dict(material="Glas", z_bottom=-100.0, height=50.0)]
    assert not fa.box_in_background(lay)[0]
    # the box (r_max + margin/2, z_struct +- margin/2) encloses every part by construction: only substrate and layers can cross it


# ------------------------------------------------------------------------------------------------ far field on a layer stack (without hpfem)
def _stack_model(layers, substrate=None):
    m = fa.default_model()
    m["materials"].update({"GaAs": {"type": "index", "n": 3.53, "k": 0.0}, "AlAs": {"type": "index", "n": 2.95, "k": 0.0}})
    m["parts"], m["layers"], m["substrate"] = [], layers, substrate
    return m


def test_stack_layers_membrane_substrate_and_merge():
    assert fa.stack_layers(fa.default_model()) is None
    mem = fa.stack_layers(_stack_model([dict(material="GaAs", z_bottom=0.0, height=200.0)]))      # membrane in air
    assert mem["top_nm"] == 200.0 and mem["layers"] == [("GaAs", 200.0)] and mem["substrate"] == "Luft"
    sub = fa.stack_layers(_stack_model([], "Glas"))                                                 # bare substrate: no layers
    assert sub["top_nm"] == 0.0 and sub["layers"] == [] and sub["substrate"] == "Glas"
    two = fa.stack_layers(_stack_model([dict(material="GaAs", z_bottom=0.0, height=100.0), dict(material="AlAs", z_bottom=100.0, height=50.0),
                                        dict(material="AlAs", z_bottom=150.0, height=25.0)], "Glas"))
    assert two["top_nm"] == 175.0 and two["layers"] == [("AlAs", 75.0), ("GaAs", 100.0)] and two["substrate"] == "Glas"   # same neighbours merged
    gap = fa.stack_layers(_stack_model([dict(material="GaAs", z_bottom=100.0, height=50.0)], "Glas"))                  # air gap above the substrate
    assert gap["layers"] == [("GaAs", 50.0), ("Luft", 100.0)] and gap["top_nm"] == 150.0
    total = sum(t for _, t in gap["layers"])
    assert gap["top_nm"] - total == pytest.approx(0.0)                                                                  # lowest interface at z = 0
    for name, m in fa.presets().items():
        if m.get("layers") or m.get("substrate"):
            st = fa.stack_layers(m)
            assert st["substrate"] in m["materials"] and all(t > 0 for _, t in st["layers"]), name


def test_far_field_plan_modes():
    base = fa.default_model()
    base["parts"] = [dict(type="sphere", material="Glas", z_center=0.0, radius=500.0)]
    base["emitter"]["z_nm"] = 700.0
    assert fa.far_field_plan(base)[0] == "homogeneous"
    sub = copy.deepcopy(base)
    sub["substrate"] = "Glas"
    mode, text = fa.far_field_plan(sub)
    assert mode == "layered" and "Substrat" in text


def test_emitter_far_final_layered_fractions_and_down_cone():
    import fem_axi_worker as fw

    up = np.linspace(0.0, np.pi / 2 - 1e-6, 181)
    dn = np.linspace(np.pi / 2 + 1e-6, np.pi, 181)
    th = np.concatenate([up, dn])
    inten = np.array([np.where(th < np.pi / 2, 2.0, 1.0)] * fa.FAR_PHIS) / (4 * np.pi)       # 2/(4 pi) up, 1/(4 pi) down: powers 1 and 0.5
    raw = dict(theta=th, inten=inten, p_far=1.5, orders=[1, -1], layered=True, p_up=1.0, p_down=0.5, down_available=True)
    out = fw.emitter_far_final(raw, 2.0, 2.0, False)
    assert out["layered"] and out["frac_up"] == pytest.approx(0.5) and out["frac_down"] == pytest.approx(0.25)
    assert out["frac_guided_absorbed"] == pytest.approx(0.25)
    assert out["axis_gain"] == pytest.approx(2.0 / (4 * np.pi) / (2.0 / (4 * np.pi)), rel=1e-6)                 # I(0) / (P_bulk / 4 pi) = 1
    assert out["axis_gain_down"] == pytest.approx(0.5 * out["axis_gain"], rel=1e-6)
    assert len(out["cone_fraction_down"]) == len(th)
    json.dumps(out)
    import matplotlib.pyplot as plt
    fig = fa.fig_far_pattern(dict(lam_nm=450.0, far=out))
    plt.close(fig)
    # lossy substrate: only the upper half
    raw2 = dict(theta=up, inten=inten[:, :181], p_far=1.0, orders=[1, -1], layered=True, p_up=1.0, p_down=0.0, down_available=False)
    out2 = fw.emitter_far_final(raw2, 2.0, 2.0, False)
    assert "cone_fraction_down" not in out2 and out2["down_available"] is False
    plt.close(fa.fig_far_pattern(dict(lam_nm=450.0, far=out2)))
