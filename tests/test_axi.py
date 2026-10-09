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
