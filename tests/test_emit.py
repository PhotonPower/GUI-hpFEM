"""Emitters (hp-FEM 0.5): emitter of a periodic cell (validation, mesh refinement), figures and tables from synthetic
results, off-axis emitters of bodies of revolution (radii, boxes, validation, figures). Pure Python, no hpfem."""
import copy
import json
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import fem_axi as fa
import fem_emit as fe
import fem_geometry as fg
import fem_materials as fm


# ---------------------------------------------------------------------------------------------------------------------------- grating
def cell_model():
    m = copy.deepcopy(fg.presets()["TiO2-Stege auf Glas (Metaoberfläche)"])      # ridge 300 nm high, cover 640 nm, period 400 nm
    return m


def test_emitter_defaults_complete():
    em = fe.emitter_complete({"x_nm": 10.0})
    assert em["scan"]["nodes"] == 6 and em["scan"]["guided"] == "auto" and em["stage"] == "B"
    em = fe.emitter_complete({"scan": {"nodes": 8}})
    assert em["scan"]["nodes"] == 8 and em["scan"]["kx_nodes"] == 4


def test_emitter_validation_position():
    m = cell_model()
    ok = fe.emitter_defaults()
    ok.update(x_nm=200.0, y_nm=450.0, sigma_nm=10.0)
    assert fe.emitter_validate(m, ok, 600.0) == []
    inside = dict(ok, y_nm=300.0)                                                           # on the top face of the ridge
    assert any("mehr als ein Material" in t for _, t in fe.emitter_validate(m, inside, 600.0))
    edge = dict(ok, x_nm=20.0)                                                              # closer than 6 sigma to the cell edge
    assert any("innerhalb der Zelle" in t for _, t in fe.emitter_validate(m, edge, 600.0))
    above = dict(ok, y_nm=5000.0)
    assert any("außerhalb des physikalischen Gebiets" in t for _, t in fe.emitter_validate(m, above, 600.0))
    wide = dict(ok, sigma_nm=-1.0)
    assert any("positiv" in t for _, t in fe.emitter_validate(m, wide, 600.0))
    m["sweep"]["mode"] = "theta"
    assert any(lvl == "warn" for lvl, _ in fe.emitter_validate(m, ok, 600.0))


def test_emitter_in_lossy_medium_is_refused():
    m = cell_model()
    m["materials"]["Luft"] = {"type": "eps", "re": 1.0, "im": 0.05}
    em = dict(fe.emitter_defaults(), x_nm=200.0, y_nm=450.0, sigma_nm=10.0)
    assert any("verlustbehaftet" in t for _, t in fe.emitter_validate(m, em, 600.0))


def test_emitter_wavelengths():
    m = cell_model()
    m["sweep"] = {"mode": "wavelength", "start": 600.0, "stop": 700.0, "n": 3}
    assert fe.emitter_wavelengths(m) == [600.0, 650.0, 700.0]
    m["sweep"]["mode"] = "none"
    assert fe.emitter_wavelengths(m) == [m["incidence"]["wavelength_nm"]]


def stage_b_result():
    pts = []
    for i, lam in enumerate((600.0, 650.0)):
        pts.append(dict(i=i, wavelength=lam, stage="B", purcell=1.4, P_em=1.0, P_bulk=0.7, n_host=1.0, lossy=False, samples=100, plane_wave_solves=64,
                        purcell_by_orientation=dict(x=1.2, y=1.6, z=1.1, isotropic=1.3), fractions=dict(up=0.5, down=0.4, nonradiated=0.1),
                        aperture=[dict(NA=0.5, side="up", theta_max_deg=30.0, power=0.1, fraction=0.1), dict(NA=0.9, side="up", theta_max_deg=64.0, power=0.3, fraction=0.3)],
                        seconds=3.0))
    return dict(task="emitter", emitter=dict(stage="B", moment="isotropic", x_nm=200, y_nm=450, sigma_nm=10, wavelengths=[600, 650]), points=pts,
                cost=dict(stage="B", per_wavelength=dict(samples=100, plane_wave_solves=64, solves=164), wavelengths=2, solves=328, dofs=10477,
                          memory="10477 DoFs", seconds=125.0))


def test_emitter_tables_and_figures_stage_b():
    res = stage_b_result()
    df = fe.emitter_dataframe(res)
    assert len(df) == 2 and "F_P isotropic" in df and "NA 0.5 (up)" in df and df["oben"].iloc[0] == 0.5
    assert fe.fig_emit_purcell(res) is not None and fe.fig_emit_fractions(res) is not None
    assert "125" not in fe.cost_text(res["cost"]) and "2 min 5 s" in fe.cost_text(res["cost"])
    assert fe.cost_text(None) == ""


def test_emitter_tables_and_figures_stage_a_and_c():
    pa = dict(i=0, wavelength=600.0, stage="A", kx=0.0, beta=0.0, P_cell=1.0, up=0.6, down=0.3, absorbed=0.0, guided=0.1, pml_leak=0.0,
              orders_up=[[0, 0.5], [1, 0.05], [-1, 0.05]], orders_down=[[0, 0.3]], seconds=1.0)
    ra = dict(emitter=dict(stage="A"), points=[pa])
    assert len(fe.emitter_dataframe(ra)) == 1 and fe.fig_emit_orders(ra) is not None
    pc = dict(i=0, wavelength=600.0, stage="C", theta_deg=[0, 30, 60], phi_deg=[0, 90], sides=["up", "down"],
              dP_dOmega={"up": np.ones((3, 2)).tolist(), "down": np.ones((3, 2)).tolist()}, P_bulk=2.0, n=dict(up=1.0, down=1.5),
              aperture=[dict(NA=0.5, side="up", power=0.1, per_P_bulk=0.05)], seconds=2.0)
    rc = dict(emitter=dict(stage="C"), points=[pc])
    assert fe.emitter_dataframe(rc)["NA 0.5 (up)"].iloc[0] == 0.05
    assert fe.fig_emit_pattern(rc) is not None and fe.fig_emit_pattern(rc, 0, True) is not None
    assert fe.fig_emit_pattern(ra) is not None                                              # wrong stage: a placeholder, no exception


def test_mesh_is_refined_around_the_emitter(tmp_path):
    pytest.importorskip("gmsh")
    m = cell_model()
    ms = dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=False, pml_cells_per_wavelength=6.0, lam_min_nm=600.0, lam_max_nm=650.0)
    try:
        plain = fg.build_mesh(m, ms, tmp_path / "plain")
        m["emitter"] = dict(fe.emitter_defaults(), x_nm=200.0, y_nm=450.0, sigma_nm=10.0)
        fine = fg.build_mesh(m, ms, tmp_path / "fine")
    except RuntimeError as exc:
        pytest.skip(f"gmsh unavailable: {exc}")
    assert fine["cells"] > plain["cells"]
    assert fine["edge_min_nm"] < plain["edge_min_nm"]
    def local_edge(folder):                                                                 # largest edge of the cells within 40 nm of the emitter
        z = np.load(folder / "mesh_plot.npz")
        xy, tri = z["xy"], z["tri"]
        c = xy[tri].mean(axis=1)
        near = np.hypot(c[:, 0] - 200.0, c[:, 1] - 450.0) < 40.0
        a, b, d = xy[tri[near, 0]], xy[tri[near, 1]], xy[tri[near, 2]]
        return near.sum(), np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(d - b, axis=1), np.linalg.norm(a - d, axis=1)]).max()

    n_fine, h_fine = local_edge(tmp_path / "fine")
    n_plain, h_plain = local_edge(tmp_path / "plain")
    assert n_fine > 3 * n_plain and h_fine < 0.5 * h_plain and h_fine < 45.0                # about 2 sigma = 20 nm


def test_preview_marks_the_emitter():
    m = cell_model()
    m["emitter"] = dict(fe.emitter_defaults(), x_nm=200.0, y_nm=450.0, sigma_nm=10.0)
    fig = fg.preview_figure(m)
    assert any(line.get_marker() == "*" for line in fig.axes[0].lines)


# ---------------------------------------------------------------------------------------------------------------- off the axis (axi)
def axi_model(radii=(0.0, 100.0, 200.0)):
    m = fa.complete(fa.default_model())
    m["parts"] = [dict(type="sphere", material="Glas", z_center=0.0, radius=300.0)]
    m["emitter"].update(z_nm=450.0, sigma_nm=8.0, orientation="axial")
    m["emitter"]["sweep"] = dict(mode="fixed", start=700.0, stop=800.0, n=2, linewidths=1.5)
    if radii is not None:
        m["emitter"]["radii_nm"] = list(radii)
    return m


def test_offaxis_helpers():
    m = axi_model()
    assert fa.offaxis(m) and fa.emitter_radii(m) == [0.0, 100.0, 200.0]
    assert fa.source_box(m) == 48.0                                                         # 6 sigma off the axis
    m0 = axi_model(None)
    assert not fa.offaxis(m0) and fa.emitter_radii(m0) == [0.0] and fa.source_box(m0) == 32.0   # 4 sigma on the axis
    assert fa.emitter_radii(dict(m, emitter=dict(m["emitter"], radii_nm=[200.0, 100.0, 100.0]))) == [100.0, 200.0]


def test_offaxis_validation():
    m = axi_model()
    assert [t for lvl, t in fa.validate(m, "emitter") if lvl == "error"] == []
    near = axi_model((0.0, 200.0))
    near["emitter"]["z_nm"] = 250.0                                                         # inside the sphere at rho = 200 nm? z = 250 < 224 + 48
    assert any("mehreren Materialien" in t for lvl, t in fa.validate(near, "emitter") if lvl == "error")
    res = axi_model()
    res["emitter"]["sweep"]["mode"] = "resonance"
    assert any("fester Bereich" in t for lvl, t in fa.validate(res, "emitter") if lvl == "error")
    neg = axi_model((-10.0, 50.0))
    assert any("≥ 0" in t for lvl, t in fa.validate(neg, "emitter") if lvl == "error")


def test_offaxis_mesh_has_one_box_per_radius(tmp_path):
    pytest.importorskip("gmsh")
    m = axi_model()
    ms = dict(cells_per_wavelength=3.0, skin_cells=1.0, interface_factor=0.7, curved=True, pml_cells_per_wavelength=4.0, lam_min_nm=700.0, lam_max_nm=800.0)
    m["domain"] = dict(margin_nm=500.0, pml_nm=800.0)
    try:
        stats = fa.build_mesh(m, ms, tmp_path, task="emitter")
    except RuntimeError as exc:
        pytest.skip(f"gmsh unavailable: {exc}")
    assert stats["source_box_nm"] == 48.0
    z = np.load(tmp_path / "mesh_plot.npz")
    xy, tri = z["xy"], z["tri"]
    c = xy[tri].mean(axis=1)
    for rho in (0.0, 100.0, 200.0):                                                          # cells exist around every emitter position
        assert np.any(np.hypot(c[:, 0] - rho, c[:, 1] - 450.0) < 20.0)
    text = (tmp_path / "mesh.msh").read_text()
    assert "Emitterkasten" in text


def test_shift_msh_moves_only_the_y_coordinate(tmp_path):
    import fem_axi_worker as w
    src = tmp_path / "a.msh"
    src.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n10 5 0\n20 -5 0\n$EndNodes\n$Elements\n0 0 0 0\n$EndElements\n",
                   encoding="utf-8")
    w.shift_msh(src, tmp_path / "b.msh", 2.0)
    lines = (tmp_path / "b.msh").read_text().splitlines()
    i = lines.index("$Nodes")
    assert lines[i + 1:i + 3] == ["1 3 1 3", "2 1 0 3"]
    coords = [[float(v) for v in row.split()] for row in lines[i + 6:i + 9]]
    assert coords == [[0.0, -2.0, 0.0], [10.0, 3.0, 0.0], [20.0, -7.0, 0.0]]
    assert lines[i + 9] == "$EndNodes"


def offaxis_point(i=0, lam=700.0):
    rho = [0.0, 100.0, 200.0]
    tot = [np.diag([1.5 + k, 1.4 + k, 3.0 - k]) + np.array([[0, 0, 0.1 * k], [0, 0, 0], [0.1 * k, 0, 0]]) for k in range(3)]
    rad = [0.9 * t for t in tot]
    det = [[0.4 * t, 0.3 * t] for t in tot]
    return dict(index=i, lam_nm=lam, offaxis=True, radii_nm=rho, z_nm=450.0, n_host=1.0, detectors=["NA 0.5 oben", "Substrat"],
                convergence=dict(energy_projection=0.0), time_s=1.0, A_tot=[t.tolist() for t in tot], A_rad=[t.tolist() for t in rad],
                A_det=[[d.tolist() for d in row] for row in det])


def test_offaxis_figures_and_table():
    pts = [offaxis_point(0, 700.0), offaxis_point(1, 800.0), dict(index=2, lam_nm=900.0, error="x")]
    for which in ("purcell", "radiated"):
        assert fa.fig_offaxis(pts, which) is not None
    assert fa.fig_offaxis([dict(index=0, lam_nm=1.0, error="x")]) is not None
    df = fa.offaxis_dataframe(pts)
    assert len(df) == 6 and set(df["ρ (nm)"]) == {0.0, 100.0, 200.0}
    assert df["F_ρ"].iloc[0] == pytest.approx(1.5) and df["F_z"].iloc[2] == pytest.approx(1.0)
    assert df["abgestrahlt z"].iloc[0] == pytest.approx(0.9)
    assert df["A_xz"].iloc[1] == pytest.approx(0.1)
    assert df["NA 0.5 oben (Spur-Anteil)"].iloc[0] == pytest.approx(0.4)


def test_offaxis_preview_figure():
    fig = fa.preview_figure(axi_model())
    stars = [ln for ln in fig.axes[0].lines if ln.get_marker() == "*"]
    assert len(stars) == 3                                                                  # one per distance (with its mirror image)


def test_library_features():
    for name in ("Ti", "Cr"):
        assert name in fm.LIBRARY


def test_aperture_from_grid_cuts_at_the_cone_edge():
    import fem_emit_worker as w
    theta = np.radians([0, 5, 10, 15, 20, 25, 30])
    phi = np.radians([0.0, 90.0])
    flat = np.full((7, 2), 3.0)                                                              # isotropic dP/dOmega = 3 W/sr
    exact = 3.0 * 2 * math.pi * (1 - math.cos(math.asin(0.5 / 1.000293)))
    got = w.aperture_from_grid(theta, phi, flat, 0.5, 1.000293)                               # NA 0.5 in air: edge at 29.9993 deg, grid ends at 30
    assert got == pytest.approx(exact, rel=2e-3)
    assert w.aperture_from_grid(theta[:5], phi, flat[:5], 0.5, 1.0) < 0.6 * exact              # a grid that ends at 20 deg covers part of the cone only
    assert w.aperture_from_grid(theta[:1], phi, flat[:1], 0.5, 1.0) == 0.0
