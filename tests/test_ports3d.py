"""3D waveguides and ports: model, job, worker on small structured meshes (straight guide, coupler) and on a Gmsh file, app results."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_ports3d as fp  # noqa: E402
import fem_threed as ft  # noqa: E402


def errors(m):
    return [t for lv, t in fp.validate(m) if lv == "error"]


# ------------------------------------------------------------------------------------------------------------------ model and job
def test_presets_validate_and_jobs_are_json():
    pr = fp.presets()
    assert len(pr) == 6 and {m["device"] for m in pr.values()} == {"straight", "coupler", "ring"}
    for name, m in pr.items():
        assert not errors(m), name
        j = json.loads(json.dumps(fp.job(m)))
        assert j["task"] == "ports" and j["eps_core"][0] == [3.48**2, 0.0] and j["eps_clad"][0] == [1.44**2, 0.0]
    assert fp.job(pr["Richtkoppler Si/SiO2 (Halbflächen-Ports, klein)"])["num_modes"] == 1
    assert fp.job(pr["Richtkoppler Si/SiO2 (Supermoden-Ports)"])["num_modes"] == 2
    assert fp.job(next(iter(pr.values())))["num_modes"] == 1
    assert ft.worker_script(fp.job(next(iter(pr.values())))) == "fem_ports3d_worker.py" and ft.worker_script({"task": "scatter"}) == "fem_threed_worker.py"


def test_validation_of_ports_models():
    m = fp.default_model()
    m["length_nm"] = 200.0                                          # fewer than two cells of 150 nm
    assert any("zwei Zellen" in t for t in errors(m))
    m = fp.presets()["Richtkoppler Si/SiO2 (Supermoden-Ports)"]
    m["num_modes"] = 1
    assert any("zwei Moden" in t for t in errors(m))
    m = fp.default_model()
    m["materials"]["Kern"] = {"type": "index", "n": 1.3, "k": 0.0}
    assert any("keine geführte Mode" in t for lv, t in fp.validate(m) if lv == "warning")
    m = fp.default_model()
    m["device"] = "file"
    assert any("hochladen" in t for t in errors(m))
    m["file"].update(name="a.msh", path="a.msh", tags={"cells": {"1": 4}, "facets": {"11": 2}, "bbox": [[0] * 3, [1e-6] * 3]}, ports=[])
    assert any("Port" in t for t in errors(m))


def test_grid_and_cell_count_and_outline():
    m = fp.default_model()
    assert fp.cell_count(m) == (len(fp.grid_lines(fp.x_edges(m), 150.0)) - 1) * (len(fp.grid_lines(fp.y_edges(m), 150.0)) - 1) * 8
    lines = fp.grid_lines([0.0, 100.0, 400.0], 150.0)
    assert np.allclose(lines, [0.0, 100.0, 250.0, 400.0])      # the edges are always grid lines
    m["device"] = "coupler"
    ol = fp.outline(m)
    assert len(ol["xz"]) == 4 and len(ol["xy"]) == 2


def test_dataframes_and_figures():
    pts = [{"index": 0, "lam_nm": 1550.0, "R": 0.01, "T": 0.98, "loss": 0.01, "n_eff0": 1.9, "bar": 0.9, "cross": 0.08, "cross_cmt": 0.085, "back": 0.0, "dofs": 100, "seconds": 1.0,
            "channels": [{"port": 0, "mode": 0, "beta": 1.0}, {"port": 1, "mode": 0, "beta": 1.0}], "S_re": [[0.1, 0.9], [0.9, 0.1]], "S_im": [[0.0, 0.1], [0.1, 0.0]]}]
    df = fp.results_dataframe(pts, "coupler", "full")
    assert {"Durchgang (bar)", "Kopplung (cross)", "cross (gekoppelte Moden)"} <= set(df.columns)
    s = fp.s_matrix_dataframe(pts[0])
    assert s.shape == (2, 3) and s["nach P2·M1"].iloc[0] == pytest.approx(0.82)
    assert fp.fig_sparams(pts).axes
    g = np.linspace(0, 1e-6, 5)
    npz = {"xs": g, "ys": g, "zs": g, "xz": np.ones((5, 5)), "xy": np.ones((5, 5))}
    assert fp.fig_field(npz, "xz", {"xz": [([0, 1], [0, 0])]}).axes and fp.fig_field(npz, "xy").axes


# ------------------------------------------------------------------------------------------------------------------ worker
def run_worker(folder, m, mode=None):
    import fem_ports3d_worker as pw
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(fp.job(m)), encoding="utf-8")
    code = pw.main([str(folder)] + ([mode] if mode else []))
    return code, json.loads((folder / "results.json").read_text(encoding="utf-8")) if (folder / "results.json").is_file() else None


def test_straight_guide_transmits_with_the_2d_mode_as_reference(tmp_path):
    pytest.importorskip("hpfem")
    m = fp.default_model()
    code, res = run_worker(tmp_path / "s", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p
    assert p["T"] > 0.99 and p["R"] < 0.01 and abs(p["loss"]) < 1e-6                           # power balance of the S-matrix
    assert p["n_eff0"] == pytest.approx(p["n_eff_2d"], rel=1e-6)                                # the port mode is the 2D mode of the section on the same mesh
    assert 2.0 < p["n_eff0"] < p["n_eff_ref"] < 2.5                                             # p = 1 on 150 nm cells underestimates n_eff (about 9 %); the fine 2D reference does not
    assert 0.0 <= p["phase_error"] <= np.pi
    f = np.load(tmp_path / "s" / p["field_file"])
    assert f["xz"].shape == (31, 61) and np.nanmax(f["xz"]) > 0
    est = ft.run_estimate(sys.executable, HERE.parent, tmp_path / "e", fp.job(m))
    assert est["cells"] > 1000 and (tmp_path / "e" / "mesh_ports.npz").is_file()


def test_grid_box_puts_the_faces_on_the_core_edges():
    hpfem = pytest.importorskip("hpfem")
    import fem_ports3d_worker as pw
    mesh = pw.grid_box(hpfem, np.array([0.0, 1.0, 3.0, 4.0]), np.array([0.0, 2.0, 2.5]), np.array([0.0, 1.0, 5.0]))
    assert np.asarray(mesh.cell_volumes).sum() == pytest.approx(4 * 2.5 * 5)
    assert {t: len(mesh.facets_with_tag(t)) for t in range(1, 7)} == {1: 8, 2: 8, 3: 12, 4: 12, 5: 12, 6: 12}
    v = np.asarray(mesh.vertices)
    assert set(np.round(v[:, 0], 9)) == {0.0, 1.0, 3.0, 4.0}


def test_coupler_with_supermode_ports_has_a_consistent_s_matrix_and_references(tmp_path):
    pytest.importorskip("hpfem")
    m = fp.presets()["Richtkoppler Si/SiO2 (Supermoden-Ports)"]
    code, res = run_worker(tmp_path / "c", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p and p["bar"] + p["cross"] == pytest.approx(1.0) and 0.0 <= p["cross"] < 0.3
    assert p["R"] < 0.02 and abs(p["loss"]) < 1e-6 and p["back"] < 0.01
    assert p["coupling_length_nm"] > 0 and p["cross_cmt"] == pytest.approx(np.sin(p["kappa"] * 1000e-9) ** 2)
    assert p["coupling_length_ref_nm"] > p["coupling_length_nm"]                                # p = 1 on 150 nm cells couples too strongly (about 10 µm against 25 µm)
    assert 2.0 < p["n_eff_even_ref"] < 2.6 and p["n_eff_odd_ref"] < p["n_eff_even_ref"]
    assert p["phase_error_even"] <= np.pi and p["phase_error_odd"] <= np.pi


def test_half_face_ports_have_four_channels(tmp_path):
    pytest.importorskip("hpfem")
    m = fp.presets()["Richtkoppler Si/SiO2 (Halbflächen-Ports, klein)"]
    m["field"] = False
    code, res = run_worker(tmp_path / "h", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p and len(p["channels"]) == 4 and len(p["S_re"]) == 4
    assert p["bar"] + p["cross"] + p["R"] + p["back"] == pytest.approx(1.0, abs=1e-6)         # all four outgoing channels of the excited half port


def test_memory_limit_stops_the_ports_run(tmp_path):
    pytest.importorskip("hpfem")
    m = fp.default_model()
    m["max_gb"] = 1e-4
    code, res = run_worker(tmp_path / "m", m)
    assert code == 1 and res is None


def gmsh_straight_guide(path, width=450.0, height=220.0, margin_x=600.0, margin_y=450.0, length=1200.0, cell=150.0):
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("guide")
        occ = gmsh.model.occ
        X, Y = width / 2 + margin_x, height + margin_y
        box = occ.addBox(-X, -margin_y, 0, 2 * X, Y + margin_y - height + height, length)
        core = occ.addBox(-width / 2, 0, 0, width, height, length)
        _, parts = occ.fragment([(3, box)], [(3, core)])
        occ.synchronize()
        core_v = {t for _, t in parts[1]}
        clad_v = {t for _, t in parts[0]} - core_v
        gmsh.model.addPhysicalGroup(3, sorted(clad_v), 1)
        gmsh.model.addPhysicalGroup(3, sorted(core_v), 2)
        walls, left, right = [], [], []
        for dim, tag in gmsh.model.getBoundary([(3, t) for t in clad_v | core_v], oriented=False):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
            if z1 < 1e-6:
                left.append(tag)
            elif z0 > length - 1e-6:
                right.append(tag)
            else:
                walls.append(tag)
        gmsh.model.addPhysicalGroup(2, sorted(set(walls)), 10)
        gmsh.model.addPhysicalGroup(2, sorted(set(left)), 11)
        gmsh.model.addPhysicalGroup(2, sorted(set(right)), 12)
        gmsh.option.setNumber("Mesh.MeshSizeMax", cell)
        gmsh.option.setNumber("Mesh.MeshSizeMin", cell / 2)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def test_gmsh_file_device_matches_the_structured_guide(tmp_path):
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    path = tmp_path / "guide.msh"
    gmsh_straight_guide(path)
    m = fp.default_model()
    m.update(device="file")
    m["materials"].update({"Kern": {"type": "index", "n": 3.48, "k": 0.0}, "Mantel": {"type": "index", "n": 1.44, "k": 0.0}})
    (tmp_path / "i").mkdir()
    info = ft.run_inspect(sys.executable, HERE.parent, tmp_path / "i", path, "nm", "ports")
    assert set(info["cells"]) == {"1", "2"} and set(info["facets"]) == {"10", "11", "12"}
    m["file"].update(name="guide.msh", path=str(path), scale="nm", tags=info, tag_material={"1": "Mantel", "2": "Kern"}, pec_tags=[10], ports=[dict(tag=11, modes=1), dict(tag=12, modes=1)])
    assert not errors(m)
    j = fp.job(m)
    assert j["scale"] == 1e-9 and j["pec_tags"] == [10] and len(j["ports"]) == 2 and set(j["tag_eps"]) == {"1", "2"}
    code, res = run_worker(tmp_path / "f", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p and p["T"] > 0.95 and abs(p["loss"]) < 1e-6
    assert 2.1 < p["n_eff0"] < 2.45                                                           # the port mode of the 2D section solver on an unstructured mesh (converged: about 2.36)


# ------------------------------------------------------------------------------------------------------------------ ring resonator
def ring_model(coarse=True, **ring):
    """The test ring preset on a very coarse mesh (300 nm cells, p = 1: about 4500 unknowns in the half model, a few seconds per wavelength)."""
    m = copy.deepcopy([v for k, v in fp.presets().items() if k.startswith("Ringresonator Add-Drop (Test")][0])
    if coarse:
        m["ring"].update(cell_core_nm=300.0, cell_far_nm=600.0)
    m["ring"].update(ring)
    m["sweep"] = {"start": 1550.0, "stop": 1550.0, "count": 1}
    return m


def test_ring_presets_validate_and_the_job_carries_the_geometry():
    pr = {k: v for k, v in fp.presets().items() if v["device"] == "ring"}
    assert len(pr) == 3
    for name, m in pr.items():
        assert not errors(m), name
        j = json.loads(json.dumps(fp.job(m)))
        assert j["device"] == "ring" and j["task"] == "ports" and j["half"] is True
        assert j["ng"]["lams"] == [j["target_nm"] - 5.0, j["target_nm"], j["target_nm"] + 5.0] and len(j["ng"]["core"]) == 3
        assert ft.worker_script(j) == "fem_ports3d_worker.py"
    test, mid, prod = (pr[k] for k in sorted(pr, key=lambda s: ("Test" not in s, "mittel" not in s)))
    j = fp.job(test)
    assert j["auto"] is False and j["wavelengths_nm"] == [1530.0, 1550.0, 1570.0] and j["frozen"] is False and len(j["eps_core"]) == 3
    assert j["order_core"] == 1 and j["order_far"] == 1 and j["radius_nm"] == 1500.0 and j["cell_core_nm"] == 150.0
    j = fp.job(mid)
    assert j["auto"] is True and j["wavelengths_nm"] == [] and j["frozen"] is True and len(j["eps_core"]) == 1 and j["target_nm"] == 1557.0
    assert j["order_core"] == 2 and j["radius_nm"] == 2000.0
    j = fp.job(prod)
    assert j["cell_core_nm"] == 100.0 and j["order_core"] == 3 and j["order_far"] == 2 and j["max_gb"] == 400.0
    assert any("Millionen" in t for lv, t in fp.validate(prod) if lv == "warning")        # the production mesh is flagged as heavy
    assert fp.lam_ref(mid) == 1557.0 and fp.lam_ref(test) == pytest.approx(1550.0) and fp.wavelengths(mid) == []


def test_ring_validation():
    m = ring_model()
    assert any("Resonator nicht auf" in t for lv, t in fp.validate(m) if lv == "warning")                # p = 1
    m["ring"]["order_core"] = 2
    assert not any("Resonator nicht auf" in t for lv, t in fp.validate(m))
    m["ring"]["radius_nm"] = 200.0
    assert any("Ringradius" in t for t in errors(m))
    m = ring_model(cell_core_nm=0.0)
    assert any("positiv" in t for t in errors(m))
    m = ring_model(cell_far_nm=100.0, cell_core_nm=200.0)
    assert any("ferne Zelle" in t for t in errors(m))
    m = ring_model(margin_z_nm=100.0)
    assert any("PEC-Wand" in t for lv, t in fp.validate(m))
    m = ring_model(backend="mumps")
    assert any("mumps" in t for lv, t in fp.validate(m))
    m = ring_model(scan="auto", search_nm=0.0)
    assert any("Suchfenster" in t for t in errors(m))
    m = ring_model(scan="auto")
    m["materials"]["Kern"] = {"type": "library", "name": "Si"}
    assert any("eingefroren" in t for lv, t in fp.validate(m))
    m["sweep"]["count"] = 3
    m["ring"]["scan"] = "manual"
    assert any("vier Wellenlängen" in t for lv, t in fp.validate(m))


def synthetic_pole(lam, lam0=1541.0, g=0.2):
    """Amplitudes of a lossless symmetric add-drop filter: through t = δ/(δ + i g), drop d = −i g/(δ + i g) (|t|² + |d|² = 1, pole at λ0 − i g)."""
    d = lam - lam0
    return d / (d + 1j * g), -1j * g / (d + 1j * g)


def test_fit_pole_recovers_a_known_resonance():
    import fem_ports3d_worker as pw

    lam = np.linspace(1540.0, 1542.0, 9)
    t, d = synthetic_pole(lam)
    pole, residual = pw.fit_pole(lam, [t, d])
    assert pole.real == pytest.approx(1541.0, abs=1e-9) and abs(pole.imag) == pytest.approx(0.2, abs=1e-9) and residual < 1e-9
    pts = [{"lam_nm": float(x), "s": [[0.0, 0.0], [a.real, a.imag], [b.real, b.imag], [0.0, 0.0]], "power": [0.0, abs(a) ** 2, abs(b) ** 2, 0.0]} for x, a, b in zip(lam, t, d)]
    f = pw.ring_fit(pts)
    assert f["wavelength"] == pytest.approx(1541.0, abs=1e-9) and f["fwhm"] == pytest.approx(0.4, abs=1e-9) and f["q"] == pytest.approx(1541.0 / 0.4, rel=1e-8)
    assert f["max_drop"] > 0.9 and f["min_through"] < 0.05
    assert pw.ring_fit(pts[:3]) is None                                                   # fewer than four wavelengths


def test_ring_dataframes_and_figures():
    pts = [{"index": i, "lam_nm": 1540.0 + i, "R": 0.01, "T": 0.99, "through": 0.9 - 0.1 * i, "drop": 0.08 + 0.1 * i, "add": 0.01, "power_sum": 1.0, "loss": 0.0, "n_eff0": 2.35,
            "s": [[0.1, 0.0], [0.9, 0.1], [0.1, 0.3], [0.05, 0.0]], "power": [0.01, 0.8, 0.2, 0.0], "dofs": 100, "seconds": 1.0} for i in range(4)]
    ring = {"fit": {"wavelength": 1541.2, "fwhm": 0.4, "q": 3853.0, "residual": 1e-3, "points": 4, "max_drop": 0.99, "min_through": 0.01}, "lam_ref_nm": 1540.7, "chosen_m": 15, "fsr_ref_nm": 46.0,
            "n_eff_bus": 2.35, "n_g": 4.15, "fsr_ng_nm": 46.4}
    df = fp.results_dataframe(pts, "ring", "")
    assert {"Through (Port 2)", "Drop (Port 3)", "Add (Port 4)", "Summe"} <= set(df.columns) and len(df) == 4
    s = fp.s_matrix_dataframe(pts[0])
    assert s.shape == (4, 4) and s["|S|² (Leistung)"].iloc[1] == pytest.approx(0.82)
    tab = fp.ring_summary_dataframe(ring)
    assert len(tab) >= 12 and tab["Wert"].astype(str).str.contains("1541.2000").any() and fp.ring_summary_dataframe({}).empty
    assert fp.fig_ring(pts, ring).axes and fp.fig_ring(pts, {}).axes
    g = np.linspace(-3e-6, 3e-6, 12)
    npz = {"xs": g, "ys": g, "e2": np.ones((12, 12))}
    assert fp.fig_ring_field(npz, ring_model()).axes


def test_ring_mesh_physical_groups(tmp_path):
    pytest.importorskip("gmsh")
    import fem_ring3d_mesh as rm

    g = rm.geometry(radius=1500.0, cell_core=300.0, cell_far=600.0, half=True)
    assert rm.extent(g) == (1500.0 + 450.0 + 200.0, 1500.0 + 225.0 + 1000.0, 1500.0 + 450.0 + 200.0 + 225.0 + 750.0)
    path = rm.build(g, str(tmp_path / "ring.msh"))
    text = Path(path).read_text(encoding="utf-8")
    names = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0]
    assert all(n in names for n in ('"cladding"', '"core"', '"wall"', '"port_left"', '"port_right"', '"symmetry"'))
    full = rm.build(rm.geometry(radius=1500.0, cell_core=300.0, cell_far=600.0, half=False), str(tmp_path / "full.msh"))
    assert '"symmetry"' not in Path(full).read_text(encoding="utf-8")
    d = rm.core_distance(g, [[1500.0, 0.0, 110.0], [0.0, 0.0, 110.0], [-3000.0, 2150.0, 110.0]])      # on the ring, at the centre, on bus B
    assert d[0] == 0.0 and d[1] == pytest.approx(1500.0 - 225.0) and d[2] == 0.0


def test_ring_worker_runs_with_energy_balance_reference_and_group_index(tmp_path):
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    m = ring_model(field_res=True)
    code, res = run_worker(tmp_path / "r", m)
    assert code == 0 and res is not None
    assert res["meta"]["device"] == "ring" and res["meta"]["dofs"] > 3000
    p = res["points"][0]
    assert "error" not in p and abs(p["power_sum"] - 1.0) < 1e-5 and abs(p["loss"]) < 1e-5                    # lossless, PEC walls: the four port powers sum to one
    assert p["R"] == p["power"][0] and p["through"] == p["power"][1] and p["drop"] == p["power"][2] and p["add"] == p["power"][3]
    assert 1.5 < p["n_eff0"] < 2.5                                                                             # the port mode of the coarse mesh (converged: about 2.36)
    ring = res["ring"]
    assert 2.2 < ring["n_eff_bus"] < 2.5 and 3.5 < ring["n_g"] < 4.8                                          # SOI strip 450 x 220 nm: n_g about 4.2
    assert ring["fsr_ng_nm"] == pytest.approx(1550.0**2 / (ring["n_g"] * 2 * np.pi * 1500.0), rel=0.02)
    assert set(ring["reference"]) == {"13", "14", "15"} and 1450.0 < ring["lam_ref_nm"] < 1700.0 and ring["fsr_ref_nm"] > 40.0
    assert abs(ring["lam_ref_nm"] - ring["reference"][str(ring["chosen_m"])]) < 1e-12
    assert ring["fit"] is None                                                                                 # one wavelength: no pole fit
    f = np.load(tmp_path / "r" / p["field_file"])
    assert f["e2"].shape == (len(f["xs"]), len(f["ys"])) and np.nanmax(f["e2"]) > 0
    est = ft.run_estimate(sys.executable, HERE.parent, tmp_path / "e", fp.job(m))
    assert est["cells"] > 2000 and est["dofs"] == res["meta"]["dofs"] and (tmp_path / "e" / "mesh_ports.npz").is_file()


def test_ring_automatic_scan_with_a_synthetic_pole(tmp_path, monkeypatch):
    """The automatic search (nine points around the reference, the pole fit, nine points around the fitted pole) with the solver replaced by a lossless filter of known pole."""
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    import fem_ports3d_worker as pw

    lam0, g = 1541.0, 0.2
    calls = []

    def fake_solve(h, nd, job, info, lam, core, clad, solver, launch=True):
        t, d = synthetic_pole(lam, lam0, g)
        calls.append(lam)
        s = [0.0, t, d, 0.0]
        power = [float(abs(v) ** 2) for v in s]
        return ({"lam_nm": float(lam), "s": [[complex(v).real, complex(v).imag] for v in s], "power": power, "power_sum": sum(power), "R": power[0], "through": power[1], "drop": power[2],
                 "add": power[3], "T": power[1] + power[2], "loss": 1 - sum(power), "n_eff0": 2.35, "seconds": 0.0, "dofs": int(nd.num_dofs)}, None, None)

    monkeypatch.setattr(pw, "ring_solve", fake_solve)
    m = ring_model(scan="auto", target_nm=1545.0, search_nm=4.0)
    code, res = run_worker(tmp_path / "a", m)
    assert code == 0
    assert len(res["points"]) == 18 and len(calls) == 18
    ring = res["ring"]
    assert ring["lam_ref_nm"] == pytest.approx(1540.66, abs=0.2) and ring["chosen_m"] == 15
    f = ring["fit"]
    assert f["wavelength"] == pytest.approx(lam0, abs=1e-6) and f["fwhm"] == pytest.approx(2 * g, abs=1e-6) and f["max_drop"] > 0.99
    first, second = calls[:9], calls[9:]
    assert first == pytest.approx(list(ring["lam_ref_nm"] + np.linspace(-4.0, 4.0, 9)))
    assert second == pytest.approx(list(lam0 + 2 * g * np.linspace(-2.0, 2.0, 9)), abs=1e-5)             # refined around the pole, ±2 FWHM
    log = (tmp_path / "a" / "results.json").read_text(encoding="utf-8")
    assert '"fit"' in log


def test_ring_memory_limit_and_unavailable_backend_stop_the_run(tmp_path):
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    m = ring_model()
    m["max_gb"] = 1e-4
    code, res = run_worker(tmp_path / "m", m)
    assert code == 1 and res is None
    import hpfem
    if hpfem.DirectSolverBackend.MUMPS in hpfem.available_backends():
        pytest.skip("this build has MUMPS")
    m = ring_model(backend="mumps", reference=False)
    code, res = run_worker(tmp_path / "b", m)
    assert code == 1 and res["points"] == []                                                                  # clear error before any solve


def test_app_ring_device_and_results_without_exception(tmp_path):
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    from streamlit.testing.v1 import AppTest

    run_worker(tmp_path / "threed", ring_model(field_res=True))
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=240)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for name in [n for n, pm in ft.presets().items() if pm["task"] == "ports" and pm["p3"]["device"] == "ring"]:
        at.selectbox(key="td_preset").select(name).run()
        [b for b in at.button if b.key == "td_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])
    assert at.session_state["td_model"]["p3"]["device"] == "ring"
    ver = at.session_state["td_ver"]
    at.radio(key=f"p3_rsc_{ver}").set_value("auto").run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key=f"p3_dev_{ver}").set_value("straight").run()
    assert not at.exception, [e.value for e in at.exception]


# ------------------------------------------------------------------------------------------------------------------ app
def test_app_ports_task_and_results_without_exception(tmp_path):
    pytest.importorskip("hpfem")
    from streamlit.testing.v1 import AppTest

    m = copy.deepcopy(ft.default_model())
    m["task"] = "ports"
    run_worker(tmp_path / "threed", m["p3"])
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=180)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for name in [n for n, pm in ft.presets().items() if pm["task"] == "ports"]:
        at.selectbox(key="td_preset").select(name).run()
        [b for b in at.button if b.key == "td_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])
    assert at.session_state["td_model"]["task"] == "ports"
    ver = at.session_state["td_ver"]
    at.radio(key=f"p3_dev_{ver}").set_value("file").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("hochladen" in e.value for e in at.error)
