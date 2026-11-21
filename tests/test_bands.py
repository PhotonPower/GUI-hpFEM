"""Band structures of photonic crystals: model, path, gaps, mesh with periodic edges, short runs against the empty lattice."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_bands as fb  # noqa: E402


def empty(lattice, n_seg=1, bands=4):
    m = copy.deepcopy(fb.default_model())
    m.update(shapes=[], lattice=lattice, n_seg=n_seg, bands=bands)
    m["mesh"].update(cells_per_wavelength=5.0, order=3)
    return m


def test_presets_validate_and_job_is_json():
    for name, m in fb.presets().items():
        assert not [t for lv, t in fb.validate(m) if lv == "error"], name
        d = json.loads(json.dumps(fb.job(m)))
        assert len(d["k"]) == len(d["x"]) == 3 * m["n_seg"] + 1 and d["corner_idx"][0] == 0 and d["corner_idx"][-1] == len(d["k"]) - 1
    assert len(fb.presets()) >= 4


def test_validation_errors():
    m = fb.default_model()
    m["materials"]["Dielektrikum"] = {"type": "eps", "re": 12.0, "im": 0.1}
    assert any("verlustbehaftet" in t for lv, t in fb.validate(m) if lv == "error")
    m = fb.default_model()
    m["materials"]["Dielektrikum"] = {"type": "eps", "re": -3.0, "im": 0.0}
    assert any("positiv" in t for lv, t in fb.validate(m) if lv == "error")
    m = fb.default_model()
    m["shapes"][0]["r"] = -0.1
    m["bands"] = 0
    errs = [t for lv, t in fb.validate(m) if lv == "error"]
    assert any("Radius" in t for t in errs) and any("Bänder" in t for t in errs)
    m = fb.default_model()
    m["shapes"][0]["material"] = "gibt es nicht"
    assert any("fehlt" in t for lv, t in fb.validate(m) if lv == "error")


def test_paths_and_empty_lattice_reference():
    sq = fb.k_path(empty("square", 2))
    assert [lab for _, lab in sq[2]] == ["Γ", "X", "M", "Γ"] and np.allclose(sq[0][2], (0.5, 0.0)) and np.allclose(sq[0][4], (0.5, 0.5))
    hx = fb.k_path(empty("hex", 2))
    assert [lab for _, lab in hx[2]] == ["Γ", "M", "K", "Γ"]
    ks = hx[0]
    assert np.isclose(np.linalg.norm(ks[2]), 1 / np.sqrt(3)) and np.isclose(np.linalg.norm(ks[4]), 2 / 3)       # |M| = 1/sqrt 3, |K| = 2/3 in 2 pi / a
    assert np.all(np.diff(sq[1]) > 0)
    ref = fb.empty_lattice_bands(empty("square", 1, 3), 3)                                                     # X: 0.5, 0.5; M: 0.7071 x 2
    assert np.allclose(ref[1][:2], 0.5) and np.allclose(ref[2][:2], np.sqrt(0.5))
    refh = fb.empty_lattice_bands(empty("hex", 1, 3), 3)
    assert np.allclose(refh[1][:2], 1 / np.sqrt(3)) and np.allclose(refh[2][:3], 2 / 3)                         # M, K of the triangular lattice


def test_clean_bands_and_gaps():
    assert fb.clean_bands([0.0, 1e-6, 0.5, 0.7], 3, True) == [0.0, 0.5, 0.7]
    assert fb.clean_bands([0.2, 0.3, 0.5], 2, False) == [0.2, 0.3]
    f = np.array([[0.0, 0.5, 0.55], [0.2, 0.6, 0.95], [0.3, 0.55, 1.0]])
    gaps = fb.band_gaps(f)
    assert len(gaps) == 1 and gaps[0][0] == 1 and gaps[0][1] == pytest.approx(0.3) and gaps[0][2] == pytest.approx(0.5)      # bands 2 and 3 overlap or not
    b, lo, hi, w, mid, pct = fb.band_gaps(np.array([[0.1, 0.5], [0.2, 0.6]]))[0]
    assert (lo, hi, w, mid) == pytest.approx((0.2, 0.5, 0.3, 0.35)) and pct == pytest.approx(100 * 0.3 / 0.35)
    assert fb.band_gaps(np.array([[0.1, 0.15], [0.3, 0.4]])) == []                                             # overlap: no gap


def test_mesh_has_matching_periodic_edges_and_materials(tmp_path):
    pytest.importorskip("gmsh")
    hpfem = pytest.importorskip("hpfem")
    m = copy.deepcopy(fb.presets()["Dreiecksgitter: Luftlöcher r = 0,4 a in ε = 13"])
    m["mesh"].update(cells_per_wavelength=2.0)
    stats = fb.build_mesh(m, tmp_path)
    mesh = hpfem.read_gmsh(str(tmp_path / "mesh.msh"), 1.0, 2)
    assert stats["cells"] == mesh.num_cells
    tags = np.asarray(mesh.cell_tags)
    area_hex = np.sqrt(3) / 2
    areas = np.asarray(mesh.cell_volumes)
    holes = float(areas[tags == 1].sum())                      # material 1 = Luft (first in the dict)
    assert holes == pytest.approx(np.pi * 0.4 ** 2, rel=0.02)
    assert float(areas.sum()) == pytest.approx(area_hex, rel=1e-9)
    # a disc larger than the cell: continues on the opposite edges (periodic images), the cell stays fully covered
    big = copy.deepcopy(m)
    big["shapes"][0]["r"] = 0.55
    fb.build_mesh(big, tmp_path / "big")
    mesh2 = hpfem.read_gmsh(str(tmp_path / "big" / "mesh.msh"), 1.0, 2)
    assert float(np.asarray(mesh2.cell_volumes).sum()) == pytest.approx(area_hex, rel=1e-9)
    # the periodic pairs are accepted by the solver
    nd, h1 = hpfem.NedelecDofMap2D(mesh, 2), hpfem.DofMap2D(mesh, 2)
    import fem_bands_worker as bw
    job = fb.job(m)
    problem = bw.build_problem(hpfem, job, mesh)[2]
    assert problem.lattice_constant == pytest.approx(1.0)
    del nd, h1


def run_worker(model, folder):
    hpfem = pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    import fem_bands_worker as bw
    folder.mkdir(parents=True, exist_ok=True)
    fb.build_mesh(model, folder)
    (folder / "job.json").write_text(json.dumps(fb.job(model)), encoding="utf-8")
    assert bw.main([str(folder)]) == 0
    del hpfem
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def test_worker_empty_lattices_match_analytic_bands(tmp_path):
    for lattice in ("square", "hex"):
        m = empty(lattice, 1, 3)
        res = run_worker(m, tmp_path / lattice)
        assert all("f" in p for p in res["points"]), res["points"]
        got = fb.bands_matrix(res["points"], 3)
        ref = fb.empty_lattice_bands(m, 3)
        assert np.allclose(got, ref, atol=3e-3), (lattice, got, ref)
        assert got[0, 0] == pytest.approx(0.0, abs=1e-4) and got[0, 1] > 0.5               # the second zero at Gamma is dropped


def test_worker_fields_tables_and_figures(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = empty("square", 1, 3)
    m["field"] = "corners"
    res = run_worker(m, tmp_path)
    pts = res["points"]
    files = [p["modes_file"] for p in pts if p.get("modes_file")]
    assert len(files) == 4 and all((tmp_path / f).is_file() for f in files)
    npz = np.load(tmp_path / files[1])
    assert npz["et_0"].shape == (len(npz["points"]), 2) and "et_2" in npz and "et_3" not in npz
    for q in ("abs2", "ex", "ey"):
        plt.close(fb.fig_mode(npz, 0, q, "t", m))
    plt.close(fb.fig_bands(pts, res["x"], res["ticks"], gaps=True, empty_lattice=fb.empty_lattice_bands(m, 3)))
    plt.close(fb.fig_preview(fb.default_model()))
    plt.close(fb.fig_preview(fb.presets()["Dreiecksgitter: Luftlöcher r = 0,4 a in ε = 13"]))
    df = fb.bands_dataframe(pts, res["x"])
    assert list(df.columns)[:5] == ["Nr", "kx (2π/a)", "ky (2π/a)", "Pfad s (2π/a)", "f1"] and len(df) == 4
    assert list(fb.gaps_dataframe(pts).columns)[0] == "zwischen Band"


def test_worker_cancel_file(tmp_path):
    pytest.importorskip("hpfem")
    pytest.importorskip("gmsh")
    import fem_bands_worker as bw
    m = empty("square", 1, 3)
    fb.build_mesh(m, tmp_path)
    (tmp_path / "job.json").write_text(json.dumps(fb.job(m)), encoding="utf-8")
    (tmp_path / "cancel").write_text("")
    assert bw.main([str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res.get("cancelled") and res["points"] == []
