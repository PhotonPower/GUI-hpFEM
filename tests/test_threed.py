"""3D structures: model, job, checks, cut figures, worker (short runs on tiny meshes), app without exception."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_threed as ft  # noqa: E402


def small(task="scatter", shape="sphere", **kw):
    m = copy.deepcopy(ft.default_model())
    m.update(dict(task=task, shape=shape, n=1, order=2, half_width=2.0, outer=3.0), **kw)
    if task == "resonance":
        m["quarter"] = False
    return m


# ------------------------------------------------------------------------------------------------------------------ model and job
def test_presets_validate_and_jobs_are_json():
    names = ft.presets()
    assert len(names) >= 5 and {m["task"] for m in names.values()} == {"scatter", "resonance", "bands"}
    for name, m in names.items():
        assert not [t for lv, t in ft.validate(m) if lv == "error"], name
        j = json.loads(json.dumps(ft.job(m)))
        assert j["task"] == m["task"] and j["order"] == m["order"]
    j = ft.job(ft.presets()["Dielektrische Kugel n = 2, ka = 2 (Mie-Referenz)"])
    assert j["eps"] == [[4.0, 0.0]] and j["quarter"] is True and j["wavelengths_nm"] == [314.159]


def test_validation_errors():
    m = small()
    m["half_width"] = 1.75                               # 1.75 * n = 1.75 is not an integer
    assert any("ganzzahlig" in t for lv, t in ft.validate(m) if lv == "error")
    m = small()
    m["outer"] = 1.5
    assert any("Innenraum" in t for lv, t in ft.validate(m) if lv == "error")
    m = small("resonance")
    m["quarter"] = True
    assert any("ganze Modell" in t for lv, t in ft.validate(m) if lv == "error")
    m = small("bands", "sphere")
    m["bands"].update(radius=0.3)
    m["n"] = 2                                           # 0.5 / 0.3 * 2 = 3.33
    assert any("ganzzahlig" in t for lv, t in ft.validate(m) if lv == "error")
    m["n"] = 3
    assert not [t for lv, t in ft.validate(m) if lv == "error"]
    m = small("bands", "cube")
    m["bands"].update(radius=0.2, cells=8)               # 0.2 * 2 * 8 = 3.2
    assert any("Vielfaches" in t for lv, t in ft.validate(m) if lv == "error")


def test_eps_is_evaluated_per_wavelength():
    m = small()
    m["materials"][ft.OBJ] = {"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07}
    m["sweep"] = {"start": 400.0, "stop": 600.0, "count": 3}
    j = ft.job(m)
    assert len(j["eps"]) == 3 and j["eps"][0] != j["eps"][2] and all(e[1] > 0 for e in j["eps"])


# ------------------------------------------------------------------------------------------------------------------ figures
def test_section_polygons_of_one_tetrahedron():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    polys, idx = ft.section_polygons(v, np.array([[0, 1, 2, 3]]), 2, 0.5)
    assert idx == [0] and len(polys[0]) == 3
    p = polys[0]
    d1, d2 = p[1] - p[0], p[2] - p[0]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    assert area == pytest.approx(0.5 * 0.5 * 0.5)                  # the triangle x + y <= 1/2
    polys, _ = ft.section_polygons(v, np.array([[0, 1, 2, 3]]), 2, 2.0)
    assert polys == []


def test_mirror_and_empty_lattice():
    xs, ys, f = ft.mirror([0.0, 1.0, 2.0], [0.0, 1.0], np.arange(6.0).reshape(3, 2), True, True)
    assert f.shape == (5, 3) and np.allclose(xs, [-2, -1, 0, 1, 2]) and np.allclose(f, f[::-1]) and np.allclose(f, f[:, ::-1])
    e = ft.empty_lattice_bands([(0, 0, 0), (0.5, 0, 0)], 4)
    assert np.allclose(e[0], [0, 0, 1, 1])
    assert e[1][0] == e[1][1] == pytest.approx(0.5) and e[1][2] == e[1][3] == pytest.approx(0.5)


def test_dataframes_and_figures():
    pts = [{"index": i, "lam_nm": 300.0 + 10 * i, "eps": [4.0, 0.0], "size_parameter": 2.0, "sigma_sca": 1e-13, "sigma_abs": 0.0, "sigma_ext": 1e-13, "q_sca": 4.8, "q_abs": 0.0,
            "q_ext": 4.8, "mie_q_sca": 4.77, "mie_q_abs": 0.0, "mie_q_ext": 4.77, "seconds": 1.0} for i in range(3)]
    df = ft.scatter_dataframe(pts)
    assert "ΔQ_sca (%)" in df and df["ΔQ_sca (%)"].iloc[0] == pytest.approx(100 * 0.03 / 4.77)
    assert len(ft.modes_dataframe([{"index": 0, "lam_nm": 700.0, "omega": [1.0, -0.1], "Q": 5.0, "residual": 1e-10}])) == 1
    fig = ft.fig_efficiency(pts)
    assert fig.axes
    npz = {"xs": np.linspace(0, 1, 4), "zs": np.linspace(-1, 1, 7), "ys": np.linspace(0, 1, 4), "xz": np.ones((4, 7)), "xy": np.ones((4, 4))}
    assert ft.fig_field(npz, "xz", True, 1.0, "m", 0.5).axes and ft.fig_field(npz, "xy", True, 1.0, "m", 0.5, "cube").axes
    bp = [{"index": 0, "k": [0, 0, 0], "f": [0.0, 0.2]}, {"index": 1, "k": [0.5, 0, 0], "f": [0.1, 0.3]}]
    assert len(ft.bands_dataframe(bp, [0.0, 0.5])) == 2


# ------------------------------------------------------------------------------------------------------------------ worker
def test_meshes_tags_and_estimate(tmp_path):
    hpfem = pytest.importorskip("hpfem")
    import fem_threed_worker as tw
    m = small()
    mesh, info = tw.build_mesh(hpfem, ft.job(m))
    assert info["quarter"] and mesh.num_cells == 324 and np.all(np.asarray(mesh.vertices)[:, 0] >= -1e-15)
    sym = [f for f in mesh.boundary_facets if mesh.facet_tag(int(f)) == tw.TAG_SYM]
    assert sym, "symmetry cut x = 0 not tagged"
    a = info["a"]
    vol = np.asarray(mesh.cell_volumes)[np.asarray(mesh.cell_tags) == tw.TAG_OBJ].sum()
    assert vol == pytest.approx(np.pi * a**3 / 3, rel=0.15)             # quarter ball (n = 1 is coarse)
    cube, info = tw.build_mesh(hpfem, ft.job(small(shape="cube", quarter=False)))
    vol = np.asarray(cube.cell_volumes)[np.asarray(cube.cell_tags) == tw.TAG_OBJ].sum()
    assert vol == pytest.approx((2 * info["a"]) ** 3, rel=1e-9)         # the cube of half edge a is made of whole cells
    est = ft.run_estimate(sys.executable, HERE.parent, tmp_path / "e", ft.job(small()))
    assert est["cells"] == 324 and est["dofs"] > 1000 and est["bytes"] > 1e6 and (tmp_path / "e" / "mesh_scatter.npz").is_file()
    assert ft.time_hint(18264) == pytest.approx(70.0) and ft.time_hint(36528) > 200


def run_worker(folder, m):
    import fem_threed_worker as tw
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(ft.job(m)), encoding="utf-8")
    code = tw.main([str(folder)])
    return code, json.loads((folder / "results.json").read_text(encoding="utf-8")) if (folder / "results.json").is_file() else None


def test_sphere_scattering_against_mie_on_a_coarse_mesh(tmp_path):
    pytest.importorskip("hpfem")
    m = small(order=2)
    code, res = run_worker(tmp_path / "threed", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p and p["size_parameter"] == pytest.approx(2.0, rel=1e-4)
    assert p["mie_q_sca"] == pytest.approx(4.7705, rel=1e-3)
    assert abs(p["q_sca"] - p["mie_q_sca"]) / p["mie_q_sca"] < 0.10                   # coarse mesh n = 1, p = 2 (about 5 %)
    assert p["q_ext"] == pytest.approx(p["q_sca"] + p["q_abs"], rel=1e-6)
    f = np.load(tmp_path / "threed" / p["field_file"])
    assert f["xz"].shape == (41, 81) and f["xy"].shape == (41, 41) and np.isfinite(f["xz"]).all()
    assert f["xz"].max() > 1.0                                                          # field enhancement inside the sphere (n = 2)


def test_memory_limit_stops_the_run(tmp_path):
    pytest.importorskip("hpfem")
    m = small()
    m["max_gb"] = 1e-4
    code, res = run_worker(tmp_path / "t", m)
    assert code == 1 and res is None                       # stops before the first solve, no result file


def test_cube_scattering_runs_and_is_positive(tmp_path):
    pytest.importorskip("hpfem")
    m = small(shape="cube", order=1)
    code, res = run_worker(tmp_path / "c", m)
    assert code == 0 and "mie_q_sca" not in res["points"][0] and res["points"][0]["q_sca"] > 0


# ------------------------------------------------------------------------------------------------------------------ app
def test_app_threed_mode_without_exception(tmp_path):
    pytest.importorskip("hpfem")
    from streamlit.testing.v1 import AppTest

    script = str(HERE / "fem_app.py")
    run_worker(tmp_path / "threed", small())
    at = AppTest.from_file(script, default_timeout=120)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.dataframe or any("Mie" in str(c.value) for c in at.caption)
    for name in ft.presets():
        at.selectbox(key="td_preset").select(name).run()
        [b for b in at.button if b.key == "td_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])


def test_cubic_path_and_bands_job():
    import fem_threed_worker as tw
    ks, x, ticks = tw.cubic_path(2)
    assert len(ks) == len(x) == 9 and [lab for _, lab in ticks] == ["Γ", "X", "M", "R", "Γ"]
    assert ks[2] == (0.5, 0.0, 0.0) and ks[-1] == (0.0, 0.0, 0.0)
    assert [t for t, _ in ticks] == pytest.approx([0.0, 0.5, 1.0, 1.0 + 0.5, 1.5 + np.sqrt(0.75)])      # |GX| = |XM| = |MR| = 1/2, |RG| = sqrt(3)/2
    j = ft.job(small("bands", "cube"))
    assert j["task"] == "bands" and j["cells"] == 4 and j["eps"] == 12.0 and j["half_width"] == 0.5


def test_app_renders_resonance_and_bands_results(tmp_path):
    from streamlit.testing.v1 import AppTest

    folder = tmp_path / "threed"
    folder.mkdir()
    g = np.linspace(-1e-7, 1e-7, 41)
    np.savez(folder / "field_0.npz", xs=g, ys=g, zs=g, xz=np.random.rand(41, 41), xy=np.random.rand(41, 41))
    meta = dict(name="t", task="resonance", shape="sphere", cells=1, order=2, dofs=2, human="1 MB", hpfem="0.5.0", started="x", quarter=False)
    res = dict(meta=meta, points=[], seconds=1.0, modes=[dict(index=0, omega=[2.6e15, -1e14], lam_nm=715.0, Q=13.0, residual=1e-12, field_file="field_0.npz")])
    (folder / "results.json").write_text(json.dumps(res), encoding="utf-8")
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=120)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.dataframe
    meta.update(task="bands", shape="cube")
    pts = [dict(index=i, k=[0.1 * i, 0, 0], f=[0.1 * i, 0.1 * i, 0.5 + i * 0.01], residual=1e-12, seconds=1.0) for i in range(4)]
    res = dict(meta=meta, points=pts, modes=[], x=[0.0, 0.1, 0.2, 0.3], ticks=[[0.0, "Γ"], [0.3, "X"]])
    (folder / "results.json").write_text(json.dumps(res), encoding="utf-8")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
