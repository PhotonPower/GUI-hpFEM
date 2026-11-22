"""3D extensions: layer stack, Gmsh geometry and file import, hp-adaptive loop (model, job, worker on tiny meshes, app)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_threed as ft  # noqa: E402
import fem_threed_gmsh as gm  # noqa: E402


def gmsh_model(kind="sphere", stack=False, **kw):
    """A deliberately coarse Gmsh model (p = 1, a few hundred cells to a few thousand): checks the mechanics, not the accuracy."""
    m = copy.deepcopy(ft.default_model())
    m.update(task="scatter", mesh_kind="gmsh", order=1, quarter=False, radius_nm=100.0, sweep={"start": 628.3, "stop": 628.3, "count": 1})
    m["gmsh"].update(kind=kind, hw_nm=200.0, pml_nm=300.0, clear_nm=100.0, h_obj_nm=80.0, h_far_nm=250.0, grade_nm=200.0, height_nm=120.0)
    if stack:
        m["stack"].update(enabled=True, layers=[], substrate=ft.AIR, gap_nm=100.0, depth_nm=100.0, angle_deg=0.0, pol="p")
    m.update(kw)
    return m


def errors(m):
    return [t for lv, t in ft.validate(m) if lv == "error"]


# ------------------------------------------------------------------------------------------------------------------ model and job
def test_layout_of_the_vertical_regions():
    g = dict(kind="sphere", r=100.0, stack=True, layers=[50.0, 30.0], depth=100.0, gap=40.0, clear=80.0)
    lay = gm.layout(g)
    assert lay["interfaces"] == [0.0, -50.0, -80.0] and lay["z_lo"] == -180.0
    assert lay["obj_z0"] == 40.0 and lay["zc"] == 140.0 and lay["z_hi"] == 40.0 + 200.0 + 80.0
    assert lay["regions"] == [(0.0, 320.0), (-50.0, 0.0), (-80.0, -50.0), (-180.0, -80.0)]
    lay = gm.layout(dict(kind="block", lx=100.0, ly=80.0, lz=60.0, clear=50.0, stack=False))
    assert lay["zc"] == 0.0 and lay["z_lo"] == -80.0 and lay["z_hi"] == 80.0 and gm.object_extent(dict(kind="block", lx=100.0, ly=80.0, lz=60.0)) == (50.0, 40.0, 60.0)


def test_validation_of_gmsh_stack_adaptive_and_file():
    m = gmsh_model()
    assert not errors(m)
    m["gmsh"]["hw_nm"] = 100.0                                       # not larger than the object (half width 100 nm)
    assert any("Innenraum" in t for t in errors(m))
    m = gmsh_model(stack=True)
    m["stack"]["angle_deg"] = 95.0
    assert any("Einfallswinkel" in t for t in errors(m))
    m = gmsh_model(stack=True)
    m["stack"]["layers"] = [{"material": "gibt es nicht", "thickness_nm": 10.0}]
    assert any("fehlt" in t for t in errors(m))
    m = gmsh_model(stack=True, task="resonance")
    assert any("Schichtstapel" in t for lv, t in ft.validate(m) if lv == "warning") and "stack" not in ft.job(m) and ft.job(m)["g"]["stack"] is False
    m = copy.deepcopy(ft.default_model())
    m["task"] = "resonance"
    m["adaptive"]["enabled"] = True
    assert any("Adaptivität" in t for t in errors(m))
    m = gmsh_model(mesh_kind="file")
    assert any("hochladen" in t for t in errors(m))
    m["file"].update(name="a.msh", path="a.msh", tags={"cells": {"11": 5, "2": 3}, "facets": {}, "bbox": [[-5e-7] * 3, [5e-7] * 3]}, object_tags=[2], tag_material={"11": ft.AIR})
    assert any("ohne Material" in t for t in errors(m))
    m["file"]["tag_material"]["2"] = ft.OBJ
    assert not errors(m)


def test_jobs_carry_the_new_blocks_and_are_json():
    m = gmsh_model("cylinder", stack=True)
    m["stack"].update(layers=[{"material": ft.GLASS, "thickness_nm": 40.0}], substrate=ft.GLASS, angle_deg=20.0, pol="s")
    j = json.loads(json.dumps(ft.job(m)))
    assert j["mesh_kind"] == "gmsh" and j["g"]["kind"] == "cylinder" and j["g"]["stack"] is True and j["g"]["layers"] == [40.0]
    assert j["stack"]["pol"] == "s" and len(j["stack"]["eps"]) == 3 and j["stack"]["eps"][1][0] == [2.25, 0.0] and j["g"]["region_n"] == pytest.approx([1.0, 1.5, 1.5])
    assert j["quarter"] is False
    m = copy.deepcopy(ft.default_model())
    m["adaptive"].update(enabled=True, max_steps=3)
    assert ft.job(m)["adaptive"]["max_steps"] == 3 and "adaptive" not in ft.job(copy.deepcopy(ft.default_model()))
    s = ft.suggest_sizes(gmsh_model())
    assert s["h_far_nm"] == pytest.approx(628.3 / 2.5, abs=0.1) and s["pml_nm"] == pytest.approx(314.2, abs=0.1) and s["h_obj_nm"] <= 40.0


def test_figures_for_steps_outline_and_mesh_colours():
    steps = [dict(step=i, cells=100 + i, dofs=1000 * (i + 1), eta=1.0 / (i + 1), orders_max=2 + i, orders_min=2, seconds=1.0, q_sca=4.0 + 0.1 * i, q_abs=0.0, mie_q_sca=4.5) for i in range(3)]
    df = ft.steps_dataframe(steps)
    assert len(df) == 3 and "ΔQ_sca (%)" in df
    assert ft.fig_convergence(steps).axes
    m = gmsh_model("sphere", stack=True)
    xz = ft.object_outline(m, "xz")
    assert len(xz) == 2 and xz[0][1].max() == pytest.approx(300.0, abs=0.01)           # circle centred at z = 100 + 100, radius 100: top at 300 nm
    assert len(ft.object_outline(gmsh_model("block"), "xy")) == 1 and ft.object_outline(gmsh_model(mesh_kind="file"), "xz") == []
    fig = ft.fig_mesh({"vertices": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), "cells": np.array([[0, 1, 2, 3]]), "tags": np.array([11])},
                      2, 0.2, 1.0, "m", obj_tags=(2,))
    assert fig.axes


# ------------------------------------------------------------------------------------------------------------------ worker
@pytest.fixture(scope="module")
def gmsh_available():
    pytest.importorskip("gmsh")
    pytest.importorskip("hpfem")


def run_worker(folder, m, mode=None):
    import fem_threed_worker as tw
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(ft.job(m)), encoding="utf-8")
    code = tw.main([str(folder)] + ([mode] if mode else []))
    return code, json.loads((folder / "results.json").read_text(encoding="utf-8")) if (folder / "results.json").is_file() else None


def test_gmsh_scene_tags_and_regions(tmp_path, gmsh_available):
    import hpfem
    g = ft.gmsh_params(gmsh_model("cylinder", stack=True))
    g.update(layers=[40.0], region_n=[1.0, 1.5, 1.5])
    info = gm.scene(g, tmp_path / "s.msh")
    assert info["regions"] == [11, 12, 13] and info["object_tag"] == 2
    mesh = hpfem.read_gmsh(str(tmp_path / "s.msh"), 1e-9, 3)
    tags = np.asarray(mesh.cell_tags)
    assert set(np.unique(tags)) == {2, 11, 12, 13}
    vol = np.asarray(mesh.cell_volumes)
    assert vol[tags == 2].sum() == pytest.approx(np.pi * (100e-9) ** 2 * 120e-9, rel=0.05)         # cylinder r = 100 nm, h = 120 nm (second-order geometry)
    zc = np.asarray(mesh.cell_centroids)[:, 2]
    assert zc[tags == 12].max() <= 1e-9 and zc[tags == 12].min() >= -40e-9 - 1e-12                 # the layer lies between its interfaces z = 0 and z = -40 nm
    assert all(len(mesh.facets_with_tag(t)) > 0 for t in range(1, 7))                              # the six box sides carry the box tags


def test_gmsh_sphere_scatter_and_file_import_agree(tmp_path, gmsh_available):
    m = gmsh_model("sphere")
    code, res = run_worker(tmp_path / "g", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p and p["q_sca"] > 0 and p["mie_q_sca"] == pytest.approx(0.7969, rel=2e-3)
    assert p["q_ext"] == pytest.approx(p["q_sca"] + p["q_abs"], rel=1e-6)
    f = np.load(tmp_path / "g" / p["field_file"])
    assert f["xz"].shape == (41, 81) and np.isfinite(f["xz"]).mean() > 0.95
    # the same mesh read as a user file: the numbers are identical
    mf = copy.deepcopy(m)
    mf["mesh_kind"] = "file"
    mf["file"].update(name="scene.msh", path=str(tmp_path / "g" / "scene.msh"), scale="nm", pml_nm=300.0, object_tags=[2], tag_material={"11": ft.AIR, "2": ft.OBJ})
    (tmp_path / "i").mkdir()
    info = ft.run_inspect(sys.executable, HERE.parent, tmp_path / "i", mf["file"]["path"], "nm")
    assert info["cells"].keys() == {"2", "11"} and set(info["facets"]) == {"1", "2", "3", "4", "5", "6"} and info["n_cells"] > 100
    assert info["bbox"][1][0] == pytest.approx(500e-9, rel=1e-6)
    mf["file"]["tags"] = info
    assert not errors(mf)
    code, res2 = run_worker(tmp_path / "f", mf)
    assert code == 0
    q = res2["points"][0]
    assert q["sigma_sca"] == pytest.approx(p["sigma_sca"], rel=1e-6) and q["sigma_abs"] == pytest.approx(p["sigma_abs"], abs=1e-6 * p["sigma_sca"])
    assert q["q_sca"] == pytest.approx(p["q_sca"], rel=0.30)                       # Q of a file: cross section over the projected area of the (straight-facet) surface


def test_layer_stack_background_gives_fresnel_and_a_finite_particle_response(tmp_path, gmsh_available):
    m = gmsh_model("cylinder", stack=True)
    m["gmsh"].update(height_nm=100.0, hw_nm=250.0)
    m["stack"].update(substrate=ft.GLASS, gap_nm=0.0, depth_nm=150.0, angle_deg=30.0, pol="p")
    m["materials"][ft.OBJ] = {"type": "index", "n": 2.0, "k": 0.0}
    code, res = run_worker(tmp_path / "s", m)
    assert code == 0
    p = res["points"][0]
    assert "error" not in p
    rp = (1.5 * np.cos(np.deg2rad(30)) - np.sqrt(1 - (0.5 / 1.5) ** 2)) / (1.5 * np.cos(np.deg2rad(30)) + np.sqrt(1 - (0.5 / 1.5) ** 2))
    assert p["stack_R"] == pytest.approx(rp**2, rel=1e-6) and p["stack_T"] == pytest.approx(1 - rp**2, rel=1e-6) and p["stack_A"] == pytest.approx(0.0, abs=1e-9)
    assert p["q_sca"] > 0.1 and abs(p["q_abs"]) < 0.25 * p["q_sca"] and p["q_ext"] == pytest.approx(p["q_sca"] + p["q_abs"], rel=1e-6)
    df = ft.scatter_dataframe(res["points"])
    assert {"R (Stapel)", "T (Stapel)", "A (Stapel)"} <= set(df.columns)


def test_hp_adaptive_loop_records_steps_and_orders(tmp_path, gmsh_available):
    m = copy.deepcopy(ft.default_model())
    m.update(task="scatter", n=1, order=1, quarter=True)
    m["adaptive"].update(enabled=True, max_steps=3, max_dofs=20000, theta=0.5)
    code, res = run_worker(tmp_path / "a", m)
    assert code == 0
    p = res["points"][0]
    steps = p["steps"]
    assert len(steps) == 3 and [s["step"] for s in steps] == [0, 1, 2] and p["stop_reason"] == "Schrittzahl erreicht"
    assert steps[0]["dofs"] < steps[-1]["dofs"] and steps[-1]["eta"] < steps[0]["eta"]            # the estimator falls with the refinement
    assert steps[0]["h_marked"] + steps[0]["p_marked"] > 0 and p["dofs_final"] == steps[-1]["dofs"] and "mie_q_sca" in p
    npz = np.load(tmp_path / "a" / "mesh_scatter.npz")
    assert "orders" in npz.files and len(npz["orders"]) == len(npz["cells"]) and npz["orders"].max() >= 1
    # the unknown limit stops the loop before the next solve
    m["adaptive"]["max_dofs"] = steps[0]["dofs"] + 1
    code, res = run_worker(tmp_path / "b", m)
    assert code == 0 and len(res["points"][0]["steps"]) == 1 and "Grenze" in res["points"][0]["stop_reason"]


def test_hp_adaptive_loop_on_a_curved_gmsh_mesh(tmp_path, gmsh_available):
    m = gmsh_model("sphere")
    m["adaptive"].update(enabled=True, max_steps=2, max_dofs=30000, theta=0.5)
    code, res = run_worker(tmp_path / "ag", m)
    assert code == 0
    steps = res["points"][0]["steps"]
    assert len(steps) == 2 and steps[1]["dofs"] > steps[0]["dofs"] and steps[1]["eta"] < steps[0]["eta"] and "mie_q_sca" in res["points"][0]


# ------------------------------------------------------------------------------------------------------------------ app
def test_app_new_controls_and_results_without_exception(tmp_path, gmsh_available):
    from streamlit.testing.v1 import AppTest

    m = copy.deepcopy(ft.default_model())
    m.update(task="scatter", n=1, order=1, quarter=True)
    m["adaptive"].update(enabled=True, max_steps=2, max_dofs=20000)
    run_worker(tmp_path / "threed", m)
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=180)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("hp-Verfeinerung" in t.label for t in at.tabs) or at.dataframe
    for name in ("Silizium-Zylinder auf Glas, schräger Einfall (p, 30°; grob, p = 1)", "Gold-Kugel auf SiO2-Schicht über Glas (Stapel, grob)"):
        at.selectbox(key="td_preset").select(name).run()
        [b for b in at.button if b.key == "td_preset_load"][0].click().run()
        assert not at.exception, (name, [e.value for e in at.exception])
        assert at.checkbox(key=f"td_st_on_{at.session_state['td_ver']}").value is True
    ver = at.session_state["td_ver"]
    at.radio(key=f"td_mk_{ver}").set_value("file").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("hochladen" in e.value for e in at.error)
    at.radio(key=f"td_task_{ver}").set_value("ports").run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_file_editors_render_with_read_tags(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=180)
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    m = copy.deepcopy(ft.default_model())
    m.update(task="scatter", mesh_kind="file")
    tags = dict(n_cells=10, cells={"11": 7, "2": 3}, facets={"1": 4, "2": 4}, bbox=[[-5e-7] * 3, [5e-7] * 3], geometry_order=2)
    m["file"].update(name="a.msh", path="a.msh", tags=tags, tag_material={"11": ft.AIR, "2": ft.OBJ}, object_tags=[2])
    for key, val in (("td_model", m), ("td_ver", 0), ("td_proc", None), ("td_dir", None), ("td_msg", None), ("td_est", None)):
        at.session_state[key] = val
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="td_ftm_0_11").value == ft.AIR and at.checkbox(key="td_fto_0_2").value is True
    assert not [e for e in at.error if "hochladen" in e.value or "ohne Material" in e.value]
    # the same for the ports task with a file device
    p = copy.deepcopy(ft.default_model())
    p["task"] = "ports"
    p["p3"].update(device="file")
    p["p3"]["file"].update(name="g.msh", path="g.msh", tags=dict(n_cells=10, cells={"1": 7, "2": 3}, facets={"10": 4, "11": 2, "12": 2}, bbox=[[-5e-7] * 3, [5e-7] * 3], geometry_order=1),
                           tag_material={"1": "Mantel", "2": "Kern"}, pec_tags=[10], ports=[dict(tag=11, modes=1), dict(tag=12, modes=1)])
    at = AppTest.from_file(str(HERE / "fem_app.py"), default_timeout=180)            # new session: the task radio of the first one keeps its widget state
    at.session_state["app_mode"] = "threed"
    at.session_state["work_dir"] = str(tmp_path)
    for key, val in (("td_model", p), ("td_ver", 0), ("td_proc", None), ("td_dir", None), ("td_msg", None), ("td_est", None)):
        at.session_state[key] = val
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="p3_fr_0_11").value == "port" and at.selectbox(key="p3_fr_0_10").value == "pec"
