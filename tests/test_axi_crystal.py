"""Kristalle im Rotationskörper-Modus: Vorlagen, Achsenprüfung (r, φ, z), Modellprüfung, Worker-Material und ein Rechentest (mit hpfem)."""
import copy
import json

import numpy as np
import pytest

import fem_axi as fa
import fem_materials as fm

SPHERE = "Dielektrische Kugel n = 2,5 (Mie-Resonanzen, Emitter außen)"


def uni(n_o, n_e, theta=0.0, phi=0.0):
    return {"type": "crystal", "kind": "uniaxial", "n_o": n_o, "k_o": 0.0, "n_e": n_e, "k_e": 0.0, "theta_deg": theta, "phi_deg": phi}


def sphere_model(material):
    m = copy.deepcopy(fa.presets()[SPHERE])
    m["materials"]["Kugel"] = material
    return m


def errors(m, task="resonance"):
    return [t for lvl, t in fa.validate(m, task) if lvl == "error"]


# ----------------------------------------------------------------------------------------------------------- templates
def test_crystal_presets_are_uniaxial_specs():
    assert len(fm.CRYSTAL_PRESETS) >= 5
    for name, v in fm.CRYSTAL_PRESETS.items():
        spec = fm.crystal_preset(name)
        json.dumps(spec)
        assert fm.is_crystal(spec) and spec["theta_deg"] == 0.0
        t = fm.tensor_at(spec, v["lam_nm"])
        assert np.allclose(np.diag(t).real, [v["n_o"] ** 2, v["n_o"] ** 2, v["n_e"] ** 2])
        assert "589" in name or "633" in name           # the wavelength of the constant indices is part of the label
        assert fm.describe(spec).startswith("einachsig")


def test_sapphire_template_matches_hpfem_sellmeier():
    hp = pytest.importorskip("hpfem")
    v = fm.CRYSTAL_PRESETS["Saphir Al2O3 (c-Schnitt, 589 nm)"]
    omega = 2 * np.pi * hp.constants.c0 / (v["lam_nm"] * 1e-9)
    t = hp.materials.get("Al2O3").eps_tensor(omega)
    assert np.sqrt(t[0, 0].real) == pytest.approx(v["n_o"], abs=2e-4)
    assert np.sqrt(t[2, 2].real) == pytest.approx(v["n_e"], abs=2e-4)


# --------------------------------------------------------------------------------------------- axis in (r, phi, z)
def test_axi_diagonal_allowed_orientations():
    e_o, e_e = 2.4**2, 2.7**2
    assert np.allclose(fm.axi_diagonal(uni(2.4, 2.7, 0.0)), [e_o, e_o, e_e])                    # axis along z
    assert np.allclose(fm.axi_diagonal(uni(2.4, 2.7, 90.0, 0.0)), [e_e, e_o, e_o])              # radial
    assert np.allclose(fm.axi_diagonal(uni(2.4, 2.7, 90.0, 90.0)), [e_o, e_e, e_o])             # azimuthal
    bi = {"type": "crystal", "kind": "biaxial", "n": [1.5, 1.6, 1.7], "k": [0, 0, 0], "euler_deg": [0, 0, 0]}
    assert np.allclose(fm.axi_diagonal(bi), [1.5**2, 1.6**2, 1.7**2])
    diag = {"type": "crystal", "kind": "tensor", "re": [[2, 0, 0], [0, 3, 0], [0, 0, 4]], "im": np.zeros((3, 3)).tolist()}
    assert np.allclose(fm.axi_diagonal(diag), [2, 3, 4])


@pytest.mark.parametrize("spec", [uni(2.4, 2.7, 45.0), uni(2.4, 2.7, 60.0, 30.0)])
def test_axi_diagonal_refuses_tilted_axis(spec):
    with pytest.raises(ValueError, match="nicht diagonal"):
        fm.axi_diagonal(spec)


def test_tilted_axis_is_isotropic_limit_when_n_o_equals_n_e():
    assert np.allclose(fm.axi_diagonal(uni(2.5, 2.5, 45.0)), 2.5**2)


# ----------------------------------------------------------------------------------------------------- model checks
def test_validate_crystal_part_ok_and_tilted_refused():
    assert not errors(sphere_model(uni(2.4, 2.7)))
    assert not errors(sphere_model(uni(2.4, 2.7, 90.0, 0.0)), "scattering")
    bad = errors(sphere_model(uni(2.4, 2.7, 45.0)))
    assert bad and "Kugel" in bad[0] and "verkippt" in bad[0]


def test_validate_crystal_background_substrate_layers_and_host():
    m = sphere_model(uni(2.4, 2.7))
    m["materials"]["Luft"] = uni(1.0, 1.1)
    assert any("Umgebungsmaterial" in t and "isotrop" in t for t in errors(m))
    m = sphere_model(uni(2.4, 2.7))
    m["materials"]["Sub"] = uni(1.5, 1.6)
    m["substrate"] = "Sub"
    assert not errors(m, "resonance")                                       # a crystal substrate is a mesh region: resonances take it
    assert any("Substrat" in t for t in errors(m, "scattering"))
    assert any("Substrat" in t for t in errors(m, "emitter"))
    m = sphere_model(uni(2.4, 2.7))
    m["emitter"]["z_nm"] = 0.0                                              # the dipole inside the crystal sphere
    assert any("Wirt" in t for t in errors(m, "emitter"))


def test_offaxis_with_crystal_body_is_reported_clearly():
    m = sphere_model(uni(2.4, 2.7))
    m["emitter"].update(z_nm=450.0, sigma_nm=20.0)
    assert not errors(m, "emitter")                                         # on the axis: the built-in path takes crystals
    m["emitter"]["radii_nm"] = [0.0, 200.0]
    bad = errors(m, "emitter")                                              # off the axis: hpfem.dataports takes isotropic materials only
    assert len(bad) == 1 and "außerhalb der Achse" in bad[0] and "Kugel" in bad[0]
    m["materials"]["Kugel"] = {"type": "index", "n": 2.5, "k": 0.0}
    assert not errors(m, "emitter")


# ------------------------------------------------------------------------------------------------- worker (hpfem)
def build_job(tmp_path, material, hp, lam=700.0):
    pytest.importorskip("gmsh")
    import fem_axi_worker as fw

    m = sphere_model(material)
    m["resonance"].update(wavelength_nm=lam, m=1, num_modes=2)
    ms = dict(cells_per_wavelength=2.5, skin_cells=1.0, interface_factor=0.7, curved=True, pml_cells_per_wavelength=4.0, lam_min_nm=600.0, lam_max_nm=800.0)
    fa.build_mesh(m, ms, tmp_path, "resonance")
    job = dict(model=m, task="resonance", solver=dict(order=2, backend="AUTO"))
    return fw.Job(hp, job, tmp_path), fw


def test_worker_material_tensor_in_r_phi_z(tmp_path):
    hp = pytest.importorskip("hpfem")
    job, _ = build_job(tmp_path, uni(2.4, 2.7), hp)
    mat = job.material("Kugel", 700.0)
    assert np.allclose(np.diag(mat.eps_tensor), [2.4**2, 2.4**2, 2.7**2])
    assert np.allclose(job.material("Luft", 700.0).eps_tensor, np.eye(3) * 1.000293**2)
    mm, eps = job.materials(700.0)
    assert eps["Kugel"].real == pytest.approx(2.7**2)                       # scalar summary for mesh sizes: the largest principal value


def test_worker_resonance_isotropic_tensor_equals_scalar_and_anisotropy_shifts(tmp_path):
    hp = pytest.importorskip("hpfem")
    res = {}
    for key, mat in (("iso", {"type": "index", "n": 2.5, "k": 0.0}), ("tensor", uni(2.5, 2.5)), ("aniso", uni(2.4, 2.7))):
        d = tmp_path / key
        d.mkdir()
        job, fw = build_job(d, mat, hp)
        modes, _, _ = fw.resonances(job, 700.0, 1, 2)
        res[key] = np.array([md["lam_nm"] for md in modes])
    assert np.allclose(res["tensor"], res["iso"], rtol=1e-6)
    assert np.abs(res["aniso"][0] - res["iso"][0]) > 5.0                    # the birefringence moves the resonance by tens of nm
