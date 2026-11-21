"""Drude-Lorentz-Material, Fit und dispersive Modelle für hpfem (kurze Rechnungen, kein Solver)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "fem_gui"
sys.path.insert(0, str(HERE))

import fem_materials as fm  # noqa: E402
import fem_run as fr  # noqa: E402

DL = {"type": "drude_lorentz", "eps_inf": 2.0, "omega_p_eV": 8.0, "gamma_eV": 0.1, "osc": [[0.4, 3.2, 0.5], [0.1, 5.0, 0.8]]}


def test_drude_lorentz_reduces_to_drude_without_oscillators():
    d = {"type": "drude", "eps_inf": 1.5, "omega_p_eV": 9.0, "gamma_eV": 0.07}
    dl = {"type": "drude_lorentz", "eps_inf": 1.5, "omega_p_eV": 9.0, "gamma_eV": 0.07, "osc": []}
    for lam in (400.0, 633.0, 1550.0):
        assert fm.eps_at(dl, lam) == pytest.approx(fm.eps_at(d, lam))
    assert fm.eps_at(DL, 600.0).imag > 0                                   # loss: Im eps > 0
    assert fm.dl_eps(DL, 2.0 + 0.1j) != fm.dl_eps(DL, 2.0)                  # complex energies (analytic continuation)
    assert "Drude-Lorentz" in fm.describe(DL)


def test_is_dispersive():
    assert fm.is_dispersive(DL) and fm.is_dispersive({"type": "library", "name": "Ag"}) and fm.is_dispersive({"type": "table", "rows": [[1, 1, 0], [2, 1, 0]]})
    assert not fm.is_dispersive({"type": "index", "n": 1.5, "k": 0.0}) and not fm.is_dispersive({"type": "library", "name": "air"})


def test_matches_hpfem_model():
    hm = pytest.importorskip("hpfem.materials")
    import fem_disp
    model = fem_disp.dispersive_model(DL)
    for lam in (450.0, 700.0, 1200.0):
        omega = float(2 * np.pi * 299792458.0 / (lam * 1e-9))
        assert complex(model.eps_r(omega)) == pytest.approx(fm.eps_at(DL, lam), rel=1e-6)
    assert isinstance(fem_disp.dispersive_model({"type": "library", "name": "Ag"}), hm.Dispersive)
    tab = fem_disp.dispersive_model({"type": "table", "rows": [[400, 1.5, 0.0], [800, 1.7, 0.1]]})
    assert complex(tab.eps_r(float(2 * np.pi * 299792458.0 / 600e-9))) == pytest.approx(fm.eps_at({"type": "table", "rows": [[400, 1.5, 0.0], [800, 1.7, 0.1]]}, 600.0))
    with pytest.raises(ValueError):
        fem_disp.dispersive_model({"type": "crystal", "kind": "uniaxial"})


def test_fit_silver_and_errors():
    pytest.importorskip("hpfem.materials")
    r = fr.fit_material(sys.executable, None, {"type": "library", "name": "Ag"}, 450.0, 800.0, 2, count=30)
    assert r["spec"]["type"] == "drude_lorentz" and len(r["spec"]["osc"]) == 2
    assert 0 < r["max_rel_error"] < 0.1
    assert len(r["lam_nm"]) == len(r["eps_fit_re"]) == len(r["eps_ref_im"]) == 30
    for lam, e_fit in zip(r["lam_nm"][::10], r["eps_fit_re"][::10]):
        assert e_fit == pytest.approx(fm.eps_at(r["spec"], lam).real)       # the saved spec reproduces the fit curve
    assert fm.eps_at(r["spec"], 600.0).imag > 0
    with pytest.raises(ValueError, match="Konstante|konstante"):
        fr.fit_material(sys.executable, None, {"type": "index", "n": 1.5, "k": 0.0}, 450.0, 800.0, 1)
    with pytest.raises(ValueError, match="außerhalb"):
        fr.fit_material(sys.executable, None, {"type": "library", "name": "Ag"}, 100.0, 800.0, 1)


def test_resonance_point_selfconsistent_wiring(monkeypatch):
    """Worker: dispersive models go to grating.resonances, every mode is refined; a failing refinement keeps the frozen mode (no solver run)."""
    pytest.importorskip("hpfem.materials")
    from types import SimpleNamespace as NS

    import hpfem.grating as grating

    import fem_worker as fw

    seen = {}
    modes = [NS(index=i, omega=complex(3.0e15 + i * 1e14, -1e13), wavelength=600e-9 - i * 1e-8, Q=150.0, residual=1e-12, beta=0.0, field=None) for i in range(2)]
    res = NS(modes=modes, dofs=10, timing={"total": 0.1})

    def fake_resonances(mesh, mats, stack, omega, **kw):
        seen["mats"] = mats
        return res

    def fake_refine(r, mode=0, **kw):
        if mode == 1:
            raise grating.GratingError("no convergence")
        return NS(modes=[NS(index=0, omega=3.01e15 - 2e13j, wavelength=598e-9, Q=75.0, residual=1e-13, beta=0.0)], iterations=4)

    case = NS(h=NS(), model={"materials": {"Luft": {"type": "library", "name": "air"}, "Metall": DL}}, names=["Luft", "Metall"], mesh=None, P=4e-7,
              y_phys_bottom=0.0, y_phys_top=1e-6)
    ctx = NS(omega=3e15, stack=None, mat={"Luft": 1, "Metall": 2}, eps={"Luft": 1 + 0j, "Metall": -5 + 1j}, lam_nm=600.0)
    monkeypatch.setattr(fw, "prepare", lambda c, lam: ctx)
    monkeypatch.setattr(fw, "bloch_wavenumbers", lambda c, t, p: (1.0e6, 0.0))
    monkeypatch.setattr(fw, "grating_options", lambda c, x, s: dict(pml=None, bottom="pml", pml_target=1e-6))
    monkeypatch.setattr(fw, "backend_enum", lambda h, s: None)
    monkeypatch.setattr(grating, "resonances", fake_resonances)
    monkeypatch.setattr(grating, "refine_resonance", fake_refine)
    out, _ = fw.resonance_point(case, 600.0, 0.0, 0.0, {"order": 2}, {"num_modes": 2, "selfconsistent": True})
    import hpfem.materials as hm
    assert isinstance(seen["mats"][2], hm.DrudeLorentz) and isinstance(seen["mats"][1], hm.Constant)
    m0, m1 = out["modes"]
    assert m0["selfconsistent"] is True and m0["lam_nm"] == pytest.approx(598.0) and m0["lam_frozen_nm"] == pytest.approx(600.0) and m0["iterations"] == 4
    assert m1["selfconsistent"] is False and "no convergence" in m1["refine_error"] and m1["lam_nm"] == pytest.approx(590.0)
    import fem_post as fp
    df = fp.resonances_dataframe([dict(index=0, theta=0.0, phi=0.0, kx_over_G=0.0, dofs=10, time_s=0.1, modes=out["modes"])])
    assert "λ_eingefroren (nm)" in df.columns
