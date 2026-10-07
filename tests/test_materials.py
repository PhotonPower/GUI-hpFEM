import numpy as np
import pytest

import fem_materials as fm


def test_constant_types():
    assert fm.eps_at({"type": "eps", "re": 2.25, "im": 0.1}, 500) == 2.25 + 0.1j
    assert fm.eps_at({"type": "index", "n": 1.5, "k": 0.0}, 500) == pytest.approx(2.25)
    assert fm.eps_at({"type": "library", "name": "vacuum"}, 500) == 1.0


def test_sellmeier_sio2():
    n, k = fm.nk({"type": "library", "name": "SiO2"}, 589.3)
    assert n == pytest.approx(1.4585, abs=2e-4) and k == 0.0


def test_sellmeier_range_error():
    with pytest.raises(ValueError):
        fm.eps_at({"type": "library", "name": "TiO2"}, 400)
    assert fm.valid_range({"type": "library", "name": "TiO2"}) == (430.0, 1530.0)


def test_drude_loss_sign():
    e = fm.eps_at({"type": "drude", "eps_inf": 1.0, "omega_p_eV": 9.0, "gamma_eV": 0.07}, 600)
    assert e.real < 0 and e.imag > 0  # exp(-i omega t): Verlust = Im eps > 0


def test_table_interpolation_and_range():
    spec = {"type": "table", "rows": [[400, 1.0, 0.0], [800, 2.0, 0.0]]}
    assert fm.nk(spec, 600)[0] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        fm.eps_at(spec, 900)


def test_describe_all_types():
    for spec in ({"type": "library", "name": "Si"}, {"type": "index", "n": 1, "k": 0}, {"type": "eps", "re": 1, "im": 0},
                 {"type": "drude", "eps_inf": 1, "omega_p_eV": 9, "gamma_eV": 0.1}, {"type": "table", "rows": [[1, 1, 0], [2, 1, 0]]}):
        assert isinstance(fm.describe(spec), str)


def test_tabulated_needs_data(tmp_path):
    fm.set_repo(tmp_path)
    try:
        if not fm.library_available():
            with pytest.raises(FileNotFoundError):
                fm.eps_at({"type": "library", "name": "Si"}, 405)
    finally:
        fm.set_repo(None)


def test_tabulated_with_data(tmp_path):
    d = tmp_path / "python" / "hpfem" / "data"
    d.mkdir(parents=True)
    (d / "Si.csv").write_text("# test\n0.4,4.0,0.0\n0.5,3.0,0.0\n", encoding="utf-8")
    fm.set_repo(tmp_path)
    try:
        assert fm.library_available()
        assert fm.eps_at({"type": "library", "name": "Si"}, 450) == pytest.approx(3.5**2)
        assert fm.valid_range({"type": "library", "name": "Si"}) == (400.0, 500.0)
    finally:
        fm.set_repo(None)
