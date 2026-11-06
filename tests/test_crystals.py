"""Crystal materials (hp-FEM M19) and the Jones / Mueller display: pure Python, no hpfem."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import fem_materials as fm
import fem_post as fp
import fem_worker as fw

UNIAXIAL = {"type": "crystal", "kind": "uniaxial", "n_o": 1.5, "k_o": 0.0, "n_e": 2.0, "k_e": 0.1, "theta_deg": 90.0, "phi_deg": 90.0}


def test_uniaxial_axis_and_principal_values():
    t = fm.tensor_at(UNIAXIAL, 600)
    # axis theta = 90°, phi = 90°: along y (the lines)
    assert t == pytest.approx(np.diag([2.25, complex(2.0, 0.1) ** 2, 2.25]))
    w = fm.principal_values(UNIAXIAL, 600)
    assert w[-1] == pytest.approx(complex(2.0, 0.1) ** 2)
    assert fm.eps_at(UNIAXIAL, 600) == pytest.approx(complex(2.0, 0.1) ** 2)  # the largest, for mesh sizes
    assert fm.nk(UNIAXIAL, 600) == pytest.approx((2.0, 0.1))
    assert fm.is_crystal(UNIAXIAL) and not fm.is_crystal({"type": "index", "n": 1.5, "k": 0.0})


def test_tilted_axis_is_symmetric_with_off_diagonals():
    spec = dict(UNIAXIAL, theta_deg=30.0, phi_deg=20.0)
    t = fm.tensor_at(spec, 600)
    assert np.allclose(t, t.T)
    assert abs(t[0, 2]) > 0.1 and abs(t[1, 2]) > 0.01
    a = np.array([np.sin(np.radians(30)) * np.cos(np.radians(20)), np.sin(np.radians(30)) * np.sin(np.radians(20)), np.cos(np.radians(30))])
    assert t @ a == pytest.approx(complex(2.0, 0.1) ** 2 * a)  # the axis is the extraordinary eigenvector


def test_biaxial_euler_rotation():
    spec = {"type": "crystal", "kind": "biaxial", "n": [1.5, 1.6, 1.7], "k": [0.0, 0.0, 0.0], "euler_deg": [0.0, 0.0, 0.0]}
    assert fm.tensor_at(spec, 500) == pytest.approx(np.diag([2.25, 2.56, 2.89]))
    spec["euler_deg"] = [90.0, 0.0, 0.0]  # principal axis 1 turned from x to y
    assert fm.tensor_at(spec, 500) == pytest.approx(np.diag([2.56, 2.25, 2.89]))
    spec["euler_deg"] = [10.0, 35.0, -20.0]
    t = fm.tensor_at(spec, 500)
    assert np.allclose(t, t.T) and np.sort(np.linalg.eigvalsh(t.real)) == pytest.approx([2.25, 2.56, 2.89])


def test_full_tensor_must_be_symmetric():
    spec = {"type": "crystal", "kind": "tensor", "re": [[2.0, 0.1, 0.0], [0.1, 2.5, 0.0], [0.0, 0.0, 3.0]], "im": [[0.0] * 3] * 3}
    assert fm.tensor_at(spec, 500)[0, 1] == pytest.approx(0.1)
    spec["re"] = [[2.0, 0.1, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 3.0]]
    with pytest.raises(ValueError):
        fm.tensor_at(spec, 500)


def test_isotropic_tensor_and_describe():
    assert fm.tensor_at({"type": "index", "n": 1.5, "k": 0.0}, 500) == pytest.approx(2.25 * np.eye(3))
    assert fm.CRYSTAL_KINDS == ("uniaxial", "biaxial", "tensor")
    assert "einachsig" in fm.describe(UNIAXIAL)


class _Order:
    def __init__(self, m, side, j, channels=("s", "p")):
        self.m, self.side, self.jones, self.channels = m, side, np.asarray(j, dtype=complex), channels
        a = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
        self.mueller = (a @ np.kron(self.jones, self.jones.conj()) @ np.linalg.inv(a)).real


class _JonesResult:
    def __init__(self):
        self.R_orders = [_Order(0, "R", [[0.3, 0.05j], [-0.05j, -0.2]]), _Order(-1, "R", [[0.1, 0.0], [0.0, 0.1]])]
        self.T_orders = [_Order(0, "T", [[0.8, 0.1], [0.05, 0.7]], ("ordinary", "extraordinary"))]


def test_jones_data_round_trip_and_tables():
    data = fw.jones_data(_JonesResult())
    assert set(data) == {"R0", "R-1", "T0"}
    pt = {"index": 0, "value": 600.0, "jones": data}
    j = fp.jones_matrix(pt, "R0")
    assert j == pytest.approx(np.array([[0.3, 0.05j], [-0.05j, -0.2]]))
    assert fp.jones_orders([pt]) == ["R-1", "R0", "T0"]
    df = fp.jones_dataframe(pt, "T0")
    assert list(df["aus"]) == ["ordinary", "ordinary", "extraordinary", "extraordinary"]
    # power normalisation: the column sum of |J|^2 is the efficiency for that incident polarisation, M00 the unpolarised mean
    eff_s = df[df["ein"] == "s"]["|J|² (Effizienz)"].sum()
    assert eff_s == pytest.approx(0.8**2 + 0.05**2)
    m = np.asarray(data["T0"]["M"])
    assert m[0, 0] == pytest.approx(0.5 * np.sum(np.abs(np.array([[0.8, 0.1], [0.05, 0.7]])) ** 2))
    mdf = fp.mueller_dataframe(pt, "R0")
    assert mdf.iloc[0, 1] == pytest.approx(1.0)
    assert fp.jones_dataframe({"index": 1}, "R0").empty


def test_mueller_figures_single_point_and_sweep():
    data = fw.jones_data(_JonesResult())
    pts = [{"index": i, "value": 500.0 + 10 * i, "jones": data} for i in range(3)]
    fig = fp.fig_mueller(pts[:1], "none", "R0")
    assert fig is not None
    fig = fp.fig_mueller(pts, "wavelength", "R0")
    assert len(fig.axes) == 16
    fig = fp.fig_mueller([{"index": 0, "error": "x"}], "wavelength", "R0")
    assert fig is not None
