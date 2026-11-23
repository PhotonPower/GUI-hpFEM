import json

import pytest

import fem_materials as fm
import fem_xgrating as fx


def test_default_model_valid_and_roundtrip():
    m = fx.default_model()
    assert not [x for x in fx.validate(m) if x[0] == "error"]
    back = fx.load_model(json.dumps(m))
    assert back == m


def test_presets_valid():
    for name, m in fx.presets().items():
        assert not [x for x in fx.validate(m) if x[0] == "error"], name
        assert fx.load_model(json.dumps(m))["name"] == name


def test_load_normalises_foreign_materials():
    m = fx.default_model()
    m["materials"] = {"Luft": {"n": 1.0}, "Glas": {"n": 1.5, "k": 0.0}}
    back = fx.load_model(json.dumps(m))
    assert back["materials"]["Glas"] == {"type": "index", "n": 1.5, "k": 0.0}
    assert fm.describe(back["materials"]["Luft"]) == "n = 1, k = 0"


def test_load_rejects_readably():
    m = fx.default_model()
    with pytest.raises(ValueError, match="FourierGrating"):
        fx.load_model(json.dumps({"dimension": "2D", "period_um": [0.5, 0.5]}))
    bad = dict(m, materials={"Luft": {"foo": 1}})
    with pytest.raises(ValueError, match="Material 'Luft'"):
        fx.load_model(json.dumps(bad))
    bad = dict(m, substrate="Diamant")
    with pytest.raises(ValueError, match="nicht definiert"):
        fx.load_model(json.dumps(bad))
    bad = json.loads(json.dumps(m))
    bad["layers"][0]["shapes"][0]["type"] = "stern"
    with pytest.raises(ValueError, match="unbekannte Form"):
        fx.load_model(json.dumps(bad))


def test_validate_catches_errors_and_warnings():
    m = fx.default_model()
    m["layers"][0]["shapes"][0]["radius"] = 0.0
    m["incidence"]["theta"] = 90.0
    m["layers"][0]["thickness_nm"] = -1.0
    texts = " ".join(t for lvl, t in fx.validate(m) if lvl == "error")
    assert "radius" in texts and "θ" in texts and "Dicke" in texts
    m = fx.default_model()
    m["layers"][0]["shapes"][0]["radius"] = 300.0
    assert any(lvl == "warn" for lvl, _ in fx.validate(m))


def test_orders_normal_incidence_example():
    """λ = 405 nm, p = 500 nm, cover air, substrate glass 1.5: reflection only 0 (p/λ = 1.23 > 1), transmission up to |(m, n)|² ≤ 4 (1.5 p/λ = 1.85)."""
    m = fx.default_model()
    po = fx.propagating_orders(m, 405.0)
    assert po["R"][0] == (0, 0)
    assert set(po["R"]) == {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}   # n_c p / lambda = 1.2346 -> m^2 + n^2 < 1.52
    assert (1, 1) not in po["R"] and (1, 1) in po["T"] and (2, 0) not in po["T"]
    assert po["T"] == sorted(po["T"], key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    rows = fx.orders_table(m, 405.0)
    assert rows[0]["m"] == 0 and rows[0]["n"] == 0 and rows[0]["R"] is None


def test_orders_oblique_breaks_symmetry():
    m = fx.default_model()
    m["incidence"].update(theta=30.0, phi=0.0)
    po = fx.propagating_orders(m, 405.0)
    assert (-1, 0) in po["R"] and (1, 0) not in po["R"]      # kx0 = k0 sin 30 -> +1 order evanescent in air
    assert (0, 1) in po["R"] and (0, -1) in po["R"]          # ky symmetric for phi = 0


def test_sweep_list():
    m = fx.default_model()
    assert fx.sweep_list(m["sweep"], m["incidence"]) == [405.0]
    m["sweep"].update(mode="wavelength", start=400.0, stop=500.0, n=3)
    assert fx.sweep_list(m["sweep"], m["incidence"]) == [400.0, 450.0, 500.0]


def test_rename_material():
    m = fx.default_model()
    fx.rename_material(m, "Luft", "Vakuum")
    assert m["cover"] == "Vakuum" and m["layers"][0]["shapes"][0]["material"] == "Vakuum" and "Luft" not in m["materials"]
    assert not [x for x in fx.validate(m) if x[0] == "error"]


def test_figures_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = fx.default_model()
    m["layers"][0]["shapes"].append(fx.new_shape("rect", "Luft"))
    for f in (fx.fig_stack(m), fx.fig_cell(m, 0, 2)):
        plt.close(f)
