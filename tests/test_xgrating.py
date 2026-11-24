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


# ---------------------------------------------------------------------------------------------------------------- job, worker (fake hpfem), tables
def _fake_hpfem(monkeypatch, record):
    """A stand-in for hpfem and hpfem.crossed that records the grating and returns a lossless two-order result."""
    import sys
    import types
    from dataclasses import dataclass, field

    h = types.ModuleType("hpfem")
    h.__version__ = "0.0-fake"
    h.set_log_level = lambda level: None
    h.Material = lambda eps, mu=1.0: ("Material", complex(eps), mu)
    cg = types.ModuleType("hpfem.crossed")

    @dataclass
    class Shape:
        kind: str
        material: object
        center: tuple = (0.0, 0.0)
        radius: float | None = None
        rx: float | None = None
        ry: float | None = None
        width: float | None = None
        height: float | None = None
        points: object = None
        angle: float = 0.0

    @dataclass
    class Layer:
        thickness: float
        material: object
        shapes: list = field(default_factory=list)

    @dataclass
    class CrossedGrating:
        period: tuple
        superstrate: object
        substrate: object
        layers: list

    class Result:
        def __init__(self, order, pol):
            self.R, self.T, self.absorption, self.balance, self.num_dofs, self.num_cells = 0.25, 0.75, 0.0, 1e-5 * order, 1000 * order, 50
            self._pol = pol

        def table(self):
            return [{"side": "R", "m": 0, "n": 0, "efficiency": 0.25}, {"side": "T", "m": 0, "n": 0, "efficiency": 0.7}, {"side": "T", "m": 1, "n": 0, "efficiency": 0.05}]

        def polarised(self, pol):
            return Result(1, pol)

    def solve(grating, wavelength, theta, phi, polarisation, **kw):
        record.append(dict(grating=grating, wavelength=wavelength, theta=theta, phi=phi, pol=polarisation, **kw))
        return Result(kw["order"], polarisation)

    cg.Shape, cg.Layer, cg.CrossedGrating, cg.solve = Shape, Layer, CrossedGrating, solve
    h.crossed = cg
    monkeypatch.setitem(sys.modules, "hpfem", h)
    monkeypatch.setitem(sys.modules, "hpfem.crossed", cg)


def test_job_document():
    m = fx.default_model()
    m["sweep"].update(mode="wavelength", start=400.0, stop=500.0, n=3)
    m["convergence"].update(enabled=True, orders=[4, 2, 4])
    j = fx.job(m)
    assert j["wavelengths_nm"] == [400.0, 450.0, 500.0] and j["convergence_orders"] == [2, 4] and j["mesh"]["order"] == 3
    assert fx.job(fx.default_model())["convergence_orders"] == []
    json.dumps(j)


def test_worker_maps_model_and_collects_results(monkeypatch, tmp_path):
    import fem_xgrating_worker as fw
    rec = []
    _fake_hpfem(monkeypatch, rec)
    m = fx.default_model()
    m["layers"][0]["shapes"] += [fx.new_shape("rect", "Luft"), fx.new_shape("polygon", "Luft")]
    m["layers"][0]["shapes"][1]["angle"] = 30.0
    m["incidence"].update(theta=30.0, phi=20.0, pol="p")
    m["sweep"].update(mode="wavelength", start=400.0, stop=410.0, n=2)
    m["convergence"].update(enabled=True, orders=[2, 3])
    m["mesh"]["elements_per_wavelength"] = 2.5
    (tmp_path / "job.json").write_text(json.dumps(fx.job(m)), encoding="utf-8")
    assert fw.main([str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    # steps: convergence p = 2 (the main order 3 is the first wavelength at p = 3), then the two wavelengths
    assert [c["order"] for c in res["convergence"]] == [2, 3] and [p["lam_nm"] for p in res["points"]] == [400.0, 410.0]
    assert [r["order"] for r in rec] == [2, 3, 3]
    call = rec[1]
    assert call["wavelength"] == pytest.approx(400e-9) and call["theta"] == pytest.approx(0.5235987756) and call["phi"] == pytest.approx(0.3490658504)
    assert call["pol"] == "p" and call["elements_per_wavelength"] == 2.5
    g = call["grating"]
    assert g.period == pytest.approx((500e-9, 500e-9)) and g.layers[0].thickness == pytest.approx(150e-9)
    circle, rect, poly = g.layers[0].shapes
    assert circle.kind == "circle" and circle.radius == pytest.approx(125e-9) and circle.center == pytest.approx((250e-9, 250e-9))
    assert rect.kind == "rectangle" and rect.width == pytest.approx(200e-9) and rect.height == pytest.approx(200e-9)
    assert poly.kind == "polygon" and len(poly.points) == 3
    p0 = res["points"][0]
    assert set(p0["pols"]) == {"s", "p"} and p0["R"] == 0.25 and p0["dofs"] == 3000


def test_worker_reports_errors_per_point(monkeypatch, tmp_path):
    import fem_xgrating_worker as fw
    _fake_hpfem(monkeypatch, [])
    import hpfem.crossed as cg

    def boom(*a, **k):
        raise ValueError("kaputt")
    monkeypatch.setattr(cg, "solve", boom)
    (tmp_path / "job.json").write_text(json.dumps(fx.job(fx.default_model())), encoding="utf-8")
    assert fw.main([str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res["points"][0]["error"] == "kaputt"


def test_worker_cancel(monkeypatch, tmp_path):
    import fem_xgrating_worker as fw
    rec = []
    _fake_hpfem(monkeypatch, rec)
    (tmp_path / "job.json").write_text(json.dumps(fx.job(fx.default_model())), encoding="utf-8")
    (tmp_path / "cancel").write_text("cancel", encoding="utf-8")
    assert fw.main([str(tmp_path)]) == 0
    assert rec == [] and json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["cancelled"] is True


_POINT = {"lam_nm": 405.0, "R": 0.03, "T": 0.97, "A": 0.0, "balance": -1e-5, "dofs": 10, "cells": 5, "time_s": 1.0,
          "rows": [{"side": "R", "m": 0, "n": 0, "efficiency": 0.0247}, {"side": "R", "m": 1, "n": 0, "efficiency": 0.0018}, {"side": "T", "m": 0, "n": 0, "efficiency": 0.5}]}


def test_tables():
    df = fx.orders_dataframe(_POINT)
    assert list(df.columns) == ["m", "n", "R", "T"] and df.iloc[0]["m"] == "0" and df.iloc[-1]["m"] == "Σ" and df.iloc[-1]["R"] == 0.03
    assert "ΣR + ΣT + A − 1 = -1.00e-05" in fx.balance_text(_POINT)
    sp = fx.spectrum_dataframe([_POINT])
    assert sp.iloc[0]["R₀"] == 0.0247 and sp.iloc[0]["T₀"] == 0.5
    conv = fx.convergence_dataframe([dict(_POINT, order=2), dict(_POINT, order=3)], {("T", 0, 0): 0.49, ("R", 0, 0): 0.0247})
    assert list(conv["p"]) == [2, 3] and conv.iloc[0]["|Δ T₀|"] == pytest.approx(0.01)


_FG_CSV = """lambda_um,phi_deg,psi_deg,delta_deg,side,n,m,efficiency
0.405,0,0,90,R,0,0,0.0246797573746
0.405,0,0,90,R,1,0,0.00183
0.405,0,0,90,T,0,1,0.02
0.405,0,0,0,R,0,0,0.99
0.5,0,0,90,R,0,0,0.5
"""


def test_fg_csv_selection_and_order_labels():
    ref = fx.read_fg_csv(_FG_CSV, 405.0, "p")
    assert ref == {("R", 0, 0): 0.0246797573746, ("R", 1, 0): 0.00183, ("T", 0, -1): 0.02}      # FG (n, m) = (0, 1) is (m, n) = (0, -1) here
    assert fx.read_fg_csv(_FG_CSV, 405.0, "s") == {("R", 0, 0): 0.99}
    with pytest.raises(ValueError, match="Keine Zeilen"):
        fx.read_fg_csv(_FG_CSV, 600.0, "p")
    with pytest.raises(ValueError, match="Format"):
        fx.read_fg_csv("a,b\n1,2\n", 405.0, "p")
    cmp = fx.comparison_dataframe(_POINT, ref)
    row = cmp[(cmp["Seite"] == "R") & (cmp["m"] == 0) & (cmp["n"] == 0)].iloc[0]
    assert row["Δ = FEM − RCWA"] == pytest.approx(0.0247 - 0.0246797573746)
    # azimuth: FG psi = -phi
    assert fx.read_fg_csv(_FG_CSV.replace(",0,0,90,R,0,0,0.0246", ",0,-20,90,R,0,0,0.0246"), 405.0, "p", phi=20.0)[("R", 0, 0)] == 0.0246797573746


def test_new_figures_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [dict(_POINT, order=2), dict(_POINT, lam_nm=410.0, order=3)]
    ref = {("T", 0, 0): 0.49}
    for f in (fx.fig_spectrum(pts), fx.fig_convergence(pts, ref), fx.fig_convergence(pts)):
        plt.close(f)


def test_polygon_and_rotation_outline():
    m = fx.default_model()
    r = fx.new_shape("rect", "Luft")
    r.update(wx=200.0, wy=100.0, angle=90.0)
    hx, hy = fx.shape_extent(r)
    assert (round(hx), round(hy)) == (50, 100)
    m["layers"][0]["shapes"] = [fx.new_shape("polygon", "Luft")]
    assert not [x for x in fx.validate(m) if x[0] == "error"]
    m["layers"][0]["shapes"][0]["points"] = [[0, 0], [1, 1]]
    assert any("Polygon" in t for lvl, t in fx.validate(m) if lvl == "error")


def test_validate_rejects_lossy_halfspace_and_crystal():
    m = fx.default_model()
    m["materials"]["Glas"] = {"type": "index", "n": 1.5, "k": 0.1}
    assert any("verlustfreie" in t for lvl, t in fx.validate(m) if lvl == "error")
    m = fx.default_model()
    m["materials"]["Kristall"] = {"type": "crystal", "kind": "uniaxial", "n_o": 1.5, "k_o": 0.0, "n_e": 1.6, "k_e": 0.0, "theta_deg": 0.0, "phi_deg": 0.0}
    m["layers"][0]["material"] = "Kristall"
    assert any("Kristalle" in t for lvl, t in fx.validate(m) if lvl == "error")


def test_pol_difference_swaps_orders():
    rows_s = [{"side": "T", "m": 0, "n": 0, "efficiency": 0.8}, {"side": "T", "m": 1, "n": 0, "efficiency": 0.1}]
    rows_p = [{"side": "T", "m": 0, "n": 0, "efficiency": 0.8}, {"side": "T", "m": 0, "n": 1, "efficiency": 0.1}]
    pt = {"pols": {"s": {"R": 0.1, "T": 0.9, "rows": rows_s}, "p": {"R": 0.1, "T": 0.9, "rows": rows_p}}}
    d = fx.pol_difference(pt)
    assert d["swapped"] == 0.0 and d["plain"] == pytest.approx(0.1) and d["sum_T"] == 0.0
