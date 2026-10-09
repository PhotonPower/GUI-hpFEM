"""Neue Funktionen für hp-FEM 0.4 / M16 (ohne hpfem: Job, Protokoll, Abbruch, Auswertung der neuen Ergebnisse)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import fem_geometry as fg  # noqa: E402
import fem_post as fp  # noqa: E402
import fem_run as fr  # noqa: E402
import fem_worker as fw  # noqa: E402


def test_boundary_tags_are_hpfem_box_tags():
    assert (fg.TAG_LEFT, fg.TAG_RIGHT, fg.TAG_BOTTOM, fg.TAG_TOP) == (1, 2, 3, 4)
    assert (fw.TAG_LEFT, fw.TAG_RIGHT, fw.TAG_BOTTOM, fw.TAG_TOP) == (1, 2, 3, 4)


def test_make_job_task_and_resonance():
    model = next(iter(fg.presets().values()))
    job = fr.make_job(model, {"order": 3}, {"enabled": False}, None, "resonances", {"num_modes": 5})
    assert job["task"] == "resonances" and job["resonance"]["num_modes"] == 5
    assert fr.make_job(model, {"order": 3}, {"enabled": False})["task"] == "scattering"


def test_cancel_file(tmp_path):
    fr.request_cancel(tmp_path)
    assert (tmp_path / "cancel").is_file()


def test_pythonpath_only_for_built_source(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    (tmp_path / "python" / "hpfem").mkdir(parents=True)
    assert not fr.source_has_module(tmp_path)
    assert "PYTHONPATH" not in fr.worker_env(tmp_path)
    (tmp_path / "python" / "hpfem" / "_hpfem.cp311-win_amd64.pyd").write_bytes(b"")
    assert fr.source_has_module(tmp_path)
    assert fr.worker_env(tmp_path, 4)["PYTHONPATH"].endswith("python")
    assert fr.worker_env(tmp_path, 4)["OMP_NUM_THREADS"] == "4"
    assert not fr.source_has_module("")


def test_log_phase_estimate_diagnostics_cancel():
    log = ("ESTIMATE 35357 DoFs, 398 MiB (p = 4, SPARSE_LU)\nPROGRESS 2/5 lambda\nPHASE factorisation 2/5 (0.3 s)\n"
           "DIAG warning grazing_order: order 1 propagates at 80.2 deg\nCANCELLED\n")
    frac, label = fr.progress_info(log)
    assert 0.2 <= frac < 0.4 and "Faktorisierung" in label
    assert fr.estimate_line(log).startswith("35357 DoFs")
    assert fr.diagnostics_lines(log) == [("warning", "grazing_order", "order 1 propagates at 80.2 deg")]
    assert fr.log_flags(log)["cancelled"]


def test_worker_helpers():
    ctx = type("Ctx", (), {"k0": 2 * np.pi / 500e-9, "n_cover": 1.0})()
    kx, beta = fw.bloch_wavenumbers(ctx, 30.0, 0.0)
    assert abs(kx - ctx.k0 * 0.5) < 1e-6 * ctx.k0 and beta == 0.0
    kx, beta = fw.bloch_wavenumbers(ctx, 30.0, 90.0)
    assert abs(kx) < 1e-9 * ctx.k0 and abs(beta - ctx.k0 * 0.5) < 1e-6 * ctx.k0
    job = {"incidence": {"wavelength_nm": 600.0, "theta": 10.0, "phi": 0.0}, "sweep": {"mode": "theta"}}
    assert fw.point_values(job, 25.0) == (600.0, 25.0, 0.0)
    d = type("D", (), {"code": "pml_thin", "severity": "warning", "text": "t", "hint": "h"})()
    assert fw.diag_list([d]) == [dict(code="pml_thin", severity="warning", text="t", hint="h")]


def _map(with_hs=True):
    nx, ny = 6, 5
    rng = np.random.default_rng(0)
    E = (rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3))).astype(np.complex64)
    mp = dict(x_nm=np.linspace(0, 200, nx), y_nm=np.linspace(-100, 100, ny), E=E, omega=3e15, lam_nm=600.0, S_inc=1e-3, eps_names=np.array(["Luft", "Si"]),
              eps=np.array([1.0 + 0j, 12 + 0.1j]), theta=0.0, phi=0.0, pol="TE", kx=0.0, period_nm=200.0)
    if with_hs:
        mp["H"] = E * 0.002
        mp["S"] = (np.abs(E) ** 2 * 1e-3).astype(np.complex64)
    return mp


def test_quantities_h_and_s():
    model = fg.presets()[next(n for n in fg.presets() if n.startswith("Dünnschicht"))]
    d = fp.derived(_map(True), model)
    keys = fp.available_quantities(d)
    assert any(k.startswith("|H|") for k in keys) and any(k.startswith("Poynting S_y") for k in keys)
    assert fp.quantity(d, "Poynting S_y (Fluss vertikal)").shape == (6, 5)
    d0 = fp.derived(_map(False), model)
    assert not any(k.startswith("|H|") or k.startswith("Poynting") for k in fp.available_quantities(d0))
    import matplotlib.pyplot as plt
    plt.close(fp.fig_map(_map(True), d, model, "Poynting S_y (Fluss vertikal)", periods=2))


def test_jacobian_resonance_balance_tables_and_figures():
    import matplotlib.pyplot as plt
    jac = dict(rows=["R0", "R1", "T0"], cols=["Re ε(Si)", "Im ε(Si)", "λ", "θ"], units=["", "", "1/nm", "1/°"],
               J=[[0.1, 0.2, 1e-3, 2e-3], [0.0, 0.1, 0.0, 1e-3], [-0.1, -0.3, -1e-3, 0.0]])
    pts = [dict(index=i, value=500.0 + 10 * i, jacobian=jac, R=0.3, T=0.6, absorbed={"total": 0.1}, A_exact=0.1,
                flux_balance={"relative_residual": 1e-4}, timing={"solver.assembly": 0.5, "solver.factorisation": 1.0, "total": 2.0}) for i in range(3)]
    df = fp.jacobian_dataframe(pts[0])
    assert list(df["Ordnung"])[-2:] == ["R gesamt", "T gesamt"]
    assert abs(df.iloc[-2, 1] - 0.1) < 1e-12
    plt.close(fp.fig_jacobian(pts, "wavelength"))
    plt.close(fp.fig_jacobian(pts[:1], "none"))
    assert abs(fp.balance_dataframe(pts)["R + T + A"].iloc[0] - 1.0) < 1e-12
    assert "Faktorisierung" in fp.timing_dataframe(pts).columns
    res = [dict(index=i, value=float(i), theta=float(i), phi=0.0, kx_over_G=0.1 * i, lam_target_nm=600.0, dofs=10, time_s=1.0,
                modes=[dict(m=0, lam_nm=610.0 + i, Q=50.0, omega=[3e15, -3e13], residual=1e-9), dict(m=1, lam_nm=590.0, Q=-5.0, omega=[3e15, 1e13], residual=1e-9)])
           for i in range(3)]
    rdf = fp.resonances_dataframe(res)
    assert len(rdf) == 6 and (rdf["Q"] < 0).sum() == 3
    plt.close(fp.fig_resonances(res, "theta"))
    plt.close(fp.fig_resonances(res[:1], "none"))
