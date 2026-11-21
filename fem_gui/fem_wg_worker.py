"""Worker for waveguides and ports (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_wg_worker.py <job folder>

Reads <folder>/job.json (fem_wg.job) and <folder>/mesh.msh (fem_wg.build_mesh).

* task ``ports``: per wavelength hpfem.s_parameters of the device between the modal ports (tags 101 left, 102 right; PEC walls 103, 104):
  the power-normalised S-matrix of all propagating channels, the effective indices of the port modes, transmission / reflection / loss of the
  excitation channel and (optionally) the field of one excitation on the subdivided mesh (field_<i>.npz: points [nm], simplices, values E_x, E_y).
* task ``modes``: per wavelength hpfem.PropagatingMode on the cross-section (PEC on all four sides): n_eff and the mode fields
  (mode_<i>.npz: points [nm], simplices, et_<k> (n, 2), ez_<k> (n,)).

Writes <folder>/results.json after every wavelength and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: the
file <folder>/cancel, checked between the wavelengths.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_materials as fm  # noqa: E402

NM = 1e-9
TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 101, 102, 103, 104
SUB = 2                                                              # subdivisions per cell for the field maps


def say(*a):
    print(*a, flush=True)


def material_map(h, job, lam_nm):
    """MaterialMap with every material of the job on the tag 1 + its index."""
    names = list(job["materials"])
    default = complex(fm.eps_at(job["materials"][job["background"]], lam_nm))
    mm = h.MaterialMap(h.Material(default))
    for i, n in enumerate(names):
        mm.set(1 + i, h.Material(complex(fm.eps_at(job["materials"][n], lam_nm))))
    return mm


def triangulation(h, mesh, nd, vec, h1=None, h1vec=None):
    """(points [nm], simplices, E) on the subdivided mesh: vec = Nedelec unknowns (n, 2); h1vec optional E_z (n,)."""
    sub = h.subdivide(mesh, SUB)
    sm = sub.mesh
    pts = np.asarray(sm.vertices)[:, :2] / NM
    simp = np.asarray(sm.cells, dtype=int)
    cells = list(map(int, sub.vertex_parent))
    xi = np.asarray(sub.vertex_xi, dtype=float)
    e = np.asarray(h.sample_hcurl(nd, np.asarray(vec, dtype=complex), cells, xi))
    ez = None if h1 is None else np.asarray(h.sample_h1(h1, np.asarray(h1vec, dtype=complex), cells, xi))
    return pts, simp, e, ez


def run_ports(h, job, mesh, folder, i, lam):
    t0 = time.time()
    p = int(job["mesh"]["order"])
    nd = h.NedelecDofMap2D(mesh, p)
    omega = 2 * np.pi / (lam * NM) * h.constants.c0
    sides = [s for s in ("left", "right") if job["ports"][s]["enabled"]]
    tag_of = {"left": TAG_LEFT, "right": TAG_RIGHT}

    def make_setup(incident=None):
        setup = h.ScatteringSetup2D()
        setup.omega = omega
        setup.materials = material_map(h, job, lam)
        setup.pec_tags = [TAG_BOTTOM, TAG_TOP]
        ports = []
        for k, s in enumerate(sides):
            n_modes = int(job["ports"][s]["modes"])
            if incident is not None and incident[0] == k:
                amp = [0.0] * n_modes
                amp[min(incident[1], n_modes - 1)] = 1.0
                ports.append(h.WaveguidePort(tag_of[s], n_modes, amp))
            else:
                ports.append(h.WaveguidePort(tag_of[s], n_modes))
        setup.ports = ports
        return setup

    setup = make_setup()
    probe = h.Scattering2D(nd, setup)
    port_modes = {}
    for k, s in enumerate(sides):
        pm = probe.port_modes(k)
        port_modes[s] = [dict(n_eff=float(m.effective_index.real), propagating=bool(m.propagating)) for m in pm.modes]
    sp = h.s_parameters(nd, setup)
    channels = [dict(port=int(c.port), side=sides[int(c.port)], mode=int(c.mode), beta=float(complex(c.beta).real)) for c in sp.channels]
    s_mat = np.asarray(sp.s)
    exc = job.get("excite", {"port": 0, "mode": 0})
    exc_side = sides[min(int(exc["port"]), len(sides) - 1)] if exc["port"] < len(sides) else sides[0]
    cands = [k for k, c in enumerate(channels) if c["side"] == exc_side and c["mode"] == int(exc["mode"])]
    c0 = cands[0] if cands else 0
    col = np.abs(s_mat[:, c0]) ** 2
    refl = float(sum(col[k] for k, c in enumerate(channels) if c["port"] == channels[c0]["port"]))
    trans = float(sum(col[k] for k, c in enumerate(channels) if c["port"] != channels[c0]["port"]))
    res = dict(index=i, lam_nm=float(lam), channels=channels, S_re=s_mat.real.tolist(), S_im=s_mat.imag.tolist(), port_modes=port_modes,
               excitation=channels[c0], R=refl, T=trans, loss=float(1.0 - refl - trans), num_dofs=int(nd.num_dofs))
    want = job.get("field", "first")
    if want == "all" or (want == "first" and i == 0):
        k_exc = sides.index(channels[c0]["side"])
        prob = h.Scattering2D(nd, make_setup((k_exc, channels[c0]["mode"])))
        sol = prob.solve()
        pts, simp, e, _ = triangulation(h, mesh, nd, sol.unknown)
        np.savez_compressed(Path(folder) / f"field_{i}.npz", points=pts, simplices=simp, values=e, tag=np.asarray(mesh.cell_tags)[np.asarray(h.subdivide(mesh, SUB).parent_cell, dtype=int)])
        res["field"] = f"field_{i}.npz"
    res["time_s"] = time.time() - t0
    return res


def run_modes(h, job, mesh, folder, i, lam):
    t0 = time.time()
    p = int(job["mesh"]["order"])
    nd, h1 = h.NedelecDofMap2D(mesh, p), h.DofMap2D(mesh, p)
    ws = h.WaveguideSetup()
    ws.omega = 2 * np.pi / (lam * NM) * h.constants.c0
    ws.materials = material_map(h, job, lam)
    ws.pec_tags = [TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP]
    ws.num_modes = int(job["modes"]["num"])
    modes = h.PropagatingMode(nd, h1, ws).solve()
    arrays = {}
    for k, m in enumerate(modes):
        pts, simp, e, ez = triangulation(h, mesh, nd, m.transverse, h1, m.longitudinal)
        arrays.update({"points": pts, "simplices": simp, f"et_{k}": e, f"ez_{k}": ez})
    if arrays:
        np.savez_compressed(Path(folder) / f"mode_{i}.npz", **arrays)
    res = dict(index=i, lam_nm=float(lam), n_eff=[float(m.effective_index.real) for m in modes],
               beta=[float(complex(m.beta).real) for m in modes], modes_file=f"mode_{i}.npz" if arrays else None,
               num_dofs=int(nd.num_dofs + h1.num_dofs), time_s=time.time() - t0)
    return res


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    mesh = hpfem.read_gmsh(str(folder / "mesh.msh"), NM, 2)
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, mesh: {mesh.num_cells} cells")
    values = [float(v) for v in job["wavelengths_nm"]]
    out = dict(meta=dict(name=job.get("name", ""), task=job["task"], started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         hpfem=str(getattr(hpfem, "__version__", "?")), cells=int(mesh.num_cells), order=int(job["mesh"]["order"])), points=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    fn = run_ports if job["task"] == "ports" else run_modes
    for i, lam in enumerate(values):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return 0
        say(f"PROGRESS {i + 1}/{len(values)} lambda = {lam:.3f} nm")
        try:
            res = fn(hpfem, job, mesh, folder, i, lam)
            out["points"].append(res)
            if job["task"] == "ports":
                say(f"  {res['time_s']:.1f} s: T = {res['T']:.4f}, R = {res['R']:.4f}, loss = {res['loss']:.4f}")
            else:
                say(f"  {res['time_s']:.1f} s: n_eff = " + ", ".join(f"{v:.4f}" for v in res["n_eff"]))
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(index=i, lam_nm=lam, error=str(exc)))
        flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
