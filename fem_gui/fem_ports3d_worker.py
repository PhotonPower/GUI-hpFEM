"""Worker for waveguides and ports in 3D (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_ports3d_worker.py <job folder> [estimate | inspect]

Reads <folder>/job.json (fem_ports3d.job). Devices: ``straight`` (one core), ``coupler`` (two cores), both on the structured tetrahedral mesh of hpfem.box with PEC
side walls and modal ports on the end faces z = 0, z = L; ``file`` (Gmsh mesh of the user: physical surfaces PEC / ports, physical volumes materials).
Per wavelength: hpfem.s_parameters (power-normalised S-matrix of all propagating channels), reflection / transmission of the excited channel, n_eff of the port
modes, for ``straight`` the phase of the transmission against beta L of the 2D mode solver, for ``coupler`` the coupled-mode reference
P_cross = sin^2(kappa L) from the supermodes of the cross-section, and (optional) |E|^2 of the excitation on two cut planes.
``estimate``: only the mesh and hpfem.estimate_memory (``ESTIMATE{json}``); ``inspect``: tags of a mesh file (``INSPECT{json}``).

Writes <folder>/results.json after every wavelength and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: file <folder>/cancel.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

NM = 1e-9
TAG_CLAD, TAG_A, TAG_B = 1, 2, 3
PORT_TAGS = (11, 12, 13, 14)
GRID = 61


def say(*a):
    print(*a, flush=True)


def grid_lines(edges, cell):
    lines = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, round((b - a) / cell))
        lines.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(lines)


def grid_box(h, xs, ys, zs):
    """Tetrahedral mesh (Kuhn subdivision, sides tagged with box_tag) of the tensor grid xs x ys x zs: hpfem.box on index coordinates, then the vertices moved to the grid
    lines, so the cell faces meet the core edges exactly (hpfem.box itself is uniform)."""
    mesh = h.box(len(xs) - 1, len(ys) - 1, len(zs) - 1, [0, 0, 0], [len(xs) - 1, len(ys) - 1, len(zs) - 1])
    for i, v in enumerate(np.asarray(mesh.vertices)):
        mesh.set_vertex(i, [xs[int(round(v[0]))], ys[int(round(v[1]))], zs[int(round(v[2]))]])
    return mesh


def grid_rectangle(h, xs, ys):
    mesh = h.rectangle(len(xs) - 1, len(ys) - 1, [0, 0], [len(xs) - 1, len(ys) - 1])
    for i, v in enumerate(np.asarray(mesh.vertices)):
        mesh.set_vertex(i, [xs[int(round(v[0]))], ys[int(round(v[1]))]])
    return mesh


def edges_of(job):
    w, g, mx, my, hgt = (job[k] * NM for k in ("width_nm", "gap_nm", "margin_x_nm", "margin_y_nm", "height_nm"))
    if job["device"] == "straight":
        xe = [-w / 2 - mx, -w / 2, w / 2, w / 2 + mx]
    else:
        a0, b1 = -g / 2 - w, g / 2 + w
        xe = [a0 - mx, a0, -g / 2, g / 2, b1, b1 + mx]
    return xe, [-my, 0.0, hgt, hgt + my]


def tag_cores(job, mesh):
    xe, _ = edges_of(job)
    hgt = job["height_nm"] * NM
    cores = [(xe[1], xe[2], TAG_A)] if job["device"] == "straight" else [(xe[1], xe[2], TAG_A), (xe[3], xe[4], TAG_B)]
    for c, x in enumerate(np.asarray(mesh.cell_centroids)):
        tag = TAG_CLAD
        if 0.0 < x[1] < hgt:
            for a, b, t in cores:
                if a < x[0] < b:
                    tag = t
        mesh.set_cell_tag(c, tag)


def build_mesh(h, job):
    """(mesh, info) in SI units."""
    if job["device"] == "file":
        mesh = h.read_gmsh(job["path"], float(job["scale"]), 3)
        return mesh, dict(kind="file")
    xe, ye = edges_of(job)
    cell = job["cell_nm"] * NM
    xs, ys = grid_lines(xe, cell), grid_lines(ye, cell)
    L = job["length_nm"] * NM
    nz = max(1, round(L / ((job.get("cell_z_nm") or job["cell_nm"]) * NM)))
    mesh = grid_box(h, xs, ys, np.linspace(0.0, L, nz + 1))
    tag_cores(job, mesh)
    half = job["device"] == "coupler" and job["style"] == "half"
    if half:
        for face, (left, right) in ((h.box_tag.Z_MIN, PORT_TAGS[:2]), (h.box_tag.Z_MAX, PORT_TAGS[2:])):
            for f in mesh.facets_with_tag(face):
                x = np.mean([mesh.vertex(int(v)) for v in mesh.facet_vertices(int(f))], axis=0)
                mesh.set_facet_tag(int(f), left if x[0] < 0 else right)
    return mesh, dict(kind=job["device"], half=half, length=L)


def inspect_mesh(h, path, scale):
    mesh = h.read_gmsh(path, float(scale), 3)
    cells = np.asarray(mesh.cell_tags)
    ftags = np.asarray(mesh.facet_tags)
    v = np.asarray(mesh.vertices)
    return dict(n_cells=int(mesh.num_cells), cells={str(int(t)): int((cells == t).sum()) for t in np.unique(cells)},
                facets={str(int(t)): int((ftags == t).sum()) for t in np.unique(ftags) if int(t) != int(h.NO_TAG)},
                bbox=[v.min(axis=0).tolist(), v.max(axis=0).tolist()], geometry_order=int(mesh.geometry_order))


def port_tags(h, job):
    if job["device"] == "file":
        return [(int(p["tag"]), int(p["modes"])) for p in job["ports"]]
    if job["device"] == "coupler" and job["style"] == "half":
        return [(t, 1) for t in PORT_TAGS]
    return [(int(h.box_tag.Z_MIN), int(job["num_modes"])), (int(h.box_tag.Z_MAX), int(job["num_modes"]))]


def wall_tags(h, job):
    if job["device"] == "file":
        return [int(t) for t in job["pec_tags"]]
    t = h.box_tag
    return [t.X_MIN, t.X_MAX, t.Y_MIN, t.Y_MAX]


def material_map(h, job, mesh, i):
    if job["device"] == "file":
        mm = h.MaterialMap(h.Material(1.0))
        for t, lst in job.get("tag_eps", {}).items():
            e = lst[i]
            mm.set(int(t), h.Material(complex(e[0], e[1])))
        return mm
    ec, el = job["eps_core"][i], job["eps_clad"][i]
    mm = h.MaterialMap(h.Material(complex(el[0], el[1])))
    mm.set(TAG_A, h.Material(complex(ec[0], ec[1])))
    mm.set(TAG_B, h.Material(complex(ec[0], ec[1])))
    return mm


def estimate(h, job, mesh):
    nd = h.NedelecDofMap3D(mesh, int(job["order"]))
    e = h.estimate_memory(nd)
    return dict(cells=int(mesh.num_cells), dofs=int(nd.num_dofs), bytes=int(e.total_bytes), text=e.describe(), human=h.format_bytes(int(e.total_bytes)))


def save_mesh(folder, mesh):
    np.savez_compressed(folder / "mesh_ports.npz", vertices=np.asarray(mesh.vertices), cells=np.asarray(mesh.cells), tags=np.asarray(mesh.cell_tags))


def section_modes(h, job, i, omega, count, refine=False):
    """beta (descending) of the cross-section modes (2D mode solver, PEC on the four sides): straight: the core; coupler: the two-core section. refine: half the cell
    and order p + 2 (at most 4): a reference that is nearly free of the discretisation error of the 3D problem."""
    xe, ye = edges_of(job)
    cell = job["cell_nm"] * NM * (0.5 if refine else 1.0)
    xs, ys = grid_lines(xe, cell), grid_lines(ye, cell)
    mesh = grid_rectangle(h, xs, ys)
    hgt = job["height_nm"] * NM
    cores = [(xe[1], xe[2], TAG_A)] if job["device"] == "straight" else [(xe[1], xe[2], TAG_A), (xe[3], xe[4], TAG_B)]
    for c, x in enumerate(np.asarray(mesh.cell_centroids)):
        tag = TAG_CLAD
        if 0.0 < x[1] < hgt:
            for a, b, t in cores:
                if a < x[0] < b:
                    tag = t
        mesh.set_cell_tag(c, tag)
    p = min(int(job["order"]) + 2, 4) if refine else int(job["order"])
    nd, h1 = h.NedelecDofMap2D(mesh, p), h.DofMap2D(mesh, p)
    setup = h.WaveguideSetup()
    setup.omega = omega
    setup.materials = material_map(h, job, mesh, i)
    t = h.box_tag
    setup.pec_tags = [t.X_MIN, t.X_MAX, t.Y_MIN, t.Y_MAX]
    setup.num_modes = count
    modes = h.PropagatingMode(nd, h1, setup).solve()
    return sorted((complex(m.beta).real for m in modes), reverse=True)


def field_planes(job, info, mesh):
    """Points of the two cut views: xz at y = mid height of the cores (z horizontal) and xy at z = L / 2, as (xs, zs / ys, pts_xz, pts_xy)."""
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    eps = 1e-6 * float(np.max(hi - lo))
    xs = np.linspace(lo[0] + eps, hi[0] - eps, GRID // 2 + 1)
    zs = np.linspace(lo[2] + eps, hi[2] - eps, GRID)
    ys = np.linspace(lo[1] + eps, hi[1] - eps, GRID // 2 + 1)
    y_mid = job["height_nm"] * NM / 2 if job["device"] != "file" else 0.5 * (lo[1] + hi[1])
    z_mid = 0.5 * (lo[2] + hi[2])
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pxz = np.stack([X.ravel(), np.full(X.size, y_mid), Z.ravel()], axis=1)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    pxy = np.stack([X2.ravel(), Y2.ravel(), np.full(X2.size, z_mid)], axis=1)
    return xs, zs, ys, pxz, X.shape, pxy, X2.shape


def run(h, job, folder, mesh, info, est, out, flush):
    p = int(job["order"])
    nd = h.NedelecDofMap3D(mesh, p)
    ports = port_tags(h, job)
    lams = job["wavelengths_nm"]
    coupler = job["device"] == "coupler"
    straight = job["device"] == "straight"
    L = info.get("length", 0.0)
    locator = h.PointLocator3D(mesh) if job.get("field", True) else None
    planes = field_planes(job, info, mesh) if locator is not None else None
    ex_port, ex_mode = int(job.get("excite_port", 0)), int(job.get("excite_mode", 0))
    for i, lam in enumerate(lams):
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return
        say(f"PROGRESS {i + 1}/{len(lams)} lambda = {lam:g} nm")
        t0 = time.time()
        pt = dict(index=i, lam_nm=float(lam), dofs=int(nd.num_dofs))
        try:
            omega = 2 * np.pi * h.constants.c0 / (lam * NM)

            def make_setup(incoming=None):
                s = h.ScatteringSetup3D()
                s.omega = omega
                s.materials = material_map(h, job, mesh, i)
                s.pec_tags = wall_tags(h, job)
                lst = []
                for k, (tag, nm) in enumerate(ports):
                    if incoming is not None and incoming[0] == k:
                        amp = [0.0] * nm
                        amp[min(incoming[1], nm - 1)] = 1.0
                        lst.append(h.WaveguidePort(tag, nm, amp))
                    else:
                        lst.append(h.WaveguidePort(tag, nm))
                s.ports = lst
                return s

            setup = make_setup()
            probe = h.Scattering3D(nd, setup)
            port_modes = {}
            for k in range(len(ports)):
                pm = probe.port_modes(k)
                port_modes[str(k)] = [dict(n_eff=float(complex(m.effective_index).real), propagating=bool(m.propagating)) for m in pm.modes]
            say(f"  ports: {port_modes}")
            sp = h.s_parameters(nd, setup)
            ch = [dict(port=int(c.port), mode=int(c.mode), beta=float(complex(c.beta).real)) for c in sp.channels]
            S = np.asarray(sp.s)
            cands = [k for k, c in enumerate(ch) if c["port"] == ex_port and c["mode"] == ex_mode]
            c0 = cands[0] if cands else 0
            col = np.abs(S[:, c0]) ** 2
            refl = float(sum(col[k] for k, c in enumerate(ch) if c["port"] == ch[c0]["port"]))
            trans = float(sum(col[k] for k, c in enumerate(ch) if c["port"] != ch[c0]["port"]))
            pt.update(channels=ch, S_re=S.real.tolist(), S_im=S.imag.tolist(), port_modes=port_modes, R=refl, T=trans, loss=float(1.0 - refl - trans),
                      n_eff0=float(port_modes["0"][0]["n_eff"]) if port_modes.get("0") else float("nan"), excitation=ch[c0])
            if straight:
                beta = section_modes(h, job, i, omega, 1)[0]
                tcol = [k for k, c in enumerate(ch) if c["port"] != ch[c0]["port"] and c["mode"] == ch[c0]["mode"]]
                if tcol:
                    pt["phase_error"] = float(abs(np.angle(S[tcol[0], c0] * np.exp(1j * beta * L))))
                pt["beta_2d"] = float(beta)
                pt["n_eff_2d"] = float(beta / (omega / h.constants.c0))
                if job.get("reference", True):
                    pt["n_eff_ref"] = float(section_modes(h, job, i, omega, 1, True)[0] / (omega / h.constants.c0))
            if coupler:
                be, bo = section_modes(h, job, i, omega, 2)[:2]
                kappa = 0.5 * (be - bo)
                pt.update(beta_even=float(be), beta_odd=float(bo), kappa=float(kappa), cross_cmt=float(np.sin(kappa * L) ** 2), coupling_length_nm=float(np.pi / (2 * kappa) / NM))
                if job.get("reference", True):
                    re_, ro_ = section_modes(h, job, i, omega, 2, True)[:2]
                    kr = 0.5 * (re_ - ro_)
                    pt.update(cross_cmt_ref=float(np.sin(kr * L) ** 2), coupling_length_ref_nm=float(np.pi / (2 * kr) / NM), n_eff_even_ref=float(re_ / (omega / h.constants.c0)),
                              n_eff_odd_ref=float(ro_ / (omega / h.constants.c0)))
                if job["style"] == "half":
                    # ports 0, 1 at z = 0 (core A, core B), 2, 3 at z = L
                    idx = {(c["port"], c["mode"]): k for k, c in enumerate(ch)}
                    if (2, 0) in idx and (3, 0) in idx and (0, 0) in idx:
                        pt.update(bar=float(abs(S[idx[(2, 0)], idx[(0, 0)]]) ** 2), cross=float(abs(S[idx[(3, 0)], idx[(0, 0)]]) ** 2),
                                  back=float(abs(S[idx[(1, 0)], idx[(0, 0)]]) ** 2))
                else:
                    idx = {(c["port"], c["mode"]): k for k, c in enumerate(ch)}
                    if all(key in idx for key in ((0, 0), (0, 1), (1, 0), (1, 1))):
                        te, to = S[idx[(1, 0)], idx[(0, 0)]], S[idx[(1, 1)], idx[(0, 1)]]
                        cross = float(np.sin(np.angle(te * np.conj(to)) / 2) ** 2)
                        pt.update(bar=1.0 - cross, cross=cross, back=float(abs(S[idx[(0, 1)], idx[(0, 0)]]) ** 2), t_even=[te.real, te.imag], t_odd=[to.real, to.imag],
                                  phase_error_even=float(abs(np.angle(te * np.exp(-1j * be * L)))), phase_error_odd=float(abs(np.angle(to * np.exp(-1j * bo * L)))))
            if locator is not None and (i == 0 or job.get("field_all")):
                prob = h.Scattering3D(nd, make_setup((ch[c0]["port"], ch[c0]["mode"])))
                sol = prob.solve()
                xs, zs, ys, pxz, sxz, pxy, sxy = planes
                fields = {}
                for key, pts, shp in (("xz", pxz, sxz), ("xy", pxy, sxy)):
                    vals, cells = prob.sample(sol, locator, pts)
                    e2 = np.sum(np.abs(np.asarray(vals)) ** 2, axis=1)
                    e2[np.asarray(cells) < 0] = np.nan
                    fields[key] = e2.reshape(shp)
                np.savez_compressed(folder / f"field_{i}.npz", xs=xs, ys=ys, zs=zs, xz=fields["xz"], xy=fields["xy"])
                pt["field_file"] = f"field_{i}.npz"
            pt["seconds"] = time.time() - t0
        except Exception as exc:
            pt["error"] = f"{type(exc).__name__}: {exc}"
            say(f"ERROR wavelength {lam:g} nm: {pt['error']}")
            say(traceback.format_exc())
        out["points"].append(pt)
        flush()


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    if len(argv) > 1 and argv[1] == "inspect":
        try:
            print("INSPECT" + json.dumps(inspect_mesh(hpfem, job["path"], job["scale"])), flush=True)
            return 0
        except Exception as exc:
            say(f"ERROR inspect: {type(exc).__name__}: {exc}")
            return 1
    try:
        mesh, info = build_mesh(hpfem, job)
        est = estimate(hpfem, job, mesh)
        save_mesh(folder, mesh)
    except Exception as exc:
        say(f"ERROR mesh: {type(exc).__name__}: {exc}")
        return 1
    if len(argv) > 1 and argv[1] == "estimate":
        print("ESTIMATE" + json.dumps(est), flush=True)
        return 0
    say(f"hpfem {getattr(hpfem, '__version__', '?')}, mesh: {est['cells']} cells, {est['dofs']} unknowns, memory estimate {est['human']}: {est['text']}")
    if est["bytes"] > float(job.get("max_gb", 16.0)) * 1e9:
        say(f"ERROR memory estimate {est['human']} exceeds the limit of {job.get('max_gb', 16.0)} GB: reduce the mesh or the order (or raise the limit).")
        return 1
    out = dict(meta=dict(name=job.get("name", ""), task="ports", device=job["device"], started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         hpfem=str(getattr(hpfem, "__version__", "?")), order=int(job["order"]), **est), points=[], modes=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    flush()
    try:
        run(hpfem, job, folder, mesh, info, est, out, flush)
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        say(traceback.format_exc())
        return 1
    flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
