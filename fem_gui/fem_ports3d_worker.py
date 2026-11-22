"""Worker for waveguides and ports in 3D (hp-FEM 0.5): runs in the Python that has hpfem, called by the app.

    python fem_ports3d_worker.py <job folder> [estimate | inspect]

Reads <folder>/job.json (fem_ports3d.job). Devices: ``straight`` (one core), ``coupler`` (two cores), both on the structured tetrahedral mesh of hpfem.box with PEC
side walls and modal ports on the end faces z = 0, z = L; ``file`` (Gmsh mesh of the user: physical surfaces PEC / ports, physical volumes materials).
Per wavelength: hpfem.s_parameters (power-normalised S-matrix of all propagating channels), reflection / transmission of the excited channel, n_eff of the port
modes, for ``straight`` the phase of the transmission against beta L of the 2D mode solver, for ``coupler`` the coupled-mode reference
P_cross = sin^2(kappa L) from the supermodes of the cross-section, and (optional) |E|^2 of the excitation on two cut planes.
``estimate``: only the mesh and hpfem.estimate_memory (``ESTIMATE{json}``); ``inspect``: tags of a mesh file (``INSPECT{json}``).

Writes <folder>/results.json after every wavelength and log lines on stdout (PROGRESS i/n, ERROR ..., CANCELLED, DONE). Cancellation: file <folder>/cancel.
Device ``ring``: the add-drop ring resonator (fem_ring3d_mesh.py, Gmsh): launch at the input port, through / drop / add / reflection per wavelength, the pole fit, the isolated
ring as the axisymmetric-cavity reference and the FSR estimate from the group index of the bus; ``results.json`` has an extra entry ``ring``.
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


def build_mesh(h, job, folder=None):
    """(mesh, info) in SI units."""
    if job["device"] == "ring":
        return ring_mesh(h, job, folder)
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


def estimate(h, job, mesh, orders=None):
    nd = h.NedelecDofMap3D(mesh, list(orders) if orders is not None else int(job["order"]))
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


# ===================================================================================================================== add-drop ring resonator
def fit_pole(wavelengths, responses):
    """Common complex pole of several responses near one resonance (port of ``fit_pole`` of examples/ring_resonator_3d/run.py), S_k(λ) = A_k / (λ − λ_p) + B_k + C_k (λ − λ_c):
    multiplied by (λ − λ_p) the model is linear in (A_k − B_k q, B_k − C_k q, C_k, q) with q = λ_p − λ_c, solved by least squares (exact for data of the model). Returns λ_p
    [unit of λ] and the relative residual; needs at least four wavelengths. Re λ_p is the resonance wavelength, 2 |Im λ_p| the FWHM, Re λ_p / (2 |Im λ_p|) the loaded Q."""
    lam = np.asarray(wavelengths, dtype=float)
    lc = lam.mean()
    t = lam - lc
    Sm = np.asarray(responses, dtype=complex).reshape(-1, len(lam))
    k, n = Sm.shape
    rows = np.zeros((k * n, 3 * k + 1), dtype=complex)
    rhs = np.zeros(k * n, dtype=complex)
    for j in range(k):                                   # S t = a + b t + c t^2 + q S
        rows[j * n:(j + 1) * n, 3 * j:3 * j + 3] = np.stack([np.ones(n), t, t ** 2], axis=1)
        rows[j * n:(j + 1) * n, -1] = Sm[j]
        rhs[j * n:(j + 1) * n] = Sm[j] * t
    x, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    residual = float(np.linalg.norm(rows @ x - rhs) / np.linalg.norm(rhs))
    return complex(x[-1] + lc), residual


def ring_fit(points, previous=None, window=None):
    """Pole fit of through and drop amplitudes of the points (dicts with lam_nm, s); with ``window`` (in FWHM of the previous fit) only the points near it. Returns a dict or
    None (fewer than four points)."""
    pts = sorted((p for p in points if "s" in p), key=lambda p: p["lam_nm"])
    if window is not None and previous:
        near = [p for p in pts if abs(p["lam_nm"] - previous["wavelength"]) <= window * previous["fwhm"]]
        pts = near if len(near) >= 6 else pts
    if len(pts) < 4:
        return None
    lam = [p["lam_nm"] for p in pts]
    pole, residual = fit_pole(lam, [[complex(*p["s"][1]) for p in pts], [complex(*p["s"][2]) for p in pts]])
    fwhm = 2 * abs(pole.imag)
    return dict(wavelength=float(pole.real), fwhm=float(fwhm), q=float(pole.real / fwhm) if fwhm > 0 else float("inf"), residual=residual, points=len(pts),
                max_drop=float(max(p["power"][2] for p in pts)), min_through=float(min(p["power"][1] for p in pts)))


def ring_geometry(job):
    import fem_ring3d_mesh as rm

    return rm.geometry(radius=job["radius_nm"], width=job["width_nm"], height=job["height_nm"], gap=job["gap_nm"], margin_x=job["margin_x_nm"], margin_y=job["margin_y_nm"],
                       margin_z=job["margin_z_nm"], cell_core=job["cell_core_nm"], cell_far=job["cell_far_nm"], grade=job["grade_nm"], half=bool(job["half"]))


def ring_mesh(h, job, folder=None):
    """(mesh, info): Gmsh mesh of the ring (fem_ring3d_mesh.build), the end faces split at y = 0 into the four ports, per-cell orders (order_core within ``reach_nm`` of a core)."""
    import tempfile

    import fem_ring3d_mesh as rm

    g = ring_geometry(job)
    if folder is not None:
        path = str(Path(folder) / "ring.msh")
    else:
        fd, path = tempfile.mkstemp(suffix=".msh")
        import os
        os.close(fd)
    t0 = time.time()
    rm.build(g, path)
    say(f"Gmsh: ring mesh in {time.time() - t0:.1f} s")
    mesh = h.read_gmsh(path, NM, 3)
    for face, (below, above) in ((rm.TAG_LEFT, (rm.PORT_TAGS[0], rm.PORT_TAGS[2])), (rm.TAG_RIGHT, (rm.PORT_TAGS[1], rm.PORT_TAGS[3]))):
        for f in mesh.facets_with_tag(face):
            y = np.mean([mesh.vertex(int(v))[1] for v in mesh.facet_vertices(int(f))])
            mesh.set_facet_tag(int(f), below if y < 0 else above)
    d = rm.core_distance(g, np.asarray(mesh.cell_centroids) / NM)
    orders = np.where(d < float(job["reach_nm"]), int(job["order_core"]), int(job["order_far"])).astype(int)
    return mesh, dict(kind="ring", orders=orders.tolist(), geometry=g, port_tags=list(rm.PORT_TAGS), wall=rm.TAG_WALL, symmetry=rm.TAG_SYMMETRY, half=bool(job["half"]))


def ring_materials(h, core, clad):
    mm = h.MaterialMap(h.Material(complex(clad[0], clad[1])))
    mm.set(TAG_CLAD, h.Material(complex(clad[0], clad[1])))
    mm.set(TAG_A, h.Material(complex(core[0], core[1])))
    return mm


def solver_backend(h, name):
    """DirectSolverBackend of the name (auto: None = the library default); error if the library was not built with it."""
    name = str(name).lower()
    if name in ("", "auto"):
        return None
    table = {"mumps": "MUMPS", "cudss": "CUDSS", "sparselu": "SPARSE_LU"}
    member = getattr(h.DirectSolverBackend, table[name])
    if member not in h.available_backends():
        raise RuntimeError(f"solver {name} is not available in this hpfem build (available: {[str(b) for b in h.available_backends()]})")
    return member


def ring_solve(h, nd, job, info, lam, core, clad, solver, launch=True):
    """Launch at port 1 at wavelength lam [nm]: the power-normalised complex amplitudes of the four ports (through = port 2, drop = port 3, add = port 4, reflection = port 1);
    returns (point dict, problem, solution)."""
    pt = info["port_tags"]
    setup = h.ScatteringSetup3D()
    setup.omega = 2 * np.pi * h.constants.c0 / (lam * NM)
    setup.materials = ring_materials(h, core, clad)
    setup.pec_tags = [info["wall"]]
    if solver is not None:
        setup.solver = solver
    pmc = [info["symmetry"]] if info["half"] else []
    setup.ports = [h.WaveguidePort(pt[0], 1, [1.0], pmc)] + [h.WaveguidePort(t, 1, [], pmc) for t in pt[1:]]
    t0 = time.time()
    problem = h.Scattering3D(nd, setup)
    solution = problem.solve()
    amplitudes = problem.port_coefficients(solution)
    modes = [problem.port_modes(i).modes[0] for i in range(4)]
    a_in = amplitudes[0].incoming[0] * np.sqrt(modes[0].power)
    s = [a.outgoing[0] * np.sqrt(m.power) / a_in for a, m in zip(amplitudes, modes)]
    power = [float(abs(v) ** 2) for v in s]
    point = dict(lam_nm=float(lam), s=[[float(v.real), float(v.imag)] for v in s], power=power, power_sum=float(sum(power)), R=power[0], through=power[1], drop=power[2], add=power[3],
                 T=float(power[1] + power[2] + power[3]), loss=float(1.0 - sum(power)), n_eff0=float(complex(modes[0].beta).real / setup.omega * h.constants.c0),
                 seconds=float(time.time() - t0), dofs=int(nd.num_dofs))
    return point, problem, solution


def ring_section_beta(h, job, lam, core, clad, p, cell_nm):
    """beta [1/m] of the fundamental mode of the isolated bus cross-section (width x height in cladding, PEC box with the margins of the ring model) from the 2D mode solver."""
    w, hg = job["width_nm"] * NM, job["height_nm"] * NM
    my, mz = job["margin_y_nm"] * NM, job["margin_z_nm"] * NM
    xe, ye = [-w / 2 - my, -w / 2, w / 2, w / 2 + my], [-mz, 0.0, hg, hg + mz]
    cell = cell_nm * NM
    mesh = grid_rectangle(h, grid_lines(xe, cell), grid_lines(ye, cell))
    for c, x in enumerate(np.asarray(mesh.cell_centroids)):
        mesh.set_cell_tag(c, TAG_A if (0.0 < x[1] < hg and abs(x[0]) < w / 2) else TAG_CLAD)
    nd, h1 = h.NedelecDofMap2D(mesh, int(p)), h.DofMap2D(mesh, int(p))
    setup = h.WaveguideSetup()
    setup.omega = 2 * np.pi * h.constants.c0 / (lam * NM)
    setup.materials = ring_materials(h, core, clad)
    t = h.box_tag
    setup.pec_tags = [t.X_MIN, t.X_MAX, t.Y_MIN, t.Y_MAX]
    setup.num_modes = 1
    return max(complex(m.beta).real for m in h.PropagatingMode(nd, h1, setup).solve())


def ring_group_index(h, job):
    """(n_eff, n_g) of the bus cross-section at the target wavelength from the 2D mode solver (n_g = c dβ/dω by a central difference over ±5 nm; the material enters through
    ``job['ng']``, so its dispersion is included)."""
    ng = job["ng"]
    p, cell = int(job["ref_p"]), float(job["ref_cell_nm"])
    c0 = float(h.constants.c0)
    beta = [ring_section_beta(h, job, lam, core, clad, p, cell) for lam, core, clad in zip(ng["lams"], ng["core"], ng["clad"])]
    omega = [2 * np.pi * c0 / (lam * NM) for lam in ng["lams"]]
    n_eff = beta[1] * c0 / omega[1]
    n_g = c0 * (beta[2] - beta[0]) / (omega[2] - omega[0])
    return float(n_eff), float(n_g)


def _graded(edges, cell):
    lines = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, round((b - a) / cell))
        lines.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(lines)


def ring_reference(h, job, orders, core, clad):
    """Resonance wavelengths [nm] of the fundamental TE-like mode of the isolated ring for the azimuthal orders (port of ``ring_reference`` of the library example): the meridian
    section (r, z) from the axis to the PEC wall at the bus distance, the same cladding margins in z, in µm units (the gauged eigensolver works on k0^2 in mesh units). The
    lowest mode of each order is the TE-like one. p and cell [nm] from the job (p = 4, 50 nm: 0.01 nm for the library ring, README of the example)."""
    g = ring_geometry(job)
    import fem_ring3d_mesh as rm

    p, cell = int(job["ref_p"]), float(job["ref_cell_nm"])
    um = 1e-3
    r0, r1 = g["radius"] - g["width"] / 2, g["radius"] + g["width"] / 2
    r_wall = rm.extent(g)[2]
    xs = np.concatenate([_graded([0.0, r0 - 1000.0], 4 * cell)[:-1], _graded([r0 - 1000.0, r0, r1, r_wall], cell)])
    ys = _graded([-g["margin_z"], 0.0, g["height"], g["height"] + g["margin_z"]], cell)
    mesh = h.rectangle(len(xs) - 1, len(ys) - 1, [xs[0] * um, ys[0] * um], [xs[-1] * um, ys[-1] * um])
    dx, dy = (xs[-1] - xs[0]) / (len(xs) - 1), (ys[-1] - ys[0]) / (len(ys) - 1)
    for v in range(mesh.num_vertices):                   # the uniform index grid onto the graded lines
        x = np.asarray(mesh.vertex(v)) / um
        i, j = round((x[0] - xs[0]) / dx), round((x[1] - ys[0]) / dy)
        mesh.set_vertex(v, [xs[i] * um, ys[j] * um])
    for c, x in enumerate(np.asarray(mesh.cell_centroids) / um):
        mesh.set_cell_tag(c, TAG_A if r0 < x[0] < r1 and 0 < x[1] < g["height"] else TAG_CLAD)
    nd, h1 = h.NedelecDofMap2D(mesh, p), h.DofMap2D(mesh, p)
    setup = h.AxisymmetricCavitySetup()
    setup.materials = ring_materials(h, core, clad)
    setup.axis_tag = h.box_tag.X_MIN
    setup.pec_tags = [h.box_tag.X_MAX, h.box_tag.Y_MIN, h.box_tag.Y_MAX]
    setup.num_modes = 1
    setup.krylov_dimension = 30
    out = {}
    for m in orders:
        setup.azimuthal_order = int(m)
        mode = h.AxisymmetricCavity(nd, h1, setup).solve()[0]
        out[int(m)] = float(2 * np.pi / mode.wavenumber / um)
    return out


def ring_field(h, nd, mesh, job, info, problem, solution, folder, name):
    """|E|^2 of the launch solution on the x-y plane through the mid-height of the cores (z = h/2; the half model starts there: 10 nm above) and save it."""
    g = info["geometry"]
    import fem_ring3d_mesh as rm

    bus_y, X, Y = rm.extent(g)
    z = (g["height"] / 2 + (10.0 if info["half"] else 0.0)) * NM
    nx = int(min(220, max(40, round(2 * X / 40.0))))
    ny = int(min(220, max(40, round(2 * Y / 40.0))))
    xs = np.linspace(-X * NM * 0.999, X * NM * 0.999, nx)
    ys = np.linspace(-Y * NM * 0.999, Y * NM * 0.999, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)], axis=1)
    vals, cells = problem.sample(solution, h.PointLocator3D(mesh), pts)
    e2 = np.sum(np.abs(np.asarray(vals)) ** 2, axis=1)
    e2[np.asarray(cells) < 0] = np.nan
    np.savez_compressed(Path(folder) / name, xs=xs, ys=ys, e2=e2.reshape(XX.shape))
    return name


def run_ring(h, job, folder, mesh, info, est, out, flush):
    """Reference, group index, wavelength scan (manual list or automatic search around the reference resonance), pole fit, optional field at the strongest drop."""
    nd = h.NedelecDofMap3D(mesh, list(info["orders"]))
    solver = solver_backend(h, job.get("backend", "auto"))
    g = info["geometry"]
    L_nm = 2 * np.pi * g["radius"]
    target = float(job["target_nm"])
    ng_core, ng_clad = job["ng"]["core"][1], job["ng"]["clad"][1]
    ring = dict(radius_nm=g["radius"], length_nm=float(L_nm), target_nm=target, scan="auto" if job.get("auto") else "manual", half=bool(info["half"]), reference={}, fit=None,
                chosen_m=None, lam_ref_nm=None, fsr_ref_nm=None, n_eff_bus=None, n_g=None, fsr_ng_nm=None)
    out["ring"] = ring
    flush()

    def cancelled():
        if (folder / "cancel").exists():
            say("CANCELLED")
            out["cancelled"] = True
            flush()
            return True
        return False

    total = 18 if job.get("auto") else len(job["wavelengths_nm"])
    done = [0]
    # --- group index of the bus and the free spectral range estimate
    try:
        n_eff, n_g = ring_group_index(h, job)
        ring.update(n_eff_bus=n_eff, n_g=n_g, fsr_ng_nm=float(target ** 2 / (n_g * L_nm)))
        say(f"bus cross-section at {target:g} nm: n_eff {n_eff:.4f}, n_g {n_g:.4f}; FSR = lambda^2 / (n_g L) = {ring['fsr_ng_nm']:.2f} nm (L = 2 pi R = {L_nm / 1000:.3f} um)")
        flush()
    except Exception as exc:
        say(f"WARN group index: {type(exc).__name__}: {exc}")
        n_eff = None
    if cancelled():
        return
    # --- reference: the isolated ring
    centre = target
    if job.get("reference", True):
        try:
            m0 = int(round(2 * np.pi * g["radius"] * (n_eff if n_eff else 2.3) / target))
            ref = ring_reference(h, job, (m0 - 1, m0, m0 + 1), ng_core, ng_clad)
            ring["reference"] = {str(m): lam for m, lam in ref.items()}
            m_sel = min(ref, key=lambda m: abs(ref[m] - target))
            ring.update(chosen_m=int(m_sel), lam_ref_nm=ref[m_sel])
            if m_sel + 1 in ref:
                ring["fsr_ref_nm"] = float(ref[m_sel] - ref[m_sel + 1])
            elif m_sel - 1 in ref:
                ring["fsr_ref_nm"] = float(ref[m_sel - 1] - ref[m_sel])
            centre = ref[m_sel]
            fsr = f"; FSR {ring['fsr_ref_nm']:.2f} nm" if ring["fsr_ref_nm"] else ""
            say("isolated ring (axisymmetric cavity): " + ", ".join(f"m = {m}: {ref[m]:.3f} nm" for m in sorted(ref)) + fsr)
            flush()
        except Exception as exc:
            say(f"WARN reference: {type(exc).__name__}: {exc}")
    if cancelled():
        return

    def solve_at(lam, k):
        idx = 0 if job.get("frozen") else k
        core, clad = job["eps_core"][min(idx, len(job["eps_core"]) - 1)], job["eps_clad"][min(idx, len(job["eps_clad"]) - 1)]
        done[0] += 1
        say(f"PROGRESS {min(done[0], total)}/{total} lambda = {lam:.4f} nm")
        pt, prob, sol = ring_solve(h, nd, job, info, lam, core, clad, solver)
        pt["index"] = done[0] - 1
        pt["eps_index"] = idx
        out["points"].append(pt)
        ring["fit"] = ring_fit(out["points"], ring["fit"]) or ring["fit"]
        say(f"  through {pt['through']:.5f}, drop {pt['drop']:.5f}, reflection {pt['R']:.1e}, add {pt['add']:.1e}, sum {pt['power_sum']:.6f} ({pt['seconds']:.0f} s)")
        flush()
        return pt

    try:
        if not job.get("auto"):
            for k, lam in enumerate(job["wavelengths_nm"]):
                if cancelled():
                    return
                solve_at(float(lam), k)
            ring["fit"] = ring_fit(out["points"])
        else:
            search = float(job["search_nm"])
            for lam in centre + np.linspace(-search, search, 9):
                if cancelled():
                    return
                solve_at(float(lam), 0)
            fit = ring_fit(out["points"])
            ring["fit"] = fit
            if fit is None or abs(fit["wavelength"] - centre) > search or fit["fwhm"] > search:
                say("no resonance found within the search window: widen the search window or check the mesh (p = 1 gives spurious resonances)")
                say(f"PROGRESS {total}/{total} no resonance in the window")
            else:
                for lam in fit["wavelength"] + fit["fwhm"] * np.linspace(-2.0, 2.0, 9):
                    if cancelled():
                        return
                    solve_at(float(lam), 0)
                ring["fit"] = ring_fit(out["points"], fit, window=3.0)
        flush()
        if job.get("field_res") and out["points"]:
            best = max(out["points"], key=lambda p: p["drop"])
            idx = best.get("eps_index", 0)
            core, clad = job["eps_core"][min(idx, len(job["eps_core"]) - 1)], job["eps_clad"][min(idx, len(job["eps_clad"]) - 1)]
            say(f"field at the strongest drop, lambda = {best['lam_nm']:.4f} nm")
            pt, prob, sol = ring_solve(h, nd, job, info, best["lam_nm"], core, clad, solver)
            best["field_file"] = ring_field(h, nd, mesh, job, info, prob, sol, folder, "field_ring.npz")
            flush()
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        say(traceback.format_exc())
        out["points"].append(dict(lam_nm=float("nan"), error=f"{type(exc).__name__}: {exc}"))
        flush()
        return


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
        mesh, info = build_mesh(hpfem, job, folder)
        est = estimate(hpfem, job, mesh, info.get("orders"))
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
        (run_ring if job["device"] == "ring" else run)(hpfem, job, folder, mesh, info, est, out, flush)
    except Exception as exc:
        say(f"ERROR {type(exc).__name__}: {exc}")
        say(traceback.format_exc())
        return 1
    flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
