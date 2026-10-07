"""Solver worker of the FEM model builder: runs in the Python that has `hpfem` (MSYS2), called by the app.

    PYTHONPATH=python python fem_worker.py <job folder>

Reads <job>/job.json and <job>/mesh.msh (Gmsh 4.1, written by the app), solves the periodic scattering problem with hpfem.ConicalScattering
(TE, TM, any angle theta and azimuth phi, layered background, Bloch periodic in x, PML above (and below), PEC walls) for every sweep value
and writes <job>/results.json (after every point) and <job>/maps_<i>.npz for the requested points.

Frame of the solver and of the maps: x along the period, y vertical (normal of the layers), z along the invariant direction (the lines
of a grating); E = (E_x, E_y, E_z). Plane wave of unit amplitude |E0| = 1 V/m, phase 1 at x = 0, y = top of the layer stack.

Engines (job["solver"]["engine"]):
  conical    hpfem.ConicalScattering: TE, TM, any azimuth; uniform mesh, polynomial order p.
  inplane    hpfem.Scattering2D (in-plane TM only): uniform mesh, order p.
  inplane_hp as inplane with hp-adaptive refinement per sweep value (estimate, Doerfler marking, hp decision by prediction, hp_refine),
             as in the notebook 02_hp_adaptivity_lshape and run_R3_adaptive.py; cells at the periodic faces and in the PML are never marked.

Progress lines for the app:  PROGRESS i/n ...   STEP s/S ...   WARN ...   ERROR ...
"""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_materials as fm  # noqa: E402

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 101, 102, 103, 104
Z0 = 376.730313668


def say(*a):
    print(*a, flush=True)


def ndofs(obj):
    try:
        return int(obj.num_dofs)
    except Exception:
        return 0


def snap(values_m, targets_m):
    """Nearest mesh coordinate for every target (the stack interfaces must lie on mesh lines exactly)."""
    v = np.unique(np.round(values_m, 15))
    return [float(v[np.argmin(abs(v - t))]) for t in targets_m]


class Case:
    """Everything that does not change along the sweep: mesh, locator, snapped coordinates."""

    def __init__(self, hpfem, job, folder):
        self.h = hpfem
        self.job = job
        self.model = job["model"]
        self.names = list(self.model["materials"])
        self.P = self.model["period_nm"] * 1e-9
        lay = job["layout"]
        self.lay = lay
        t0 = time.time()
        self.mesh = hpfem.read_gmsh(str(Path(folder) / "mesh.msh"), 1e-9, 2)
        self.locator = hpfem.PointLocator2D(self.mesh)
        ys = np.asarray(self.mesh.vertices)[:, 1]
        self.interfaces = snap(ys, [y * 1e-9 for y in lay["interfaces"]])
        self.y_phys_top, self.y_phys_bottom = snap(ys, [lay["y_cover_top"] * 1e-9, lay["y_sub_bottom"] * 1e-9])
        self.pml_top = float(lay["y_pml_top"] * 1e-9 - self.y_phys_top)
        self.pml_bottom = float(self.y_phys_bottom - lay["y_pml_bottom"] * 1e-9) if lay["pml_bottom"] else 0.0
        self.y_line = (lay["y_struct_top"] + 0.5 * (lay["y_cover_top"] - lay["y_struct_top"])) * 1e-9
        self.y_trans = (lay["interfaces"][-1] - 0.5 * (lay["interfaces"][-1] - lay["y_sub_bottom"])) * 1e-9
        say(f"mesh read in {time.time() - t0:.1f} s: {self.mesh.num_cells} cells")
        try:
            bad, nl, nr = periodic_defects(self.mesh, self.P)
            if bad:
                say(f"WARN the periodic sides of the mesh differ ({nl} / {nr} facets): Bloch constraints will fail")
        except Exception:
            pass
        self.spaces = {}
        self.reuse = None                                          # (mesh, orders) of an adaptive run, reused along a sweep

    def dofmaps(self, order):
        if order not in self.spaces:
            self.spaces[order] = (self.h.NedelecDofMap2D(self.mesh, order), self.h.DofMap2D(self.mesh, order))
        return self.spaces[order]


# ------------------------------------------------------------------------------------------------------------ common parts
def prepare(case, lam_nm):
    """Materials, layer stack and wavelength quantities shared by all engines."""
    h, model = case.h, case.model
    lam = lam_nm * 1e-9
    k0 = 2 * np.pi / lam
    eps = {n: fm.eps_at(model["materials"][n], lam_nm) for n in case.names}
    mat = {n: h.Material(complex(e)) for n, e in eps.items()}
    e_cover, e_sub = eps[model["cover"]], eps[model["substrate"]]
    if abs(e_cover.imag) > 1e-9 or e_cover.real < 1:
        raise ValueError("Einfallsmedium nicht verlustfrei")
    layers = [h.Layer(mat[l["material"]], case.interfaces[i] - case.interfaces[i + 1]) for i, l in enumerate(model["layers"])]
    stack = h.LayerStack2D(mat[model["cover"]], layers, mat[model["substrate"]], case.interfaces[0])
    return SimpleNamespace(lam_nm=lam_nm, k0=k0, omega=k0 * h.constants.c0, eps=eps, mat=mat, stack=stack, e_sub=e_sub,
                           n_cover=float(np.sqrt(e_cover).real), n_sub=np.sqrt(e_sub + 0j), lossless_sub=abs(e_sub.imag) < 1e-9)


def design_pml(case, ctx, theta, kx0, beta, solver):
    """PML profile for the largest angle of the propagating orders of the cover; reference index of the box."""
    h, P = case.h, case.P
    k = ctx.k0 * ctx.n_cover
    cover_orders = [m for m in range(-8, 9) if (kx0 + 2 * np.pi * m / P) ** 2 + beta ** 2 < k ** 2]
    psi = max(np.degrees(np.arccos(np.sqrt(max(k ** 2 - (kx0 + 2 * np.pi * m / P) ** 2 - beta ** 2, 0.0)) / k)) for m in cover_orders) if cover_orders else theta
    psi = min(max(psi, theta, 0.0), solver.get("pml_angle_cap_deg", 80.0))
    try:
        profile = h.PmlProfile.for_angle(psi, solver["pml_target"], 1.0, 2)
    except Exception:
        profile = h.PmlProfile(2, solver.get("pml_reflection", 1e-12))
    n_ref = min(ctx.n_cover, float(ctx.n_sub.real)) if case.pml_bottom > 0 and ctx.lossless_sub else ctx.n_cover
    box = h.PmlBox2D([0.0, case.y_phys_bottom], [case.P, case.y_phys_top], [0.0, 0.0, case.pml_bottom, case.pml_top], ctx.k0, n_ref, profile)
    return box, profile, psi


def set_backend(setup, solver):
    try:
        setup.solver = getattr(type(setup.solver), solver.get("backend", "AUTO"))
    except Exception:
        pass
    for key in ("extra_quadrature_order", "pml_extra_quadrature_order"):
        if solver.get(key) is not None:
            try:
                setattr(setup, key, int(solver[key]))
            except Exception:
                pass


def fourier_orders(field, y, index, kx0, k0, P, kn_inc, npts, max_order):
    """{m: efficiency} of the propagating orders of a (2- or 3-component) field on the line at height y (own implementation for the in-plane
    solver; the conical engine uses the library functions). eta_m = Re(kn_m) |A_m|^2 / kn_inc."""
    xs = (np.arange(npts) + 0.5) * P / npts
    vals = np.array([np.asarray(field(np.array([x, y])), dtype=complex) for x in xs])
    out = {}
    for m in range(-max_order, max_order + 1):
        kxm = kx0 + 2 * np.pi * m / P
        kn2 = (k0 * index) ** 2 - kxm ** 2
        if kn2 > 0:
            a = (vals * np.exp(-1j * kxm * xs)[:, None]).mean(axis=0)
            out[m] = float(np.sqrt(kn2) / kn_inc * np.vdot(a, a).real)
    return out


# ------------------------------------------------------------------------------------------------------------------ engines
class Run:
    """One solved problem: field access, incident field and reflected / transmitted orders, independent of the engine."""

    def total(self, x):
        raise NotImplementedError


class ConicalRun(Run):
    def __init__(self, case, ctx, theta, phi, pol, order, solver):
        h = case.h
        wave = h.layered_conical_wave(ctx.stack, ctx.k0, np.radians(theta), np.radians(phi), h.Polarisation.S if pol == "TE" else h.Polarisation.P)
        box, profile, psi = design_pml(case, ctx, theta, wave.kx, wave.beta, solver)
        setup = h.ConicalScatteringSetup()
        setup.omega, setup.beta = ctx.omega, wave.beta
        for i, n in enumerate(case.names):
            setup.materials.set(1 + i, ctx.mat[n])
        setup.background, setup.incident, setup.pml = ctx.stack, wave.field, box
        setup.pec_tags = [TAG_BOTTOM, TAG_TOP]
        shift = [case.P, 0.0]
        setup.periodic = [h.PeriodicPair2D(TAG_LEFT, TAG_RIGHT, shift, h.bloch_phase([wave.kx, 0.0], shift))]
        set_backend(setup, solver)
        nd, h1 = case.dofmaps(order)
        t0 = time.time()
        self.problem = h.ConicalScattering(nd, h1, setup)
        self.solution = self.problem.solve()
        self.t_solve = time.time() - t0
        try:
            self.dofs = int(len(self.problem.free_dofs))
        except Exception:
            self.dofs = ndofs(nd) + ndofs(h1)
        self.h, self.case, self.ctx, self.wave, self.locator, self.mesh = h, case, ctx, wave, case.locator, case.mesh
        self.profile, self.psi, self.order, self.max_order = profile, psi, order, order
        self.kx, self.beta = wave.kx, wave.beta
        self.ref_R, self.ref_T = float(wave.reflectance), float(wave.transmittance)

    def total(self, x):
        return np.asarray(self.problem.total_field(self.solution, self.locator, x), dtype=complex)

    def reflected(self, x):
        i = self.wave.incident(x)
        return self.total(x) - np.array([i[0], i[1], 1j * i[2]])

    def orders(self, solver):
        h, case, ctx, w = self.h, self.case, self.ctx, self.wave
        maxo, pts = solver.get("orders_max", 3), solver.get("order_points", 64)
        co = h.conical_fourier_coefficients(self.reflected, [0.0, case.y_line], [1.0, 0.0], case.P, w.kx, maxo, pts)
        R = {int(o.order): float(o.efficiency) for o in h.conical_diffraction_efficiencies(co, ctx.k0, ctx.n_cover, case.P, w.kx, w.beta, w.ky, 1.0)
             if o.propagating}
        T = {}
        if ctx.lossless_sub:
            ct = h.conical_fourier_coefficients(self.total, [0.0, case.y_trans], [1.0, 0.0], case.P, w.kx, maxo, pts)
            T = {int(o.order): float(o.efficiency) for o in
                 h.conical_diffraction_efficiencies(ct, ctx.k0, float(ctx.n_sub.real), case.P, w.kx, w.beta, w.ky, 1.0) if o.propagating}
        return R, T


class InplaneRun(Run):
    """hpfem.Scattering2D (TM in the plane) on any mesh with a list of orders per cell."""

    def __init__(self, case, ctx, theta, mesh, orders, solver):
        h = case.h
        kz, kx = ctx.k0 * ctx.n_cover * np.cos(np.radians(theta)), ctx.k0 * ctx.n_cover * np.sin(np.radians(theta))
        wave = ctx.stack.plane_wave(ctx.k0, np.radians(theta), h.Polarisation.P)
        box, profile, psi = design_pml(case, ctx, theta, kx, 0.0, solver)
        setup = h.ScatteringSetup2D()
        setup.omega = ctx.omega
        for i, n in enumerate(case.names):
            setup.materials.set(1 + i, ctx.mat[n])
        setup.background, setup.incident, setup.pml = ctx.stack, wave.field, box
        setup.formulation = h.Formulation.SCATTERED_FIELD
        setup.pec_tags = [TAG_BOTTOM, TAG_TOP]
        shift = np.array([case.P, 0.0])
        k = np.array([kx, -kz])
        setup.periodic = [h.PeriodicPair2D(TAG_LEFT, TAG_RIGHT, shift, h.bloch_phase(k, shift))]
        set_backend(setup, solver)
        self.dofmap = h.NedelecDofMap2D(mesh, orders)
        t0 = time.time()
        self.problem = h.Scattering2D(self.dofmap, setup)
        self.solution = self.problem.solve()
        self.t_solve = time.time() - t0
        self.dofs = int(self.dofmap.num_dofs)
        try:
            self.max_order = int(self.dofmap.max_order)
        except Exception:
            self.max_order = int(np.max(orders)) if np.ndim(orders) else int(orders)
        self.h, self.case, self.ctx, self.wave, self.mesh = h, case, ctx, wave, mesh
        self.locator = h.PointLocator2D(mesh)
        self.profile, self.psi, self.kx, self.kz, self.beta = profile, psi, kx, kz, 0.0
        self.ref_R, self.ref_T = float(wave.reflectance), float(wave.transmittance)
        self._d_inc = None

    def total(self, x):
        e = np.asarray(self.problem.total_field(self.solution, self.locator, x), dtype=complex)
        return np.array([e[0], e[1], 0.0])

    def incident(self, x):
        try:
            e = np.asarray(self.wave.incident_wave.value(x), dtype=complex)
            return np.array([e[0], e[1], 0.0])
        except Exception:                                          # fit D exp(-i kz y) + U exp(+i kz y) at two heights in the cover
            if self._d_inc is None:
                y1, y2 = self.case.y_line, 0.5 * (self.case.y_line + self.case.y_phys_top)
                f = [np.asarray(self.wave.field.value(np.array([0.0, y])), dtype=complex) for y in (y1, y2)]
                a = np.array([[np.exp(-1j * self.kz * y1), np.exp(1j * self.kz * y1)], [np.exp(-1j * self.kz * y2), np.exp(1j * self.kz * y2)]])
                self._d_inc = np.linalg.solve(a, np.stack(f))[0]
            e = self._d_inc * np.exp(1j * (self.kx * x[0] - self.kz * x[1]))
            return np.array([e[0], e[1], 0.0])

    def orders(self, solver):
        case, ctx = self.case, self.ctx
        maxo, pts = solver.get("orders_max", 3), max(solver.get("order_points", 64), 64)
        R = fourier_orders(lambda x: self.total(x) - self.incident(x), case.y_line, ctx.n_cover, self.kx, ctx.k0, case.P, self.kz, pts, maxo)
        T = {}
        if ctx.lossless_sub:
            T = fourier_orders(self.total, case.y_trans, float(ctx.n_sub.real), self.kx, ctx.k0, case.P, self.kz, pts, maxo)
        return R, T


def make_result(case, ctx, run, theta, phi, pol, order, extra=None):
    R_orders, T_orders = run.orders(case_solver[0])
    R = sum(R_orders.values())
    T = sum(T_orders.values()) if ctx.lossless_sub else None
    res = dict(lam_nm=ctx.lam_nm, theta=theta, phi=phi, pol=pol, order=order, dofs=run.dofs, time_s=run.t_solve, R_orders=R_orders, T_orders=T_orders,
               R=R, T=T, A=(1.0 - R - T) if T is not None else None, A_incl_substrate=1.0 - R, ref_R=run.ref_R, ref_T=run.ref_T,
               pml_psi_deg=run.psi, pml_R0=float(run.profile.reflection), eps={n: [float(e.real), float(e.imag)] for n, e in ctx.eps.items()},
               omega=ctx.omega, n_cover=ctx.n_cover, S_inc=0.5 * ctx.n_cover * np.cos(np.radians(theta)) / Z0)
    res.update(extra or {})
    return res


case_solver = [None]                                                # the solver settings of the running job (read by make_result)


def sample_field(run, pts):
    """E at the points (n, 2) [m] in one call (hpfem M15 F3: parallel point location and evaluation in C++, points on an interface in the cell above).
    Returns (n, 3) complex, rows of points outside the mesh NaN; None if this hpfem has no `sample`."""
    try:
        values, _ = run.problem.sample(run.solution, run.locator, np.ascontiguousarray(pts, dtype=float), interface_side=+1)
    except (AttributeError, TypeError):
        return None
    v = np.asarray(values, dtype=complex)
    return np.column_stack([v, np.zeros(len(v))]) if v.shape[1] == 2 else v


def field_map(case, ctx, run, res, theta, phi, pol, want):
    nx, ny = want["nx"], want["ny"]
    P = case.P
    xs = np.linspace(0.0, P, nx)
    ys = np.linspace(case.y_phys_bottom, case.y_phys_top, ny)
    X, Y = np.meshgrid(np.clip(xs, 1e-12, P - 1e-12), ys + 1e-12, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    t0 = time.time()
    flat = sample_field(run, pts)
    how = "sample"
    if flat is None:
        flat, how = np.full((len(pts), 3), np.nan, dtype=complex), "loop"
    missing = np.where(np.isnan(flat[:, 0]))[0]
    for k in missing:                                              # old hpfem without sample, or points the vectorised search did not find
        flat[k] = run.total(pts[k])
    if how == "sample" and len(missing):
        say(f"  WARN {len(missing)} map points were outside the mesh for sample() and were evaluated one by one")
    say(f"  field map {nx} x {ny} by {how} in {time.time() - t0:.1f} s")
    E = flat.reshape(nx, ny, 3)
    return dict(x_nm=xs * 1e9, y_nm=ys * 1e9, E=E.astype(np.complex64), omega=ctx.omega, lam_nm=ctx.lam_nm, S_inc=res["S_inc"],
                eps_names=np.array(case.names), eps=np.array([complex(*res["eps"][n]) for n in case.names]), theta=theta, phi=phi, pol=pol,
                kx=float(run.kx), period_nm=P * 1e9)


def triangulation_data(case, ctx, run, res, theta, phi, pol, subdivisions):
    """The field on the subdivided mesh (hpfem M15 F3 `triangulate`): exact element representation, discontinuities across material boundaries kept,
    material tag per simplex. Only the physical region (the PML cells are cut off). None if this hpfem has no `triangulate`."""
    try:
        tri = run.problem.triangulate(run.solution, subdivisions=int(subdivisions))
    except (AttributeError, TypeError):
        return None
    pts, simp = np.asarray(tri.points, dtype=float), np.asarray(tri.simplices, dtype=int)
    vals, tag = np.asarray(tri.values, dtype=complex), np.asarray(tri.tag, dtype=int)
    if vals.shape[1] == 2:
        vals = np.column_stack([vals, np.zeros(len(vals))])
    tol = 1e-9 * case.P
    y = pts[:, 1]
    keep = (y[simp] >= case.y_phys_bottom - tol).all(axis=1) & (y[simp] <= case.y_phys_top + tol).all(axis=1)
    simp, tag = simp[keep], tag[keep]
    used = np.unique(simp)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return dict(points_nm=pts[used] * 1e9, simplices=remap[simp].astype(np.int32), values=vals[used].astype(np.complex64), tag=tag.astype(np.int32),
                omega=ctx.omega, lam_nm=ctx.lam_nm, S_inc=res["S_inc"], eps_names=np.array(case.names),
                eps=np.array([complex(*res["eps"][n]) for n in case.names]), theta=theta, phi=phi, pol=pol, kx=float(run.kx), period_nm=case.P * 1e9)


def absorbed_exact(case, ctx, run, res):
    """Absorbed power per material from the volume quadrature of the Joule heating on the mesh (hpfem M15 F4), relative to the incident power per
    period, cells of the physical region only (the PML cells carry the stretched field). None if this hpfem has no `absorbed_power_by_tag`."""
    try:
        ap = run.h.absorbed_power_by_tag(run.problem, run.solution)
    except (AttributeError, TypeError):
        return None
    per_cell = np.asarray(ap.per_cell, dtype=float)
    V, C = np.asarray(run.mesh.vertices), np.asarray(run.mesh.cells)
    cy = V[C][:, :, 1].mean(axis=1)
    phys = (cy > case.y_phys_bottom) & (cy < case.y_phys_top)
    tags = np.array([int(run.mesh.cell_tag(k)) for k in range(len(C))])
    norm = res["S_inc"] * case.P
    by = {}
    for t in np.unique(tags[per_cell != 0]):
        p = float(per_cell[phys & (tags == t)].sum())
        if p != 0.0 and 1 <= t <= len(case.names):
            by[case.names[t - 1]] = p / norm
    return dict(by_material=by, total=float(per_cell[phys].sum() / norm), total_all=float(ap.total / norm))


def exclusion_mask(vertices, cells, period, y_top_inner, y_bottom_inner):
    """True for cells that must not be refined: touching the periodic faces x = 0 or x = P (the two sides must stay identical) or inside a PML."""
    v, c = np.asarray(vertices), np.asarray(cells)
    vx, vy = v[c][:, :, 0], v[c][:, :, 1]
    tol = 1e-9 * period
    touching = (vx.min(axis=1) <= tol) | (vx.max(axis=1) >= period - tol)
    centre = vy.mean(axis=1)
    return touching | (centre > y_top_inner - tol) | (centre < y_bottom_inner + tol)


def periodic_defects(mesh, period):
    """Cells that must be refined so that the two periodic sides match: the cells of boundary facets that have no identical partner on the other
    side and contain finer facets of the other side (the coarser side). Also returns the number of facets on the left and right side.
    hpfem requires both sides to be meshed identically (bloch_constraints)."""
    import bisect
    V = np.asarray(mesh.vertices)

    def facets(tag):
        out = {}
        for f in mesh.facets_with_tag(tag):
            a, b = mesh.facet_vertices(f)
            y0, y1 = sorted((float(V[a][1]), float(V[b][1])))
            out[(round(y0 * 1e12), round(y1 * 1e12))] = int(f)
        return out

    left, right = facets(TAG_LEFT), facets(TAG_RIGHT)
    only_l = {k: f for k, f in left.items() if k not in right}
    only_r = {k: f for k, f in right.items() if k not in left}

    def coarser(coarse, fine):
        starts = sorted(fine)
        out = set()
        for (c0, c1), f in coarse.items():
            i = bisect.bisect_left(starts, (c0 - 1, -(10 ** 18)))
            if i < len(starts) and starts[i][1] <= c1 + 1:           # a facet of the other side lies inside this one
                out.add(int(mesh.facet_cells(f)[0]))
        return out

    cells = coarser(only_l, only_r) | coarser(only_r, only_l)
    if not cells and (only_l or only_r):                             # not nested (should not happen): refine every unmatched cell
        cells = {int(mesh.facet_cells(f)[0]) for f in list(only_l.values()) + list(only_r.values())}
    return sorted(cells), len(left), len(right)


class PeriodicMismatch(RuntimeError):
    """The periodic sides of the refined mesh could not be made identical."""


def symmetrise_periodic(h, adaptive, orders, period, max_iter=8):
    """Refines the cells at the periodic faces whose facet is coarser than its partner on the other side, until both sides are identical. The
    1-irregularity rule of the refinement can split a boundary cell next to a marked cell on one side only; the cells at the faces are never
    marked themselves. Gives up (PeriodicMismatch) if the number of unmatched facets keeps growing or after max_iter rounds.
    Returns (orders, number of repair rounds)."""
    rounds, previous, counts = 0, None, None
    for _ in range(max_iter):
        bad, nl, nr = periodic_defects(adaptive.mesh, period)
        if not bad:
            return orders, rounds
        if previous is not None and len(bad) >= 2 * previous:
            raise PeriodicMismatch(f"the repair of the periodic sides diverges ({nl} / {nr} facets)")
        if counts == (nl, nr):
            raise PeriodicMismatch(f"the repair of the periodic sides makes no progress ({nl} / {nr} facets)")
        previous, counts = len(bad), (nl, nr)
        say(f"  periodic sides differ ({nl} / {nr} facets): refining {len(bad)} boundary cell(s) on the coarser side")
        hp = h.hp_refine(adaptive, orders.tolist(), bad, [])
        orders = np.asarray(hp.orders, dtype=int)
        rounds += 1
    raise PeriodicMismatch(f"the periodic sides did not match after {max_iter} repair rounds")


def adaptive_point(case, ctx, theta, solver, ad):
    """hp-adaptive loop at one sweep value. Returns (run, step records, final mesh, final orders)."""
    h = case.h
    adaptive = h.AdaptiveMesh2D(case.mesh)
    try:                                                              # hpfem >= M15 F16 stage 1: refinement is mirrored across the Bloch faces
        adaptive.set_periodic([(TAG_LEFT, TAG_RIGHT, np.array([case.P, 0.0]))])
        say("  periodic faces declared in the adaptive mesh (AdaptiveMesh.set_periodic)")
    except AttributeError:                                            # older build: symmetrise_periodic repairs the faces after every step
        say("  NOTE: this hpfem has no AdaptiveMesh.set_periodic; the periodic faces are matched by repair rounds (update hpfem to M15 F16)")
    orders = np.full(case.mesh.num_cells, int(ad["p0"]), dtype=int)
    predicted = np.zeros(0)
    steps, stable, prev = [], 0, None
    n_steps = int(ad["steps"])
    for step in range(n_steps):
        mesh = adaptive.mesh
        run = InplaneRun(case, ctx, theta, mesh, orders.tolist(), solver)
        R_o, T_o = run.orders(solver)
        R = sum(R_o.values())
        T = sum(T_o.values()) if ctx.lossless_sub else None
        estimate = run.problem.estimate(run.solution)
        ind = np.asarray(estimate.indicators, dtype=float)
        eta = float(estimate.total())
        change = None
        if prev is not None:
            change = max(abs(R - prev[0]), abs(T - prev[1]) if T is not None and prev[1] is not None else 0.0)
            stable = stable + 1 if (ad.get("tol", 0) and change < ad["tol"]) else 0
        prev = (R, T)
        rec = dict(step=step + 1, cells=int(mesh.num_cells), dofs=run.dofs, max_p=int(run.max_order), R=R, T=T, eta=eta, change=change,
                   R_orders=R_o, T_orders=T_o, time_s=run.t_solve)
        steps.append(rec)
        say(f"STEP {step + 1}/{n_steps} cells {rec['cells']} DoFs {run.dofs} p_max {rec['max_p']} R {R:.6f}" + (f" T {T:.6f}" if T is not None else "") +
            f" eta {eta:.3e}" + (f" change {change:.2e}" if change is not None else ""))
        if stable >= 2:
            say(f"  stop: change of R and T below {ad['tol']:g} in two consecutive steps")
            break
        if run.dofs > ad["max_dofs"]:
            say(f"  stop: more than {ad['max_dofs']} DoFs")
            break
        if step == n_steps - 1:
            break
        mask = exclusion_mask(mesh.vertices, mesh.cells, case.P, case.y_phys_top, case.y_phys_bottom)
        marked = h.dorfler_marking(np.where(mask, 0.0, ind), float(ad["dorfler"]))
        decision = h.hp_decide_by_prediction(estimate.indicators, predicted, marked)
        old_orders = orders
        hp = h.hp_refine(adaptive, old_orders.tolist(), decision.h_marked, decision.p_marked)
        try:
            orders, repaired = symmetrise_periodic(h, adaptive, np.asarray(hp.orders, dtype=int), case.P)
        except PeriodicMismatch as exc:
            say(f"  stop: {exc}; the result of the previous step is kept")
            steps[-1]["stopped"] = str(exc)
            orders = old_orders
            break
        predicted = h.predict_indicators(estimate.indicators, old_orders.tolist(), hp) if repaired == 0 else np.zeros(0)
    return run, steps, mesh, orders


def hp_mesh_data(mesh, orders):
    """Final hp mesh for the app: vertex coordinates in nm, triangles (corner indices), polynomial order and tag per cell."""
    v = np.asarray(mesh.vertices) * 1e9
    c = np.asarray(mesh.cells)
    try:
        tags = np.array([int(mesh.cell_tag(k)) for k in range(int(mesh.num_cells))])
    except Exception:
        tags = np.zeros(len(c), dtype=int)
    return dict(xy_nm=v[:, :2], tri=c[:, :3], order=np.asarray(orders, dtype=int), tag=tags)


def solve_point(case, lam_nm, theta, phi, pol, order, solver, want_map=None, first=False):
    """One frequency / angle: returns (result dict, map dict or None, hp mesh dict or None, triangulated field or None)."""
    case_solver[0] = solver
    engine = solver.get("engine", "conical")
    ctx = prepare(case, lam_nm)
    hpmesh = None
    if engine == "conical":
        run = ConicalRun(case, ctx, theta, phi, pol, order, solver)
        res = make_result(case, ctx, run, theta, phi, pol, order)
    else:
        if pol != "TM" or abs(phi) > 1e-9:
            raise ValueError("Der In-Ebenen-Löser rechnet nur TM mit φ = 0")
        if engine == "inplane_hp":
            ad = solver["adaptive"]
            if case.reuse is not None and ad.get("reuse_mesh") and not first:
                mesh, orders = case.reuse
                run = InplaneRun(case, ctx, theta, mesh, orders.tolist(), solver)
                steps = None
            else:
                run, steps, mesh, orders = adaptive_point(case, ctx, theta, solver, ad)
                case.reuse = (mesh, orders)
            res = make_result(case, ctx, run, theta, phi, pol, int(run.max_order), dict(adaptive=True, steps=steps or [], p_min=int(np.min(orders))))
            hpmesh = hp_mesh_data(mesh, orders)
        else:
            run = InplaneRun(case, ctx, theta, case.mesh, int(order), solver)
            res = make_result(case, ctx, run, theta, phi, pol, order)
    res["absorbed"] = absorbed_exact(case, ctx, run, res)
    mp = field_map(case, ctx, run, res, theta, phi, pol, want_map) if want_map else None
    tri = triangulation_data(case, ctx, run, res, theta, phi, pol, want_map.get("subdivisions", 3)) if (want_map and want_map.get("triangulate")) else None
    return res, mp, hpmesh, tri


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    case = Case(hpfem, job, folder)
    inc, sw, solver = job["incidence"], job["sweep"], job["solver"]
    values = sw["values"] if sw["mode"] != "none" else [None]
    n = len(values)
    maps = job.get("maps", {"enabled": False})
    map_idx = set(maps.get("indices", [])) if maps.get("enabled") else set()
    engine = solver.get("engine", "conical")
    out = dict(meta=dict(model=job["model"].get("name", ""), mode=sw["mode"], order=solver["order"], pol=inc["pol"], cells=int(case.mesh.num_cells),
                         engine=engine, started=time.strftime("%Y-%m-%d %H:%M:%S")), points=[], pscan=[])
    results_path = folder / "results.json"

    def flush():
        results_path.write_text(json.dumps(out, default=float), encoding="utf-8")

    pscan = job.get("pscan")
    if pscan:                                                      # convergence in the polynomial order at the first sweep value
        v = values[0]
        lam = v if sw["mode"] == "wavelength" else inc["wavelength_nm"]
        th = v if sw["mode"] == "theta" else inc["theta"]
        ph = v if sw["mode"] == "phi" else inc["phi"]
        for k, p in enumerate(pscan):
            say(f"PROGRESS {k + 1}/{len(pscan)} order p = {p}")
            try:
                res, _, _, _ = solve_point(case, lam, th, ph, inc["pol"], p, dict(solver, engine="conical" if engine == "conical" else "inplane"))
                out["pscan"].append(res)
                say(f"  p = {p}: {res['dofs']} DoFs, {res['time_s']:.1f} s, R = {res['R']:.6f}" + (f", T = {res['T']:.6f}" if res["T"] is not None else ""))
            except Exception as exc:
                say(f"ERROR p = {p}: {exc}")
                out["pscan"].append(dict(order=p, error=str(exc)))
            flush()
        return 0
    for i, v in enumerate(values):
        lam = v if sw["mode"] == "wavelength" else inc["wavelength_nm"]
        th = v if sw["mode"] == "theta" else inc["theta"]
        ph = v if sw["mode"] == "phi" else inc["phi"]
        say(f"PROGRESS {i + 1}/{n} lambda = {lam:.2f} nm, theta = {th:.2f}, phi = {ph:.2f}")
        try:
            res, mp, hpm, tri = solve_point(case, lam, th, ph, inc["pol"], solver["order"], solver, maps if i in map_idx else None, first=(i == 0))
            res["value"], res["index"] = v, i
            out["points"].append(res)
            if mp is not None:
                np.savez_compressed(folder / f"maps_{i}.npz", **mp)
            if tri is not None:
                np.savez_compressed(folder / f"tri_{i}.npz", **tri)
            if hpm is not None:
                np.savez_compressed(folder / f"hpmesh_{i}.npz", **hpm)
            msg = f"  {res['dofs']} DoFs, {res['time_s']:.1f} s: R = {res['R']:.6f}"
            if res["T"] is not None:
                msg += f", T = {res['T']:.6f}, A = {res['A']:.6f}"
            say(msg)
            if res.get("absorbed"):
                msg2 = ", ".join(f"{k} {v:.4f}" for k, v in res["absorbed"]["by_material"].items())
                say(f"  absorbed (exact, quadrature on the mesh): {msg2 or 'none'} | sum {res['absorbed']['total']:.4f}")
            if res["T"] is not None and abs(res["A"]) > 0 and res["A"] < -1e-3:
                say(f"WARN negative Absorption {res['A']:.2e}: Netz oder PML zu grob")
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(index=i, value=v, error=str(exc)))
        flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
