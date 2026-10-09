"""Solver worker of the FEM model builder: runs in the Python that has `hpfem` (MSYS2), called by the app.

    PYTHONPATH=python python fem_worker.py <job folder>

Reads <job>/job.json and <job>/mesh.msh (Gmsh 4.1, written by the app), solves the periodic scattering problem with hpfem.ConicalScattering
(TE, TM, any angle theta and azimuth phi, layered background, Bloch periodic in x, PML above (and below), PEC walls) for every sweep value
and writes <job>/results.json (after every point) and <job>/maps_<i>.npz for the requested points.

Frame of the solver and of the maps: x along the period, y vertical (normal of the layers), z along the invariant direction (the lines
of a grating); E = (E_x, E_y, E_z). Plane wave of unit amplitude |E0| = 1 V/m, phase 1 at x = 0, y = top of the layer stack.

Engines (job["solver"]["engine"]):
  grating     hpfem.grating.solve (hpfem >= 0.4, M15): the one-call grating API on the same conical solver with the library's diagnostics,
              the scalar E_z path (TE at phi = 0, a third of the DoFs), exact absorbed power, flux balance, timing per phase, progress
              and cooperative cancellation; optionally the Jacobian of the efficiencies (M16: d/d eps of the materials, d/d lambda, d/d theta,
              d/d phi) on the kept factorisation.
  grating_hp  as grating with hp-adaptive refinement per sweep value (residual estimator or goal-oriented DWR estimator of the
              specular order, hpfem >= 0.4 M15 F1/F16): TE, TM and conical incidence.
  conical     hpfem.ConicalScattering set up by this worker (older hpfem): TE, TM, any azimuth; uniform mesh, polynomial order p.
  inplane     hpfem.Scattering2D (in-plane TM only): uniform mesh, order p.
  isolated    an isolated (non-periodic) structure (model domain lateral = "pml"): hpfem.ConicalScattering with PML on all four sides and
              the layered background, no Bloch pairs (TE, TM, conical). Results: the scattered power through the closed measurement box
              (scattering width), in a homogeneous background also the absorption and extinction widths (conical_cross_sections) and the
              far-field pattern (ConicalFarField), the Mie series of a single circular cylinder as reference, and the energy flow of the
              total field through the detector segments of the model.
  inplane_hp  as inplane with hp-adaptive refinement per sweep value (estimate, Doerfler marking, hp decision by prediction, hp_refine),
              as in the notebook 02_hp_adaptivity_lshape and run_R3_adaptive.py; cells at the periodic faces and in the PML are never marked.

Tasks (job["task"]): "scattering" (default) or "resonances" (hpfem.grating.resonances / bands: complex eigenfrequencies of the open unit
cell near the wavelength of the incidence, at the Bloch wavenumber of the angle; a theta or phi sweep gives the band structure).

    python fem_worker.py <job folder> --check     diagnostics of the library and memory estimate without solving (writes check.json)

Progress lines for the app:  PROGRESS i/n ...   STEP s/S ...   PHASE name ...   DIAG severity code: text   WARN ...   ERROR ...   CANCELLED
Cancellation: the app creates <job>/cancel; the worker stops between the phases of a solve and between sweep points.
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

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 1, 2, 3, 4                  # hpfem.box_tag X_MIN, X_MAX, Y_MIN, Y_MAX (grating.solve)
Z0 = 376.730313668


def say(*a):
    print(*a, flush=True)


class Control:
    """Progress reports and cooperative cancellation of the running job (the app creates <job>/cancel)."""

    folder = None
    last_phase = None

    @classmethod
    def cancelled(cls):
        return cls.folder is not None and (Path(cls.folder) / "cancel").exists()

    @classmethod
    def progress(cls, event):
        phase = str(getattr(event, "phase", ""))
        phase = phase.split(".")[-1].lower()                     # enum or string
        if phase and phase != cls.last_phase:
            step, n = int(getattr(event, "step", 0) or 0), int(getattr(event, "num_steps", 0) or 0)
            say(f"PHASE {phase}" + (f" {step}/{n}" if n > 1 else "") + f" ({float(getattr(event, 'seconds', 0.0) or 0.0):.1f} s)")
            cls.last_phase = phase


def has_grating(hpfem):
    """True if this hpfem has the one-call grating API (0.4, M15 F2)."""
    try:
        import hpfem.grating  # noqa: F401
        return hasattr(hpfem.grating, "solve")
    except Exception:
        return False


def backend_enum(h, solver):
    name = solver.get("backend", "AUTO")
    if name in (None, "", "AUTO"):
        return None
    try:
        return getattr(h.DirectSolverBackend, name)
    except AttributeError:
        return None


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
        self.isolated = bool(lay.get("isolated"))
        self.pml_side = float(lay.get("pml_side", 0.0)) * 1e-9
        if not self.isolated:
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


class IsolatedRun(Run):
    """An isolated structure: hpfem.ConicalScattering with the layered background and PML on all four sides (no Bloch pairs); the walls behind
    the PML are PEC."""

    def __init__(self, case, ctx, theta, phi, pol, order, solver):
        h = case.h
        wave = h.layered_conical_wave(ctx.stack, ctx.k0, np.radians(theta), np.radians(phi), h.Polarisation.S if pol == "TE" else h.Polarisation.P)
        psi = min(float(solver.get("pml_angle_cap_deg", 80.0)), 85.0)          # an isolated scatterer radiates in every direction
        try:
            profile = h.PmlProfile.for_angle(psi, solver["pml_target"], 1.0, 2)
        except Exception:
            profile = h.PmlProfile(2, solver.get("pml_reflection", 1e-12))
        n_ref = min(ctx.n_cover, float(ctx.n_sub.real)) if case.pml_bottom > 0 and ctx.lossless_sub else ctx.n_cover
        box = h.PmlBox2D([0.0, case.y_phys_bottom], [case.P, case.y_phys_top], [case.pml_side, case.pml_side, case.pml_bottom, case.pml_top],
                         ctx.k0, n_ref, profile)
        setup = h.ConicalScatteringSetup()
        setup.omega, setup.beta = ctx.omega, wave.beta
        for i, n in enumerate(case.names):
            setup.materials.set(1 + i, ctx.mat[n])
        setup.background, setup.incident, setup.pml = ctx.stack, wave.field, box
        try:
            setup.incident_curl = wave.field_curl
        except Exception:
            pass
        setup.pec_tags = [TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP]
        setup.periodic = []
        set_backend(setup, solver)
        try:
            setup.progress = lambda ev: (Control.progress(ev), not Control.cancelled())[1]
        except Exception:
            pass
        nd, h1 = case.dofmaps(order)
        t0 = time.time()
        Control.last_phase = None
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


def _gauss_line(a, b, n):
    """n-point composite Gauss-Legendre rule (4 points per panel) on [a, b]: nodes, weights."""
    g, w = np.polynomial.legendre.leggauss(4)
    panels = max(int(np.ceil(n / 4)), 1)
    edges = np.linspace(a, b, panels + 1)
    xs, ws = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (e1 - e0) * g + 0.5 * (e0 + e1))
        ws.append(0.5 * (e1 - e0) * w)
    return np.concatenate(xs), np.concatenate(ws)


def line_flux(run, p0, p1, normal, scattered=False, n=400):
    """Energy flow (W per m along z) of the total (or scattered) field through the straight segment p0 -> p1 [m] along `normal` (time-averaged
    Poynting vector sampled by hpfem, M15 F12, composite Gauss rule). None if this hpfem cannot sample S."""
    p0, p1, normal = np.asarray(p0, float), np.asarray(p1, float), np.asarray(normal, float)
    L = float(np.linalg.norm(p1 - p0))
    t, w = _gauss_line(0.0, L, n)
    pts = p0[None, :] + np.outer(t / L, p1 - p0)
    pts = np.ascontiguousarray(pts)
    try:
        if scattered:                                                  # hpfem samples S of the total field only: S_sca from E_sca and H_sca
            Es, _ = run.problem.sample(run.solution, run.locator, pts, scattered=True, quantity="E")
            Ht, _ = run.problem.sample(run.solution, run.locator, pts, quantity="H")
            Hi = np.array([np.asarray(run.problem.incident_h_field(x), dtype=complex) for x in pts])
            Es, Hs = np.asarray(Es, dtype=complex), np.asarray(Ht, dtype=complex) - Hi
            S = 0.5 * np.cross(Es, np.conj(Hs)).real[:, :2]
        else:
            vals, _ = run.problem.sample(run.solution, run.locator, pts, quantity="S")
            S = np.asarray(vals, dtype=complex).real[:, :2]
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        say(f"WARN Fluss durch eine Linie nicht berechnet: {exc}")
        return None
    if np.isnan(S).any():
        return None
    return float(np.sum(w * (S @ normal)))


def box_surface_2d(case, box_nm):
    """The closed measurement box (mesh lines) as a Surface2D with outward normals (inner cells)."""
    h, mesh = case.h, case.mesh
    V = np.asarray(mesh.vertices)
    x0, x1, y0, y1 = (box_nm[k] * 1e-9 for k in ("x0", "x1", "y0", "y1"))
    tol = 1e-7 * max(abs(x1), abs(y0), abs(y1), 1e-9)
    cells = np.asarray(mesh.cells)
    centroid = V[cells][:, :, :2].mean(axis=1)
    facets = []
    for f in range(mesh.num_facets):
        v = V[mesh.facet_vertices(f)][:, :2]
        on = ((np.all(np.abs(v[:, 1] - y0) < tol) or np.all(np.abs(v[:, 1] - y1) < tol)) and np.all((v[:, 0] > x0 - tol) & (v[:, 0] < x1 + tol))) or \
             ((np.all(np.abs(v[:, 0] - x0) < tol) or np.all(np.abs(v[:, 0] - x1) < tol)) and np.all((v[:, 1] > y0 - tol) & (v[:, 1] < y1 + tol)))
        if not on:
            continue
        inside = [c for c in mesh.facet_cells(f) if c >= 0 and x0 < centroid[c][0] < x1 and y0 < centroid[c][1] < y1]
        if inside:
            facets.append(h.Surface2D.Facet(f, inside[0]))
    s_ = h.Surface2D()
    s_.facets = facets
    return s_


def homogeneous_background(case, ctx):
    model = case.model
    return not model["layers"] and abs(ctx.eps[model["cover"]] - ctx.eps[model["substrate"]]) < 1e-12 and case.pml_bottom > 0


def mie_cylinder(lam_nm, radius_nm, n_cyl, n_bg, pol):
    """Scattering, absorption and extinction widths [m] of a circular cylinder at normal incidence (Bohren & Huffman 8.4; e^{-i omega t},
    Im n > 0 lossy): pol "TE" = E along the axis (B&H case I, b_n), "TM" = H along the axis (case II, a_n)."""
    import scipy.special as sp

    k = 2 * np.pi * n_bg / (lam_nm * 1e-9)
    a = radius_nm * 1e-9
    x, m = k * a, complex(n_cyl) / n_bg
    nmax = int(x + 4 * x ** (1 / 3) + 10)
    n = np.arange(0, nmax + 1)
    J, dJ = sp.jv(n, x), sp.jvp(n, x)
    Jm, dJm = sp.jv(n, m * x), sp.jvp(n, m * x)
    H, dH = sp.hankel1(n, x), sp.h1vp(n, x)
    if pol == "TE":
        c = (Jm * dJ - m * dJm * J) / (Jm * dH - m * dJm * H)
    else:
        c = (m * dJ * Jm - J * dJm) / (m * Jm * dH - dJm * H)
    wgt = np.where(n == 0, 1.0, 2.0)
    q_sca = 2 / x * float(np.sum(wgt * np.abs(c) ** 2))
    q_ext = 2 / x * float(np.sum(wgt * c.real))
    return dict(sigma_sca=q_sca * 2 * a, sigma_ext=q_ext * 2 * a, sigma_abs=(q_ext - q_sca) * 2 * a)


def iso_result(case, ctx, run, theta, phi, pol, order):
    """Results of an isolated structure: scattering width (flux of the scattered field out of the measurement box), in a homogeneous background
    absorption and extinction widths and the far field, the Mie cylinder as reference, detector fluxes of the total field."""
    h, model, lay = case.h, case.model, case.lay
    Z0 = h.constants.Z0
    intensity = 0.5 * ctx.n_cover / Z0                                        # |E0| = 1 V/m in the cover
    th = np.radians(theta)
    res = dict(lam_nm=ctx.lam_nm, theta=theta, phi=phi, pol=pol, order=order, dofs=run.dofs, time_s=run.t_solve, isolated=True,
               R=None, T=None, A=None, R_orders={}, T_orders={}, ref_R=run.ref_R, ref_T=run.ref_T, omega=ctx.omega, n_cover=ctx.n_cover,
               eps={n: [float(e.real), float(e.imag)] for n, e in ctx.eps.items()}, S_inc=0.5 * ctx.n_cover * np.cos(th) / Z0,
               intensity=intensity, pml_R0=_reflection(run.profile))
    box = lay.get("box")
    if box:
        b = {k: v * 1e-9 for k, v in box.items()}
        sides = dict(top=line_flux(run, (b["x0"], b["y1"]), (b["x1"], b["y1"]), (0, 1), True),
                     bottom=line_flux(run, (b["x0"], b["y0"]), (b["x1"], b["y0"]), (0, -1), True),
                     left=line_flux(run, (b["x0"], b["y0"]), (b["x0"], b["y1"]), (-1, 0), True),
                     right=line_flux(run, (b["x1"], b["y0"]), (b["x1"], b["y1"]), (1, 0), True))
        if all(v is not None for v in sides.values()):
            res["P_sca_box"] = {k: float(v) for k, v in sides.items()}
            res["sigma_sca"] = float(sum(sides.values()) / intensity)
            res["sigma_sca_up"] = float(sides["top"] / intensity)
        if homogeneous_background(case, ctx):
            try:
                surf = box_surface_2d(case, box)
                cs = h.conical_cross_sections(run.problem, run.solution, surf)
                res.update(sigma_sca_lib=float(cs.scattering), sigma_abs=float(cs.absorption), sigma_ext=float(cs.extinction))
                ff = h.ConicalFarField(run.problem, run.solution, surf)
                phis = np.linspace(0.0, 2 * np.pi, 361)
                F = np.array([np.asarray(ff.pattern(float(a)), dtype=complex) for a in phis])
                ratio = float(ff.transverse_wavenumber) / float(ff.wavenumber)
                res["farfield_phi"] = phis.tolist()
                res["farfield_dsigma"] = (ratio * (np.abs(F) ** 2).sum(axis=1)).tolist()          # d sigma / d phi [m], |E0| = 1
            except Exception as exc:
                say(f"WARN Querschnitte der Bibliothek nicht berechnet: {exc}")
            circles = [s_ for s_ in model["shapes"] if s_["type"] == "circle"]
            if len(model["shapes"]) == 1 and circles and abs(phi) < 1e-9:
                try:
                    n_c = complex(np.sqrt(ctx.eps[circles[0]["material"]] + 0j))
                    res["mie"] = mie_cylinder(ctx.lam_nm, circles[0]["radius"], n_c, ctx.n_cover, pol)
                except Exception as exc:
                    say(f"WARN Mie-Referenz nicht berechnet: {exc}")
    dets = []
    inc_per_width = 0.5 * ctx.n_cover * np.cos(th) / Z0                         # incident power through a horizontal line per unit width
    for d in model.get("detectors", []):
        y, x0, x1 = d["y_nm"] * 1e-9, d["x0_nm"] * 1e-9, d["x1_nm"] * 1e-9
        f = line_flux(run, (x0, y), (x1, y), (0, -1), False)
        dets.append(dict(name=d.get("name", ""), y_nm=d["y_nm"], x0_nm=d["x0_nm"], x1_nm=d["x1_nm"], P_down=f,
                         normalised=(f / (inc_per_width * (x1 - x0)) if f is not None and inc_per_width > 0 else None)))
    res["detectors"] = dets
    return res


def grating_options(case, ctx, solver):
    """The keyword options of hpfem.grating.solve / validate / resonances for the layout of the app's mesh: the PML layers as meshed, the
    measurement lines between structure and PML."""
    bottom = "pml" if case.pml_bottom > 0 else "pec"
    pml = {"top": case.pml_top, "bottom": case.pml_bottom} if bottom == "pml" else {"top": case.pml_top}
    return dict(pml=pml, bottom=bottom, orders_max=int(solver.get("orders_max", 3)), pml_target=float(solver.get("pml_target", 1e-3)),
                cover_line=case.y_line, substrate_line=case.y_trans if (bottom == "pml" and ctx.lossless_sub) else None)


def diag_list(items):
    """hpfem Diagnostic objects as plain dicts."""
    return [dict(code=str(d.code), severity=str(d.severity), text=str(d.text), hint=str(getattr(d, "hint", "") or "")) for d in (items or [])]


class GratingRun(Run):
    """hpfem.grating.solve: the library's own grating front end on the conical solver (PML at the mesh layout, diagnostics, scalar E_z path,
    exact absorption, flux balance, progress and cancellation, kept factorisation for the Jacobian)."""

    def __init__(self, case, ctx, theta, phi, pol, order, solver, mesh=None, keep=False, check=True):
        import hpfem.grating as grating

        h = case.h
        self.mesh = mesh if mesh is not None else case.mesh
        materials = {1 + i: ctx.mat[n] for i, n in enumerate(case.names)}
        scalar = {"auto": "auto", "on": True, "off": False}.get(solver.get("scalar", "auto"), "auto")
        if keep and abs(phi) > 1e-12:
            scalar = False                                            # d/d phi needs the vector path
        opts = grating_options(case, ctx, solver)
        kwargs = dict(order=order, fourier_points=max(int(solver.get("order_points", 64)), 64), check=bool(check),
                      progress=Control.progress, cancel=Control.cancelled, scalar=scalar, keep_factorisation=bool(keep), solver=backend_enum(h, solver))
        if solver.get("extra_quadrature_order") is not None:
            kwargs["extra_quadrature_order"] = int(solver["extra_quadrature_order"])
        t0 = time.time()
        Control.last_phase = None
        self.result = r = grating.solve(self.mesh, materials, ctx.stack, "s" if pol == "TE" else "p", np.radians(theta), np.radians(phi), ctx.omega,
                                        **opts, **kwargs)
        self.t_solve = time.time() - t0
        self.h, self.case, self.ctx = h, case, ctx
        self.problem, self.solution, self.wave = r.problem, r.solution, r.wave
        self.locator = case.locator if mesh is None else h.PointLocator2D(self.mesh)
        self.dofs, self.order = int(r.dofs), order
        self.max_order = int(np.max(order)) if np.ndim(order) else int(order)
        self.kx, self.beta = float(r.wave.kx), float(r.wave.beta)
        self.ref_R, self.ref_T = float(r.wave.reflectance), float(r.wave.transmittance)
        self.profile, self.psi = getattr(r.pml, "profile", None), None
        self.materials = materials

    def total(self, x):
        return np.asarray(self.problem.total_field(self.solution, self.locator, x), dtype=complex)

    def orders(self, solver):
        r = self.result
        R = {int(o.m): float(o.efficiency) for o in r.R_orders if o.propagating}
        T = {int(o.m): float(o.efficiency) for o in r.T_orders if o.propagating} if self.ctx.lossless_sub else {}
        return R, T

    def extra(self):
        """What the library adds to the result of a point: exact absorption, balance, flux balance, timing, diagnostics, scalar path."""
        r, names = self.result, self.case.names
        out = dict(engine_lib="grating", scalar=bool(getattr(r, "scalar", False)), A_exact=float(r.A),
                   A_exact_by_material={names[t - 1]: float(v) for t, v in r.A_by_tag.items() if 1 <= t <= len(names) and abs(v) > 0},
                   balance_residual=float(r.power_balance_residual), timing={k: float(v) for k, v in dict(r.timing).items()},
                   diagnostics=diag_list(r.diagnostics))
        fb = getattr(r, "flux_balance", None)
        if fb:
            out["flux_balance"] = {k: float(v) for k, v in fb.items()}
        amps = {}
        for side, lst in (("R", r.R_orders), ("T", r.T_orders)):
            for o in lst:
                if o.propagating:
                    a = np.asarray(o.amplitude, dtype=complex)
                    amps[f"{side}{int(o.m)}"] = [[float(c.real), float(c.imag)] for c in a]
        out["amplitudes"] = amps
        return out


JAC_UNITS = {"wavelength": ("λ", 1e-9, "1/nm"), "theta": ("θ", np.pi / 180, "1/°"), "phi": ("φ", np.pi / 180, "1/°")}


def jacobian_data(case, run, model):
    """dR_m/dp and dT_m/dp of every propagating order by hpfem.grating.jacobian (M16 S1, one multi-RHS solve on the kept factorisation):
    p = Re eps and Im eps of every material inside the cell (not the incidence medium and the substrate, whose indices enter the
    post-processing), the wavelength, theta and, on the vector path, phi. Per nm and per degree; materials at fixed eps (no dispersion)."""
    import hpfem.grating as grating

    names = case.names
    inner = [n for n in names if n not in (model["cover"], model["substrate"]) and
             (any(l["material"] == n for l in model["layers"]) or any(s["material"] == n for s in model["shapes"]))]
    params = [("eps", 1 + names.index(n)) for n in inner] + ["wavelength", "theta"]
    if not run.result.scalar:
        params.append("phi")
    J, rows, cols = grating.jacobian(run.result, params)
    J = np.asarray(J, dtype=float)
    labels, units = [], []
    for j, c in enumerate(cols):
        c = str(c)
        if c.startswith("eps["):
            tag = int(c[4:c.index("]")])
            part = "Re" if c.endswith(".re") else "Im"
            labels.append(f"{part} ε({names[tag - 1]})")
            units.append("")
        else:
            sym, scale, unit = JAC_UNITS.get(c, (c, 1.0, ""))
            J[:, j] *= scale
            labels.append(sym)
            units.append(unit)
    row_labels = [f"{side}{int(m)}" for side, m in rows]
    return dict(rows=row_labels, cols=labels, units=units, J=J.tolist())


def make_result(case, ctx, run, theta, phi, pol, order, extra=None):
    R_orders, T_orders = run.orders(case_solver[0])
    R = sum(R_orders.values())
    T = sum(T_orders.values()) if ctx.lossless_sub else None
    res = dict(lam_nm=ctx.lam_nm, theta=theta, phi=phi, pol=pol, order=order, dofs=run.dofs, time_s=run.t_solve, R_orders=R_orders, T_orders=T_orders,
               R=R, T=T, A=(1.0 - R - T) if T is not None else None, A_incl_substrate=1.0 - R, ref_R=run.ref_R, ref_T=run.ref_T,
               pml_psi_deg=run.psi, pml_R0=_reflection(run.profile), eps={n: [float(e.real), float(e.imag)] for n, e in ctx.eps.items()},
               omega=ctx.omega, n_cover=ctx.n_cover, S_inc=0.5 * ctx.n_cover * np.cos(np.radians(theta)) / Z0)
    res.update(extra or {})
    return res


def _reflection(profile):
    try:
        return float(profile.reflection)
    except Exception:
        return None


case_solver = [None]                                                # the solver settings of the running job (read by make_result)


def sample_field(run, pts, quantity="E"):
    """E (or H [A/m], or the time-averaged Poynting vector S [W/m^2] with quantity "H" / "S", hpfem M15 F12) at the points (n, 2) [m] in one
    call (hpfem M15 F3: parallel point location and evaluation in C++, points on an interface in the cell above).
    Returns (n, 3) complex, rows of points outside the mesh NaN; None if this hpfem has no `sample` (or not for this quantity)."""
    try:
        if quantity == "E":
            values, _ = run.problem.sample(run.solution, run.locator, np.ascontiguousarray(pts, dtype=float), interface_side=+1)
        else:
            values, _ = run.problem.sample(run.solution, run.locator, np.ascontiguousarray(pts, dtype=float), interface_side=+1, quantity=quantity)
    except (AttributeError, TypeError, ValueError, RuntimeError):
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
    out = dict(x_nm=xs * 1e9, y_nm=ys * 1e9, E=E.astype(np.complex64), omega=ctx.omega, lam_nm=ctx.lam_nm, S_inc=res["S_inc"],
               eps_names=np.array(case.names), eps=np.array([complex(*res["eps"][n]) for n in case.names]), theta=theta, phi=phi, pol=pol,
               kx=float(run.kx), period_nm=P * 1e9)
    if how == "sample" and want.get("hs", True):                   # H and the Poynting vector (hpfem M15 F12), where the library has them
        for q in ("H", "S"):
            v = sample_field(run, pts, q)
            if v is not None and not np.isnan(v).all():
                out[q] = v.reshape(nx, ny, 3).astype(np.complex64)
    return out


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
    if case.isolated:
        x_ = pts[:, 0]
        keep &= (x_[simp] >= -tol).all(axis=1) & (x_[simp] <= case.P + tol).all(axis=1)
    simp, tag = simp[keep], tag[keep]
    used = np.unique(simp)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = dict(points_nm=pts[used] * 1e9, simplices=remap[simp].astype(np.int32), values=vals[used].astype(np.complex64), tag=tag.astype(np.int32),
               omega=ctx.omega, lam_nm=ctx.lam_nm, S_inc=res["S_inc"], eps_names=np.array(case.names),
               eps=np.array([complex(*res["eps"][n]) for n in case.names]), theta=theta, phi=phi, pol=pol, kx=float(run.kx), period_nm=case.P * 1e9)
    for q in ("H", "S"):                                            # same subdivision, same point order: H and S per point (hpfem M15 F12)
        try:
            tq = run.problem.triangulate(run.solution, subdivisions=int(subdivisions), quantity=q)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        vq = np.asarray(tq.values, dtype=complex)
        if vq.shape[0] == len(pts) and vq.ndim == 2 and vq.shape[1] == 3:
            out[f"values_{q}"] = vq[used].astype(np.complex64)
    return out


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
    if case.isolated:
        cx = V[C][:, :, 0].mean(axis=1)
        phys &= (cx > 0.0) & (cx < case.P)
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


def pml_mask(mesh, case):
    """True for the cells inside a PML (never marked: the estimator there measures the stretched field)."""
    v, c = np.asarray(mesh.vertices), np.asarray(mesh.cells)
    cy = v[c][:, :, 1].mean(axis=1)
    tol = 1e-9 * case.P
    return (cy > case.y_phys_top - tol) | (cy < case.y_phys_bottom + tol)


def goal_indicators(case, run, solver):
    """Goal-oriented (dual-weighted residual) indicators of the specular reflected order (hpfem M15 F1): the functional is the amplitude of
    order 0 on the measurement line along e = conj(A_0)/|A_0|, i.e. the linearised efficiency R_0. Returns (indicators, estimated error of
    the goal) or None if the order is not available."""
    h = case.h
    o0 = next((o for o in run.result.R_orders if int(o.m) == 0 and o.propagating), None)
    if o0 is None:
        return None
    a = np.asarray(o0.amplitude, dtype=complex)
    if np.linalg.norm(a) < 1e-12:
        return None
    e = np.conj(a) / np.linalg.norm(a)
    functional = h.conical_order_functional([0.0, case.y_line], [1.0, 0.0], case.P, float(run.kx), 0,
                                            max(int(solver.get("order_points", 64)), 64), e)
    g = h.conical_dwr_estimate(run.problem, run.solution, functional)
    return np.asarray(g.indicators, dtype=float), float(abs(g.error))


def adaptive_grating_point(case, ctx, theta, phi, pol, solver, ad):
    """hp-adaptive loop on hpfem.grating.solve at one sweep value (TE, TM, conical). The Bloch faces may be refined independently (hpfem
    M15 F16, non-matching periodic coupling); with AdaptiveMesh.set_periodic they are mirrored. Estimator "residual" (ConicalScattering.estimate)
    or "dwr" (goal: the specular order). Returns (run, step records, final mesh, final orders)."""
    h = case.h
    adaptive = h.AdaptiveMesh2D(case.mesh)
    if ad.get("mirror_periodic", True):
        try:
            adaptive.set_periodic([(TAG_LEFT, TAG_RIGHT, np.array([case.P, 0.0]))])
        except AttributeError:
            pass
    orders = np.full(case.mesh.num_cells, int(ad["p0"]), dtype=int)
    predicted = np.zeros(0)
    steps, stable, prev = [], 0, None
    n_steps = int(ad["steps"])
    estimator = ad.get("estimator", "residual")
    run = mesh = None
    for step in range(n_steps):
        if Control.cancelled():
            raise h.Cancelled("cancelled between adaptive steps")
        mesh = adaptive.mesh
        run = GratingRun(case, ctx, theta, phi, pol, orders.tolist(), solver, mesh=mesh, check=(step == 0))
        R_o, T_o = run.orders(solver)
        R = sum(R_o.values())
        T = sum(T_o.values()) if ctx.lossless_sub else None
        estimate = run.problem.estimate(run.solution)
        ind = np.asarray(estimate.indicators, dtype=float)
        eta = float(estimate.total())
        goal_err = None
        if estimator == "dwr":
            g = goal_indicators(case, run, solver)
            if g is not None:
                ind, goal_err = g
            elif step == 0:
                say("  WARN no specular order for the goal: residual estimator instead")
        change = None
        if prev is not None:
            change = max(abs(R - prev[0]), abs(T - prev[1]) if T is not None and prev[1] is not None else 0.0)
            stable = stable + 1 if (ad.get("tol", 0) and change < ad["tol"]) else 0
        prev = (R, T)
        rec = dict(step=step + 1, cells=int(mesh.num_cells), dofs=run.dofs, max_p=int(run.max_order), R=R, T=T, eta=eta, goal_error=goal_err, change=change,
                   R_orders=R_o, T_orders=T_o, time_s=run.t_solve, A_exact=float(run.result.A))
        steps.append(rec)
        say(f"STEP {step + 1}/{n_steps} cells {rec['cells']} DoFs {run.dofs} p_max {rec['max_p']} R {R:.6f}" + (f" T {T:.6f}" if T is not None else "") +
            f" eta {eta:.3e}" + (f" goal error {goal_err:.2e}" if goal_err is not None else "") + (f" change {change:.2e}" if change is not None else ""))
        if stable >= 2:
            say(f"  stop: change of R and T below {ad['tol']:g} in two consecutive steps")
            break
        if goal_err is not None and ad.get("tol", 0) and goal_err < ad["tol"]:
            say(f"  stop: estimated error of R_0 below {ad['tol']:g}")
            break
        if run.dofs > ad["max_dofs"] or step == n_steps - 1:
            if run.dofs > ad["max_dofs"]:
                say(f"  stop: more than {ad['max_dofs']} DoFs")
            break
        marked = h.dorfler_marking(np.where(pml_mask(mesh, case), 0.0, ind), float(ad["dorfler"]))
        decision = h.hp_decide_by_prediction(ind.tolist(), predicted, marked)
        old_orders = orders
        hp = h.hp_refine(adaptive, old_orders.tolist(), decision.h_marked, decision.p_marked)
        orders = np.asarray(hp.orders, dtype=int)
        predicted = h.predict_indicators(ind.tolist(), old_orders.tolist(), hp)
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
    if case.isolated or engine == "isolated":
        run = IsolatedRun(case, ctx, theta, phi, pol, order, solver)
        res = iso_result(case, ctx, run, theta, phi, pol, order)
    elif engine == "grating":
        keep = bool(solver.get("jacobian"))
        run = GratingRun(case, ctx, theta, phi, pol, order, solver, keep=keep, check=bool(solver.get("check", True)))
        res = make_result(case, ctx, run, theta, phi, pol, order, run.extra())
        if keep:
            try:
                res["jacobian"] = jacobian_data(case, run, case.model)
            except Exception as exc:
                say(f"WARN Ableitungen nicht berechnet: {exc}")
    elif engine == "grating_hp":
        ad = solver["adaptive"]
        if case.reuse is not None and ad.get("reuse_mesh") and not first:
            mesh, orders = case.reuse
            run = GratingRun(case, ctx, theta, phi, pol, orders.tolist(), solver, mesh=mesh, check=False)
            steps = None
        else:
            run, steps, mesh, orders = adaptive_grating_point(case, ctx, theta, phi, pol, solver, ad)
            case.reuse = (mesh, orders)
        res = make_result(case, ctx, run, theta, phi, pol, int(run.max_order),
                          dict(run.extra(), adaptive=True, steps=steps or [], p_min=int(np.min(orders)), estimator=ad.get("estimator", "residual")))
        hpmesh = hp_mesh_data(mesh, orders)
    elif engine == "conical":
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


# ---------------------------------------------------------------------------------------------------------------- resonances
def bloch_wavenumbers(ctx, theta, phi):
    """(kx, beta) of the plane wave from the incidence medium at theta, phi: the Bloch wavenumber along the period and the one along the lines."""
    k = ctx.k0 * ctx.n_cover * np.sin(np.radians(theta))
    return float(k * np.cos(np.radians(phi))), float(k * np.sin(np.radians(phi)))


def resonance_point(case, lam_nm, theta, phi, solver, ro, want_map=None):
    """Resonances of the open unit cell near lam_nm at the Bloch wavenumbers of (theta, phi) (hpfem.grating.resonances, M15 F14).
    Returns (result dict, {mode index: map dict})."""
    import hpfem.grating as grating

    h = case.h
    ctx = prepare(case, lam_nm)
    kx, beta = bloch_wavenumbers(ctx, theta, phi)
    opts = grating_options(case, ctx, solver)
    order = int(solver["order"])
    Control.last_phase = None
    r = grating.resonances(case.mesh, {1 + i: ctx.mat[n] for i, n in enumerate(case.names)}, ctx.stack, ctx.omega, kx=kx, beta=beta,
                           num_modes=int(ro.get("num_modes", 6)), order=order, pml=opts["pml"], bottom=opts["bottom"],
                           pml_target=opts["pml_target"], solver=backend_enum(h, solver), krylov_dimension=int(ro.get("krylov_dimension", 0)),
                           progress=Control.progress, cancel=Control.cancelled)
    modes = []
    for m in r.modes:
        modes.append(dict(m=int(m.index), omega=[float(m.omega.real), float(m.omega.imag)], lam_nm=float(m.wavelength) * 1e9, Q=float(m.Q),
                          residual=float(m.residual), beta=float(m.beta)))
    res = dict(lam_target_nm=lam_nm, theta=theta, phi=phi, kx=kx, beta=beta, kx_over_G=kx * case.P / (2 * np.pi), order=order, dofs=int(r.dofs),
               time_s=float(dict(r.timing).get("total", 0.0)), timing={k: float(v) for k, v in dict(r.timing).items()}, modes=modes,
               eps={n: [float(e.real), float(e.imag)] for n, e in ctx.eps.items()})
    maps = {}
    if want_map:
        nx, ny = want_map["nx"], want_map["ny"]
        xs = np.linspace(0.0, case.P, nx)
        ys = np.linspace(case.y_phys_bottom, case.y_phys_top, ny)
        X, Y = np.meshgrid(np.clip(xs, 1e-12, case.P - 1e-12), ys + 1e-12, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        for m, md in zip(r.modes, modes):
            E = np.asarray(m.field(pts, "E"), dtype=complex).reshape(nx, ny, 3)
            E = E / max(float(np.nanmax(np.abs(E))), 1e-300)                     # normalised to max |E| = 1
            maps[md["m"]] = dict(x_nm=xs * 1e9, y_nm=ys * 1e9, E=E.astype(np.complex64), omega=float(m.omega.real), lam_nm=md["lam_nm"], S_inc=1.0,
                                 eps_names=np.array(case.names), eps=np.array([complex(*res["eps"][n]) for n in case.names]), theta=theta, phi=phi,
                                 pol=f"Mode {md['m']}, Q = {md['Q']:.3g}", kx=kx, period_nm=case.P * 1e9, Q=md["Q"], omega_im=float(m.omega.imag))
    return res, maps


# --------------------------------------------------------------------------------------------------------------------- check
def version_info(hpfem):
    try:
        from hpfem.run import version_info as vi
        return vi()
    except Exception:
        return {"hpfem": str(getattr(hpfem, "__version__", "?"))}


def memory_estimate(case, order, solver):
    """hpfem.grating.estimate_memory (M15 F9) as a dict, None if this hpfem has none."""
    try:
        import hpfem.grating as grating
        est = grating.estimate_memory(case.mesh, order, backend_enum(case.h, solver))
    except Exception:
        return None
    out = {k: int(getattr(est, k)) for k in ("dofs", "matrix_nonzeros", "factor_entries", "matrix_bytes", "factor_bytes", "total_bytes") if hasattr(est, k)}
    out["backend"] = str(getattr(est, "backend", "")).split(".")[-1]
    try:
        out["text"] = str(est.describe())
    except Exception:
        pass
    return out


def point_values(job, v):
    """(wavelength, theta, phi) of the sweep value v."""
    inc, sw = job["incidence"], job["sweep"]
    lam = v if sw["mode"] == "wavelength" else inc["wavelength_nm"]
    th = v if sw["mode"] == "theta" else inc["theta"]
    ph = v if sw["mode"] == "phi" else inc["phi"]
    return lam, th, ph


def check_job(hpfem, case, job, folder):
    """Diagnostics of the library (hpfem.grating.validate, M15 F7) at the first and the last sweep value and the memory estimate (M15 F9),
    without solving. Writes <job>/check.json."""
    sw, inc, solver = job["sweep"], job["incidence"], job["solver"]
    values = sw["values"] if sw["mode"] != "none" else [None]
    p = int(solver["adaptive"]["p0"]) if solver.get("engine", "").endswith("_hp") else int(solver["order"])
    out = dict(version_info=version_info(hpfem), cells=int(case.mesh.num_cells), order=p, estimate=memory_estimate(case, p, solver))
    if case.isolated:
        out["diagnostics"] = None
        out["note"] = "Isolierte Struktur: die Prüfungen der Bibliothek (hpfem.grating.validate) gelten für periodische Zellen; nur die Speicherschätzung."
    elif has_grating(hpfem):
        import hpfem.grating as grating
        diags = []
        for v in [values[0]] + ([values[-1]] if len(values) > 1 else []):
            lam, th, ph = point_values(job, v)
            where = f"λ = {lam:g} nm, θ = {th:g}°, φ = {ph:g}°"
            try:
                ctx = prepare(case, lam)
                found = grating.validate(case.mesh, {1 + i: ctx.mat[n] for i, n in enumerate(case.names)}, ctx.stack, "s" if inc["pol"] == "TE" else "p",
                                         np.radians(th), np.radians(ph), ctx.omega, order=p, **grating_options(case, ctx, solver))
                diags += [dict(d, where=where) for d in diag_list(found)]
            except Exception as exc:
                diags.append(dict(code="setup", severity="error", text=str(exc), hint="", where=where))
        try:                                                       # grazing orders at every sweep value (cheap: no mesh checks)
            from hpfem import diagnostics as dg
            for v in values[1:-1]:
                lam, th, ph = point_values(job, v)
                ctx = prepare(case, lam)
                kx, _ = bloch_wavenumbers(ctx, th, ph)
                n_sub = float(ctx.n_sub.real) if ctx.lossless_sub and case.pml_bottom > 0 else None
                found = dg.validate_orders(ctx.k0, kx, ctx.n_cover, n_sub, case.P, int(solver.get("orders_max", 3)))
                diags += [dict(d, where=f"λ = {lam:g} nm, θ = {th:g}°, φ = {ph:g}°") for d in diag_list(found)]
        except Exception:
            pass
        seen, out["diagnostics"] = set(), []
        for d in diags:                                            # the same finding at both ends of the sweep once
            if (d["code"], d["text"]) not in seen:
                seen.add((d["code"], d["text"]))
                out["diagnostics"].append(d)
    else:
        out["diagnostics"] = None
        out["note"] = "Dieses hpfem hat kein hpfem.grating (vor 0.4): keine Prüfungen der Bibliothek."
    (Path(folder) / "check.json").write_text(json.dumps(out, default=str, ensure_ascii=False), encoding="utf-8")
    for d in out["diagnostics"] or []:
        say(f"DIAG {d['severity']} {d['code']}: {d['text']}")
    say("DONE")
    return 0


# ---------------------------------------------------------------------------------------------------------------------- main
def report_diagnostics(res, printed):
    """DIAG lines for the findings of the library that were not reported before."""
    for d in res.get("diagnostics") or []:
        key = (d["code"], d["text"])
        if key not in printed:
            printed.add(key)
            say(f"DIAG {d['severity']} {d['code']}: {d['text']}" + (f" | {d['hint']}" if d.get("hint") else ""))


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    folder = Path(argv[0])
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    Control.folder = folder
    Cancelled = getattr(hpfem, "Cancelled", ())
    case = Case(hpfem, job, folder)
    if check:
        return check_job(hpfem, case, job, folder)
    inc, sw, solver = job["incidence"], job["sweep"], job["solver"]
    task = job.get("task", "scattering")
    engine = solver.get("engine", "conical")
    if case.isolated:
        engine = "isolated"
        solver = dict(solver, engine="isolated")
    if (engine.startswith("grating") or task == "resonances") and not has_grating(hpfem):
        say("ERROR Dieses hpfem hat kein hpfem.grating (hp-FEM 0.4 oder neuer nötig): Löser „konisch (klassisch)“ wählen oder hpfem aktualisieren")
        return 1
    values = sw["values"] if sw["mode"] != "none" else [None]
    n = len(values)
    maps = job.get("maps", {"enabled": False})
    map_idx = set(maps.get("indices", [])) if maps.get("enabled") else set()
    vi = version_info(hpfem)
    say(f"hpfem {vi.get('hpfem', '?')}" + (f", backends {', '.join(vi['backends'])}" if vi.get("backends") else "") +
        (f", {vi['threads']} threads" if vi.get("threads") else ""))
    p_est = int(solver["adaptive"]["p0"]) if engine.endswith("_hp") else int(solver["order"])
    est = memory_estimate(case, p_est, solver)
    if est:
        say(f"ESTIMATE {est.get('dofs', 0)} DoFs, {est.get('total_bytes', 0) / 2 ** 20:.0f} MiB (p = {p_est}, {est.get('backend', '')})")
    out = dict(meta=dict(model=job["model"].get("name", ""), mode=sw["mode"], order=solver["order"], pol=inc["pol"], cells=int(case.mesh.num_cells),
                         engine=engine, task=task, started=time.strftime("%Y-%m-%d %H:%M:%S"), version_info=vi, estimate=est), points=[], pscan=[])
    results_path = folder / "results.json"
    printed = set()

    def flush():
        results_path.write_text(json.dumps(out, default=float), encoding="utf-8")

    def stop():
        say("CANCELLED")
        out["cancelled"] = True
        flush()
        return 0

    if task == "resonances":
        ro = job.get("resonance", {})
        for i, v in enumerate(values):
            if Control.cancelled():
                return stop()
            lam, th, ph = point_values(job, v)
            say(f"PROGRESS {i + 1}/{n} target lambda = {lam:.2f} nm, theta = {th:.2f}, phi = {ph:.2f}")
            try:
                res, mmaps = resonance_point(case, lam, th, ph, solver, ro, maps if i in map_idx else None)
                res["value"], res["index"] = v, i
                out["points"].append(res)
                for m, mp in mmaps.items():
                    np.savez_compressed(folder / f"mode_{i}_{m}.npz", **mp)
                say(f"  {res['dofs']} DoFs, {res['time_s']:.1f} s: " + "; ".join(f"λ = {md['lam_nm']:.2f} nm, Q = {md['Q']:.3g}" for md in res["modes"]))
            except Cancelled:
                return stop()
            except Exception as exc:
                say(f"ERROR point {i}: {exc}")
                out["points"].append(dict(index=i, value=v, error=str(exc)))
            flush()
        say("DONE")
        return 0

    pscan = job.get("pscan")
    if pscan:                                                      # convergence in the polynomial order at the first sweep value
        lam, th, ph = point_values(job, values[0])
        scan_engine = "isolated" if case.isolated else ("grating" if engine.startswith("grating") else ("conical" if engine == "conical" else "inplane"))
        for k, p in enumerate(pscan):
            if Control.cancelled():
                return stop()
            say(f"PROGRESS {k + 1}/{len(pscan)} order p = {p}")
            try:
                res, _, _, _ = solve_point(case, lam, th, ph, inc["pol"], p, dict(solver, engine=scan_engine, jacobian=False))
                out["pscan"].append(res)
                if res.get("isolated"):
                    say(f"  p = {p}: {res['dofs']} DoFs, {res['time_s']:.1f} s" + (f", sigma_sca = {res['sigma_sca']:.6e} m" if res.get("sigma_sca") is not None else "")
                        + "".join(f", {d['name']}: {d['P_down']:.6e} W/m" for d in res["detectors"] if d["P_down"] is not None))
                else:
                    say(f"  p = {p}: {res['dofs']} DoFs, {res['time_s']:.1f} s, R = {res['R']:.6f}" + (f", T = {res['T']:.6f}" if res["T"] is not None else ""))
            except Cancelled:
                return stop()
            except Exception as exc:
                say(f"ERROR p = {p}: {exc}")
                out["pscan"].append(dict(order=p, error=str(exc)))
            flush()
        say("DONE")
        return 0

    for i, v in enumerate(values):
        if Control.cancelled():
            return stop()
        lam, th, ph = point_values(job, v)
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
            report_diagnostics(res, printed)
            if res.get("isolated"):
                msg = f"  {res['dofs']} DoFs, {res['time_s']:.1f} s"
                if res.get("sigma_sca") is not None:
                    msg += f": sigma_sca = {res['sigma_sca'] * 1e9:.5g} nm"
                if res.get("sigma_ext") is not None:
                    msg += f", sigma_abs = {res['sigma_abs'] * 1e9:.5g} nm, sigma_ext = {res['sigma_ext'] * 1e9:.5g} nm"
                if res.get("mie"):
                    msg += f" | Mie: {res['mie']['sigma_sca'] * 1e9:.5g} / {res['mie']['sigma_abs'] * 1e9:.5g} / {res['mie']['sigma_ext'] * 1e9:.5g} nm"
                for d in res["detectors"]:
                    if d["P_down"] is not None:
                        msg += f" | {d['name']}: {d['P_down']:.6e} W/m (normiert {d['normalised']:.6f})"
                say(msg)
                flush()
                continue
            msg = f"  {res['dofs']} DoFs, {res['time_s']:.1f} s: R = {res['R']:.6f}"
            if res["T"] is not None:
                msg += f", T = {res['T']:.6f}, A = {res['A']:.6f}"
            if res.get("scalar"):
                msg += " (skalarer E_z-Pfad)"
            say(msg)
            if res.get("absorbed"):
                msg2 = ", ".join(f"{k} {v:.4f}" for k, v in res["absorbed"]["by_material"].items())
                say(f"  absorbed (exact, quadrature on the mesh): {msg2 or 'none'} | sum {res['absorbed']['total']:.4f}")
            if res["T"] is not None and abs(res["A"]) > 0 and res["A"] < -1e-3:
                say(f"WARN negative Absorption {res['A']:.2e}: Netz oder PML zu grob")
        except Cancelled:
            return stop()
        except Exception as exc:
            say(f"ERROR point {i}: {exc}")
            out["points"].append(dict(index=i, value=v, error=str(exc)))
        flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
