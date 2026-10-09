"""Solver worker for bodies of revolution (resonators and emitters): runs in the Python that has hpfem, called by the app.

    python fem_axi_worker.py <job folder>

Reads <job>/job.json (model of fem_axi, task, solver settings) and <job>/mesh.msh (meridian mesh x = r, y = z of fem_axi.build_mesh) and
  task "resonance": the quasi-normal modes of azimuthal order m nearest the target wavelength (hpfem.AxisymmetricResonance, cylindrical
                    PML): omega, wavelength, Q, residual; the mode fields on the subdivided mesh (mode_<k>.npz)
  task "emitter":   a Gaussian-smeared point dipole on the axis (hpfem.axisymmetric_gaussian_dipole; transverse: orders m = +-1, axial: m = 0)
                    over a wavelength sweep (hpfem.AxisymmetricScattering with the current as source): Purcell factor F_P = P / P_bulk
                    (power through the closed surface around the emitter box over the Larmor power of the smeared dipole in the bulk medium
                    of the emitter), the fractions through the plane above (beta) and below the structure; optionally the sweep is centred
                    on the resonance found first (+- k linewidths); the fields at the chosen points (field_<i>.npz).
  task "scattering": a plane wave from the homogeneous background at the angle theta to the axis (S or P), expanded in the orders
                    m = 0, +-1, ... (hpfem.oblique_plane_wave, scatter_orders until the pair +-m carries less than tol of the power): the
                    scattering cross-section from the scattered power through the closed measurement box, the extinction from the optical
                    theorem on the far field in the forward direction (axisymmetric_far_field, superpose_far_field), absorption = extinction -
                    scattering; the differential cross-section in the planes y = 0 and x = 0; the near field in the plane of incidence
                    (scatter_<i>.npz); for a single sphere the Mie series as reference.
  emitter option "modal": the Purcell spectrum also as a sum over the quasi-normal modes (hpfem.AxisymmetricRieszProjection, PML frozen at the
                    target wavelength), with the share of the chosen resonance and the background.
writes <job>/results.json after every point.

Progress lines: PROGRESS i/n ...  PHASE name  WARN ...  ERROR ...  CANCELLED  DONE.  Cancellation: <job>/cancel, between the points.
Field export: hpfem.FieldExporter2D (exact on the subdivided cells) parsed from its ASCII VTU; E = (E_r, E_phi, E_z) with E_phi = i v / r from
the scaled azimuthal unknown v = -i r E_phi.
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_axi as fa  # noqa: E402
import fem_materials as fm  # noqa: E402

NM = 1e-9


def say(*a):
    print(*a, flush=True)


class Job:
    def __init__(self, hpfem, job, folder):
        self.h, self.job, self.folder = hpfem, job, Path(folder)
        self.model = job["model"]
        self.names = list(self.model["materials"])
        self.solver = job["solver"]
        self.lay = fa.layout(self.model)
        t0 = time.time()
        self.mesh = hpfem.read_gmsh(str(self.folder / "mesh.msh"), NM, 2)
        self.tags = sorted({int(self.mesh.cell_tag(c)) for c in range(self.mesh.num_cells)})
        say(f"mesh read in {time.time() - t0:.1f} s: {self.mesh.num_cells} cells, tags {self.tags}")
        p = int(self.solver.get("order", 3))
        self.order = p
        self.nd = hpfem.NedelecDofMap2D(self.mesh, p)
        self.h1 = hpfem.DofMap2D(self.mesh, p)
        self.dofs = int(self.nd.num_dofs + self.h1.num_dofs)

    def cancelled(self):
        return (self.folder / "cancel").exists()

    def materials(self, lam_nm):
        """MaterialMap at lam_nm for every tag of the mesh (the emitter box carries 101 + index with the material of 1 + index); the background
        of the map is the surrounding medium (the contrast source of the scattered-field formulation is measured against it)."""
        e_bg = fm.eps_at(self.model["materials"][self.model["background"]], lam_nm)
        mm = self.h.MaterialMap(self.h.Material(complex(e_bg)))
        eps = {}
        for t in self.tags:
            name = self.names[(t - 1) % fa.TAG_SOURCE]
            e = fm.eps_at(self.model["materials"][name], lam_nm)
            eps[name] = e
            mm.set(int(t), self.h.Material(complex(e)))
        return mm, eps

    def pml(self, k0):
        h, lay = self.h, self.lay
        n_ref = float(np.sqrt(fm.eps_at(self.model["materials"][self.model["background"]], 2 * np.pi / k0 / NM) + 0j).real)
        profile = h.PmlProfile(2, float(self.solver.get("pml_reflection", 1e-10)))
        return h.PmlBox2D([0.0, lay["z_lo"] * NM], [lay["r_in"] * NM, lay["z_hi"] * NM],
                          [0.0, lay["pml"] * NM, lay["pml"] * NM, lay["pml"] * NM], k0, max(n_ref, 1.0), profile)

    def backend(self, setup):
        name = self.solver.get("backend", "AUTO")
        if name and name != "AUTO":
            try:
                setup.solver = getattr(self.h.DirectSolverBackend, name)
            except AttributeError:
                pass


# -------------------------------------------------------------------------------------------------------------- fields
def export_field(job, meridian, azimuthal, subdivisions, normalise=True):
    """(points [nm], simplices, E (n, 3) = (E_r, E_phi, E_z), tag per simplex) from hpfem.FieldExporter2D on the subdivided mesh, only the
    cells inside the PML box (the PML cells carry the stretched field)."""
    h = job.h
    ex = h.FieldExporter2D(job.mesh, int(subdivisions), h.VtkFormat.ASCII)
    ex.hcurl("E", job.nd, np.asarray(meridian))
    ex.h1("v", job.h1, np.asarray(azimuthal))
    root = ET.fromstring(ex.to_string())
    arrays = {da.attrib.get("Name"): da for da in root.iter("DataArray")}

    def arr(name, dtype=float):
        return np.array((arrays[name].text or "").split(), dtype=dtype)

    pts = arr("Points").reshape(-1, 3)[:, :2]
    conn = arr("connectivity", np.int64).reshape(-1, 3)
    tags = arr("cell_tag", np.int64) if "cell_tag" in arrays else np.zeros(len(conn), np.int64)
    E = (arr("E_re") + 1j * arr("E_im")).reshape(-1, 3)
    v = arr("v_re") + 1j * arr("v_im")
    r = pts[:, 0]
    tiny = 1e-6 * max(r.max(), 1e-30)
    Ephi = np.where(r > tiny, 1j * v / np.where(r > tiny, r, 1.0), np.nan)
    on_axis = np.where(~(r > tiny))[0]
    if len(on_axis):                                                 # the limit r -> 0: the value of the nearest point off the axis
        off = np.where(r > tiny)[0]
        for k in on_axis:
            j = off[np.argmin(np.abs(pts[off, 1] - pts[k, 1]) + np.abs(r[off]))]
            Ephi[k] = Ephi[j]
    field = np.column_stack([E[:, 0], Ephi, E[:, 1]])
    lay = job.lay
    tol = 1e-9 * lay["r_out"] * NM
    P = pts[conn]
    keep = (P[:, :, 0] <= lay["r_in"] * NM + tol).all(axis=1) & (P[:, :, 1] >= lay["z_lo"] * NM - tol).all(axis=1) & \
           (P[:, :, 1] <= lay["z_hi"] * NM + tol).all(axis=1)
    conn, tags = conn[keep], tags[keep]
    used = np.unique(conn)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    field = field[used]
    scale = (np.nanmax(np.abs(field)) or 1.0) if normalise else 1.0
    return dict(points_nm=pts[used] / NM, simplices=remap[conn].astype(np.int32), E=(field / scale).astype(np.complex64), tag=tags.astype(np.int32),
                scale=float(scale))


# ------------------------------------------------------------------------------------------------------------ resonances
def resonances(job, lam_nm, m, num_modes, krylov=0):
    """Modes of order m nearest lam_nm: list of dicts and the raw modes."""
    h = job.h
    k0 = 2 * np.pi / (lam_nm * NM)
    mm, _ = job.materials(lam_nm)
    setup = h.AxisymmetricResonanceSetup()
    setup.target_omega = k0 * h.constants.c0
    setup.materials = mm
    setup.axis_tag = fa.TAG_AXIS
    setup.pec_tags = [fa.TAG_WALL]
    setup.azimuthal_order = int(m)
    setup.pml = job.pml(k0)
    setup.num_modes = int(num_modes)
    setup.krylov_dimension = int(krylov) if krylov else max(40, 4 * int(num_modes) + 20)
    job.backend(setup)
    say("PHASE eigensolve")
    t0 = time.time()
    job.last_problem = h.AxisymmetricResonance(job.nd, job.h1, setup)
    raw = job.last_problem.solve()
    dt = time.time() - t0
    modes = []
    for k, md in enumerate(raw):
        w = complex(md.omega)
        modes.append(dict(k=k, lam_nm=float(md.wavelength) / NM, Q=float(md.quality), omega=[w.real, w.imag], residual=float(md.residual), m=int(m)))
    return modes, raw, dt


def pick_fundamental(modes, lam_nm, window=0.03):
    """The highest-Q mode within +-window of lam_nm (as the micropillar example of hp-FEM), else the nearest with Q > 0."""
    good = [md for md in modes if md["Q"] > 0]
    near = [md for md in good if abs(md["lam_nm"] - lam_nm) < window * lam_nm]
    if near:
        return max(near, key=lambda md: md["Q"])
    return min(good, key=lambda md: abs(md["lam_nm"] - lam_nm)) if good else None


# --------------------------------------------------------------------------------------------------------------- emitter
def plane_surface(job, z_nm, direction):
    """Horizontal plane z = z_nm for r < r_in with the normal along +z (direction +1, cell below) or -z (cell above)."""
    h, mesh = job.h, job.mesh
    V = np.asarray(mesh.vertices)
    zc = z_nm * NM
    r_max = job.lay["r_in"] * NM
    tol = 1e-7 * max(r_max, abs(zc), 1e-9)
    centroids = np.asarray(mesh.cell_centroids)
    facets = []
    for f in range(mesh.num_facets):
        v = V[mesh.facet_vertices(f)]
        if np.all(np.abs(v[:, 1] - zc) < tol) and np.all(v[:, 0] < r_max + tol):
            cells = [c for c in mesh.facet_cells(f) if c >= 0]
            if not cells:
                continue
            pick = min(cells, key=lambda c: centroids[c][1]) if direction > 0 else max(cells, key=lambda c: centroids[c][1])
            facets.append(h.Surface2D.Facet(f, pick))
    s = h.Surface2D()
    s.facets = facets
    return s


def emitter_point(job, lam_nm, surfaces, em, want_field=False):
    """One wavelength: emitted power, Purcell factor, fractions through the planes (and the field)."""
    h = job.h
    omega = 2 * np.pi * h.constants.c0 / (lam_nm * NM)
    k0 = 2 * np.pi / (lam_nm * NM)
    mm, eps = job.materials(lam_nm)
    axial = em["orientation"] == "axial"
    m = 0 if axial else 1
    sigma = float(em["sigma_nm"]) * NM
    setup = h.AxisymmetricScatteringSetup()
    setup.omega = omega
    setup.materials = mm
    setup.axis_tag = fa.TAG_AXIS
    setup.pec_tags = [fa.TAG_WALL]
    setup.azimuthal_order = m
    setup.pml = job.pml(k0)
    setup.current = h.axisymmetric_gaussian_dipole(float(em["z_nm"]) * NM, 1.0, h.AxisDipole.AXIAL if axial else h.AxisDipole.TRANSVERSE,
                                                   sigma, omega, m)
    setup.extra_quadrature_order = int(job.solver.get("extra_quadrature_order", 6))
    job.backend(setup)
    t0 = time.time()
    field = h.AxisymmetricScattering(job.nd, job.h1, setup).solve()
    t_solve = time.time() - t0
    factor = 1.0 if axial else 2.0                                    # orders +1 and -1 radiate equally

    def flux(surface):
        return factor * float(h.axisymmetric_poynting_flux(job.nd, job.h1, field.meridian, field.azimuthal, m, omega, mm, surface))

    total = flux(surfaces["around"])
    top = flux(surfaces["top"]) if surfaces.get("top") is not None and len(surfaces["top"]) else 0.0
    bottom = flux(surfaces["bottom"]) if surfaces.get("bottom") is not None and len(surfaces["bottom"]) else 0.0
    n_em = float(np.sqrt(eps[surfaces["emitter_material"]] + 0j).real)
    bulk = n_em * float(h.dipole_vacuum_power(1.0, omega)) * np.exp(-((n_em * k0 * sigma) ** 2))
    res = dict(lam_nm=lam_nm, P_total=total, P_top=top, P_bottom=bottom, P_bulk=bulk, purcell=total / bulk,
               purcell_radiative=(top + bottom) / bulk, beta_top=top / total if total else None, beta_bottom=bottom / total if total else None,
               n_emitter=n_em, dofs=job.dofs, time_s=t_solve)
    fld = export_field(job, field.meridian, field.azimuthal, job.job.get("subdivisions", 2)) if want_field else None
    if fld is not None:
        fld.update(lam_nm=lam_nm, m=m, kind="emitter", purcell=res["purcell"])
    return res, fld


def box_surface(job):
    """The closed measurement box r <= r_plane, z_plane_bottom <= z <= z_plane_top (mesh lines), normals outwards (cells inside)."""
    h, mesh, lay = job.h, job.mesh, job.lay
    V = np.asarray(mesh.vertices)
    centroids = np.asarray(mesh.cell_centroids)
    rp, zb, zt = lay["r_plane"] * NM, lay["z_plane_bottom"] * NM, lay["z_plane_top"] * NM
    tol = 1e-7 * max(rp, abs(zb), abs(zt), 1e-9)
    facets = []
    for f in range(mesh.num_facets):
        v = V[mesh.facet_vertices(f)]
        on_top = np.all(np.abs(v[:, 1] - zt) < tol) and np.all(v[:, 0] <= rp + tol)
        on_bottom = np.all(np.abs(v[:, 1] - zb) < tol) and np.all(v[:, 0] <= rp + tol)
        on_side = np.all(np.abs(v[:, 0] - rp) < tol) and np.all((v[:, 1] >= zb - tol) & (v[:, 1] <= zt + tol))
        if not (on_top or on_bottom or on_side):
            continue
        cells = [c for c in mesh.facet_cells(f) if c >= 0]
        inside = [c for c in cells if centroids[c][0] < rp and zb < centroids[c][1] < zt]
        if inside:
            facets.append(h.Surface2D.Facet(f, inside[0]))
    s_ = h.Surface2D()
    s_.facets = facets
    return s_


def mie(lam_nm, radius_nm, n_sphere, n_bg, lmax=None):
    """Mie cross-sections [m^2] of a sphere (Bohren & Huffman 4.61, 4.62; exp(-i omega t), Im n > 0 lossy)."""
    import scipy.special as sp

    k = 2 * np.pi * n_bg / (lam_nm * NM)
    a = radius_nm * NM
    x, m = k * a, complex(n_sphere) / n_bg
    lmax = lmax or int(x + 4 * x ** (1 / 3) + 10)
    ls = np.arange(1, lmax + 1)

    def psi(z):
        return z * sp.spherical_jn(ls, z)

    def dpsi(z):
        return sp.spherical_jn(ls, z) + z * sp.spherical_jn(ls, z, derivative=True)

    def xi(z):
        return z * (sp.spherical_jn(ls, z) + 1j * sp.spherical_yn(ls, z))

    def dxi(z):
        return (sp.spherical_jn(ls, z) + 1j * sp.spherical_yn(ls, z)) + z * (sp.spherical_jn(ls, z, derivative=True) + 1j * sp.spherical_yn(ls, z, derivative=True))

    mx = m * x
    an = (m * psi(mx) * dpsi(x) - psi(x) * dpsi(mx)) / (m * psi(mx) * dxi(x) - xi(x) * dpsi(mx))
    bn = (psi(mx) * dpsi(x) - m * psi(x) * dpsi(mx)) / (psi(mx) * dxi(x) - m * xi(x) * dpsi(mx))
    c = 2 * np.pi / k ** 2
    sca = c * float(np.sum((2 * ls + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)))
    ext = c * float(np.sum((2 * ls + 1) * (an + bn).real))
    return dict(sigma_sca=sca, sigma_ext=ext, sigma_abs=ext - sca)


def single_sphere(model):
    """(radius, material) if the model is one sphere centred on the axis in the background without substrate, else None."""
    parts = model["parts"]
    if len(parts) == 1 and parts[0]["type"] == "sphere" and not model.get("substrate"):
        return float(parts[0]["radius"]), parts[0]["material"]
    return None


def scattering_point(job, lam_nm, surface, want_field=False, theta_samples=181):
    """Plane wave at lam_nm: cross-sections, far-field pattern, near field."""
    h, model = job.h, job.model
    sc = model["scattering"]
    omega = 2 * np.pi * h.constants.c0 / (lam_nm * NM)
    mm, eps = job.materials(lam_nm)
    n_bg = float(np.sqrt(eps.get(model["background"], fm.eps_at(model["materials"][model["background"]], lam_nm)) + 0j).real)
    k0 = 2 * np.pi / (lam_nm * NM)
    k = n_bg * k0
    th = np.radians(float(sc["theta_deg"]))
    pol = h.PlanePolarisation.S if sc["pol"] == "S" else h.PlanePolarisation.P
    setup = h.AxisymmetricScatteringSetup()
    setup.omega = omega
    setup.materials = mm
    setup.axis_tag = fa.TAG_AXIS
    setup.pec_tags = [fa.TAG_WALL]
    setup.pml = job.pml(k0)
    setup.extra_quadrature_order = int(job.solver.get("extra_quadrature_order", 4))
    job.backend(setup)
    t0 = time.time()
    res = h.scatter_orders(job.nd, job.h1, setup, lambda m: h.oblique_plane_wave(1.0, k, th, pol, m), int(sc.get("max_order", 8)), surface,
                           float(sc.get("tol", 1e-5)))
    t_solve = time.time() - t0
    orders = [int(m) for m in res.orders]
    intensity = n_bg / (2 * h.constants.Z0)                              # |E0| = 1 V/m in the background
    sigma_sca = float(res.total_power()) / intensity
    thetas = sorted(set(np.linspace(0.0, np.pi, theta_samples).tolist()) | {float(th)})
    patterns = [h.axisymmetric_far_field(job.nd, job.h1, f.meridian, f.azimuthal, m, omega, mm, surface, thetas) for m, f in zip(orders, res.fields)]
    fwd = h.superpose_far_field(patterns, orders, 0.0)
    i_th = int(np.argmin(np.abs(np.asarray(thetas) - th)))
    F_e = complex(np.asarray(fwd.f_phi)[i_th]) if sc["pol"] == "S" else complex(np.asarray(fwd.f_theta)[i_th])
    sigma_ext = 4 * np.pi / k * F_e.imag
    out = dict(lam_nm=lam_nm, sigma_sca=sigma_sca, sigma_ext=sigma_ext, sigma_abs=sigma_ext - sigma_sca, orders=orders,
               power_by_order=[float(v) / intensity for v in res.power], n_bg=n_bg, theta_deg=float(sc["theta_deg"]), dofs=job.dofs, time_s=t_solve)
    pattern = {"pattern_theta": list(thetas)}
    for key, phi in (("xz_0", 0.0), ("xz_pi", np.pi), ("yz_0", np.pi / 2), ("yz_pi", 3 * np.pi / 2)):
        F = h.superpose_far_field(patterns, orders, phi)
        pattern[f"pattern_{key}"] = (np.abs(np.asarray(F.f_theta)) ** 2 + np.abs(np.asarray(F.f_phi)) ** 2).tolist()   # dsigma/dOmega, |E0| = 1
    out.update(pattern)
    sph = single_sphere(model)
    if sph:
        try:
            n_s = complex(np.sqrt(fm.eps_at(model["materials"][sph[1]], lam_nm) + 0j))
            out["mie"] = mie(lam_nm, sph[0], n_s, n_bg)
        except Exception as exc:
            say(f"WARN Mie-Referenz nicht berechnet: {exc}")
    fld = None
    if want_field:
        right = left = None
        for m, f in zip(orders, res.fields):
            e = export_field(job, f.meridian, f.azimuthal, job.job.get("subdivisions", 2), normalise=False)
            E = e["E"].astype(complex)                                 # (E_r, E_phi, E_z) of order m at phi = 0
            r_ = np.column_stack([E[:, 0], E[:, 1], E[:, 2]])          # Cartesian at phi = 0: (E_x, E_y, E_z)
            l_ = np.column_stack([-E[:, 0], -E[:, 1], E[:, 2]]) * (-1.0) ** m   # at phi = pi: e^{i m pi}, e_r = -x, e_phi = -y
            right = r_ if right is None else right + r_
            left = l_ if left is None else left + l_
            pts, simp = e["points_nm"], e["simplices"]
        fld = dict(points_nm=pts, simplices=simp, E_right=right.astype(np.complex64), E_left=left.astype(np.complex64), lam_nm=lam_nm,
                   theta_deg=float(sc["theta_deg"]), pol=sc["pol"], k_bg=k, kind="scatter")
    return out, fld


def modal_purcell(job, em, values, modes_raw, problem, chosen_index, m):
    """Purcell factor over `values` [nm] as a sum over the quasi-normal modes (Riesz projection on the resonance pencil): (total, share of the
    chosen mode, background) per wavelength and the convergence of the contours. As examples/micropillar_qd of hp-FEM."""
    h = job.h
    omegas = [2 * np.pi * h.constants.c0 / (lam * NM) for lam in values]
    rs = h.RieszSetup()
    rs.poles = [r.omega for r in modes_raw]
    rs.omega_min, rs.omega_max = min(omegas), max(omegas)
    rs.points_per_pole, rs.background_points = 16, 40
    rs.background_aspect = 0.5
    riesz = h.AxisymmetricRieszProjection(problem, rs)
    axial = em["orientation"] == "axial"
    sigma = float(em["sigma_nm"]) * NM
    f = h.axisymmetric_gaussian_dipole(float(em["z_nm"]) * NM, 1.0, h.AxisDipole.AXIAL if axial else h.AxisDipole.TRANSVERSE, sigma, 1.0, m)
    source = riesz.add_current(lambda x: f(x) / (1j * h.constants.mu0))
    power = riesz.add_emitted_power(source)
    riesz.run()
    contours = riesz.contours
    spec = np.asarray(riesz.spectrum(source, power, omegas))
    chosen_omega = modes_raw[chosen_index].omega
    mode_row = next((i for i, c in enumerate(contours) if chosen_omega in c.poles), None)
    factor = 1.0 if axial else 2.0
    return spec, mode_row, factor, max(float(c.convergence) for c in contours), len(contours[-1].poles)


def emitter_surfaces(job):
    h = job.h
    em = job.model["emitter"]
    src_tags = [t for t in job.tags if t > fa.TAG_SOURCE]
    if len(src_tags) != 1:
        raise RuntimeError(f"Emitterkasten: erwartet genau ein Material, gefunden Tags {src_tags} (Netz mit der Aufgabe „Emitter“ neu erzeugen)")
    around = h.Surface2D.around_cells(job.mesh, int(src_tags[0]))
    lay = job.lay
    top = plane_surface(job, lay["z_plane_top"], +1)
    bottom = plane_surface(job, lay["z_plane_bottom"], -1)
    say(f"surfaces: around the emitter {len(around)} facets, top plane {len(top)}, bottom plane {len(bottom)}")
    if not len(top):
        say("WARN keine Facetten auf der oberen Messebene: β nicht messbar")
    return dict(around=around, top=top, bottom=bottom, emitter_material=job.names[(src_tags[0] - 1) % fa.TAG_SOURCE], z=float(em["z_nm"]))


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    jd = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    import hpfem
    hpfem.set_log_level("warn")
    try:
        from hpfem.run import version_info
        vi = version_info()
    except Exception:
        vi = {"hpfem": str(getattr(hpfem, "__version__", "?"))}
    say(f"hpfem {vi.get('hpfem', '?')}" + (f", backends {', '.join(vi.get('backends', []))}" if vi.get("backends") else ""))
    if not hasattr(hpfem, "AxisymmetricResonance"):
        say("ERROR Dieses hpfem hat keinen zylindersymmetrischen Löser (AxisymmetricResonance, hp-FEM ab 0.3)")
        return 1
    job = Job(hpfem, jd, folder)
    model, task = job.model, jd["task"]
    say(f"order p = {job.order}: {job.dofs} DoFs")
    out = dict(meta=dict(model=model.get("name", ""), task=task, order=job.order, cells=int(job.mesh.num_cells), dofs=job.dofs,
                         started=time.strftime("%Y-%m-%d %H:%M:%S"), version_info=vi), resonance=None, points=[])
    path = folder / "results.json"

    def flush():
        path.write_text(json.dumps(out, default=float), encoding="utf-8")

    sub = int(jd.get("subdivisions", 2))
    try:
        if task == "resonance":
            rs = model["resonance"]
            say(f"PROGRESS 1/1 modes of order m = {rs['m']} near {rs['wavelength_nm']:g} nm")
            modes, raw, dt = resonances(job, float(rs["wavelength_nm"]), int(rs["m"]), int(rs["num_modes"]), int(jd.get("krylov_dimension", 0)))
            out["resonance"] = dict(target_nm=float(rs["wavelength_nm"]), m=int(rs["m"]), modes=modes, time_s=dt)
            for md in modes:
                say(f"  mode {md['k']}: lambda = {md['lam_nm']:.3f} nm, Q = {md['Q']:.4g}, residual {md['residual']:.1e}")
            flush()
            if jd.get("fields", True):
                say("PHASE post")
                for md, r in zip(modes, raw):
                    fld = export_field(job, r.meridian, r.azimuthal, sub)
                    fld.update(lam_nm=md["lam_nm"], Q=md["Q"], m=md["m"], kind="mode")
                    np.savez_compressed(folder / f"mode_{md['k']}.npz", **fld)
            say(f"  {dt:.1f} s")
        elif task == "scattering":
            sc = model["scattering"]
            values = fa.sweep_values(sc["sweep"])
            surface = box_surface(job)
            say(f"measurement box: {len(surface)} facets; plane wave theta = {sc['theta_deg']:g} deg, {sc['pol']}, orders up to |m| = {sc['max_order']}")
            want = set(jd.get("field_indices", []))
            if jd.get("field_at_centre", True):
                want.add(len(values) // 2)
            n = len(values)
            for i, lam in enumerate(values):
                if job.cancelled():
                    say("CANCELLED")
                    out["cancelled"] = True
                    flush()
                    return 0
                say(f"PROGRESS {i + 1}/{n} lambda = {lam:.3f} nm")
                try:
                    res, fld = scattering_point(job, float(lam), surface, want_field=i in want)
                    res["index"] = i
                    out["points"].append(res)
                    if fld is not None:
                        np.savez_compressed(folder / f"scatter_{i}.npz", **fld)
                    msg = (f"  {res['time_s']:.1f} s, orders {res['orders']}: sigma_sca = {res['sigma_sca'] * 1e12:.5g} um^2, "
                           f"sigma_abs = {res['sigma_abs'] * 1e12:.5g} um^2, sigma_ext = {res['sigma_ext'] * 1e12:.5g} um^2")
                    if res.get("mie"):
                        mi = res["mie"]
                        msg += f" | Mie: {mi['sigma_sca'] * 1e12:.5g} / {mi['sigma_abs'] * 1e12:.5g} / {mi['sigma_ext'] * 1e12:.5g}"
                    say(msg)
                except Exception as exc:
                    say(f"ERROR point {i}: {exc}")
                    out["points"].append(dict(index=i, lam_nm=float(lam), error=str(exc)))
                flush()
        else:
            em = model["emitter"]
            sw = em["sweep"]
            if sw["mode"] == "resonance":
                rs = model["resonance"]
                m = 0 if em["orientation"] == "axial" else 1
                say(f"PROGRESS 0/1 resonance of order m = {m} near {rs['wavelength_nm']:g} nm (centre of the spectrum)")
                modes, raw, dt = resonances(job, float(rs["wavelength_nm"]), m, int(rs["num_modes"]), int(jd.get("krylov_dimension", 0)))
                best = pick_fundamental(modes, float(rs["wavelength_nm"]))
                out["resonance"] = dict(target_nm=float(rs["wavelength_nm"]), m=m, modes=modes, time_s=dt, chosen=best)
                if best is None:
                    raise RuntimeError("keine Resonanz mit Q > 0 gefunden: Zielwellenlänge oder Anzahl der Moden ändern")
                span = float(sw.get("linewidths", 1.5)) * best["lam_nm"] / best["Q"]
                values = list(np.linspace(best["lam_nm"] - span, best["lam_nm"] + span, max(int(sw["n"]), 1)))
                say(f"  resonance {best['lam_nm']:.3f} nm, Q = {best['Q']:.4g}: spectrum {values[0]:.3f} ... {values[-1]:.3f} nm")
                flush()
                modal = None
                if em.get("modal"):
                    say("PHASE riesz (modal expansion)")
                    try:
                        t0 = time.time()
                        spec, row, factor, conv, npoles = modal_purcell(job, em, values, raw, job.last_problem, best["k"], m)
                        modal = (spec, row, factor)
                        out["resonance"]["modal"] = dict(convergence=conv, poles=npoles, time_s=time.time() - t0)
                        say(f"  Riesz projection: {npoles} poles in the background contour, half-rule difference {conv:.1e}, {time.time() - t0:.1f} s")
                    except Exception as exc:
                        say(f"WARN Modenzerlegung fehlgeschlagen: {exc}")
            else:
                values = fa.sweep_values(sw)
                modal = None
                if em.get("modal"):
                    say("WARN die Modenzerlegung braucht das Spektrum „um die Resonanz“ (die Moden der Resonanzsuche); übersprungen")
            surfaces = emitter_surfaces(job)
            want = set(jd.get("field_indices", []))
            if jd.get("field_at_centre", True):
                want.add(len(values) // 2)
            n = len(values)
            for i, lam in enumerate(values):
                if job.cancelled():
                    say("CANCELLED")
                    out["cancelled"] = True
                    flush()
                    return 0
                say(f"PROGRESS {i + 1}/{n} lambda = {lam:.3f} nm")
                try:
                    res, fld = emitter_point(job, float(lam), surfaces, em, want_field=i in want)
                    res["index"] = i
                    if modal is not None:
                        spec, row, factor = modal
                        tot = factor * float(spec[:, i].sum().real) / res["P_bulk"]
                        res["modal"] = dict(total=tot, mode=factor * float(spec[row, i].real) / res["P_bulk"] if row is not None else None,
                                            background=factor * float(spec[-1, i].real) / res["P_bulk"])
                    out["points"].append(res)
                    if fld is not None:
                        np.savez_compressed(folder / f"field_{i}.npz", **fld)
                    say(f"  {res['time_s']:.1f} s: F_P = {res['purcell']:.4g}, beta_top = {res['beta_top']:.4f}, down = {res['beta_bottom']:.4f}")
                except Exception as exc:
                    say(f"ERROR point {i}: {exc}")
                    out["points"].append(dict(index=i, lam_nm=float(lam), error=str(exc)))
                flush()
    except Exception as exc:
        say(f"ERROR {exc}")
        out["error"] = str(exc)
        flush()
        return 1
    flush()
    say("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
